import torch
import torch.nn as nn
import comfy.model_management as mm
import comfy.sd
import comfy.controlnet
import comfy.utils
import gc
import os
import sys
import time
import copy
from comfy.model_patcher import ModelPatcher

# 安全导入 psutil
try:
    import psutil
except ImportError:
    psutil = None

# ==========================================================
# 0. 基础工具 (Utilities)
# ==========================================================
class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False
    def __eq__(self, __value: object) -> bool: return True

any_type = AnyType("*")

# ==========================================================
# 1. 全局配置 (Global Config)
# ==========================================================
class Shadow_Config:
    enabled = False 
    mode = "Ease Mode" 
    job_type = "Image (Aggressive)"
    shadow_mode = True 
    ram_reserve_gb = 4.0
    vram_reserve_mb = 1024.0
    verbose = True

# ==========================================================
# 2. 影子系统核心 (The Shadow Legion)
# ==========================================================
class ShadowGroup:
    """管理真实的加载过程，只有在需要时才触发"""
    def __init__(self, name, loader_func, *args, **kwargs):
        self.name = name
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self.is_loaded = False
        self.cached_tuple = None

    def _execute_load(self):
        if self.is_loaded: return
        if Shadow_Config.verbose: print(f"👻 [LaoLi Shadow] ⚡ 触发延迟加载: {self.name}")
        
        # 内存安全检查
        if psutil:
            try:
                mem = psutil.virtual_memory()
                if (mem.available / (1024**3)) < Shadow_Config.ram_reserve_gb:
                    if Shadow_Config.verbose: print(f"⚠️ [LaoLi Shadow] 系统内存吃紧 -> 触发GC")
                    gc.collect()
            except: pass

        start_t = time.time()
        # 执行原始加载
        self.cached_tuple = self.loader_func(*self.args, **self.kwargs)
        self.is_loaded = True
        if Shadow_Config.verbose: print(f"✨ [LaoLi Shadow] {self.name} 加载完毕 (耗时 {time.time()-start_t:.2f}s)")

    def get_real_thing(self, index):
        if not self.is_loaded: self._execute_load()
        if self.cached_tuple and isinstance(self.cached_tuple, (list, tuple)) and len(self.cached_tuple) > index:
            return self.cached_tuple[index]
        return None

class ShadowInnerModel(torch.nn.Module):
    """代理内部模型，防止 ModelPatcher 初始化报错"""
    def __init__(self, parent_patcher):
        super().__init__()
        self._laoli_parent = parent_patcher 
    
    def __getattr__(self, name):
        # 拦截内部属性访问，防止无限递归
        if name.startswith("_laoli") or name.startswith("training") or name.startswith("__"): 
             raise AttributeError(f"ShadowInnerModel missing: {name}")
        
        # 唤醒真身并获取属性
        real_patcher = self._laoli_parent._ensure_real()
        return getattr(real_patcher.model, name)

class ShadowPatcher(ModelPatcher):
    """伪装成 ModelPatcher，在此期间捕获所有 LoRA 和设置"""
    def __init__(self, group, *args, **kwargs):
        dummy = ShadowInnerModel(self)
        # 透传参数，兼容 ComfyUI 新版本
        super().__init__(dummy, torch.device("cpu"), torch.device("cpu"), *args, **kwargs)
        self._laoli_group = group
        self._laoli_real_obj = None
        self._laoli_is_shadow = True 

    def _ensure_real(self):
        if self._laoli_real_obj is None:
            # 假定 Checkpoint 返回的第一个是 Model
            real = self._laoli_group.get_real_thing(0)
            self.become_real(real)
        return self._laoli_real_obj

    def become_real(self, real_obj):
        if real_obj is None: return 
        
        # 1. 备份影子期间积累的补丁(LoRA)和参数
        preserved_patches = copy.deepcopy(getattr(self, "patches", {}))
        preserved_obj_patches = copy.deepcopy(getattr(self, "object_patches", {}))
        preserved_options = copy.deepcopy(getattr(self, "model_options", {}))
        
        self._laoli_real_obj = real_obj
        
        # 2. 暴力覆盖属性，变身为真身
        self.__dict__.update(real_obj.__dict__)
        
        # 3. 恢复补丁
        if preserved_patches: self.patches = preserved_patches
        if preserved_obj_patches: self.object_patches = preserved_obj_patches
        if preserved_options:
            current = getattr(self, "model_options", {})
            current.update(preserved_options)
            self.model_options = current

        # 4. 更改类指针
        try: self.__class__ = real_obj.__class__
        except: pass
        if hasattr(self, "_laoli_is_shadow"): del self._laoli_is_shadow
    
    def copy(self): return self.clone()
    def clone(self, *args, **kwargs):
        if self._laoli_real_obj: return self._laoli_real_obj.clone()
        # 创建新的影子副本
        new_shadow = ShadowPatcher(self._laoli_group)
        new_shadow.patches = copy.deepcopy(getattr(self, "patches", {}))
        new_shadow.object_patches = copy.deepcopy(getattr(self, "object_patches", {}))
        new_shadow.model_options = copy.deepcopy(getattr(self, "model_options", {}))
        return new_shadow

    def __getattr__(self, name):
        if name.startswith("_laoli"): raise AttributeError(name)
        real = self.__dict__.get("_laoli_real_obj", None)
        # 处理 pinned 等特殊属性
        if name == "pinned" and real is None: return set()
        if real is None: real = self._ensure_real()
        return getattr(real, name)

class ShadowGenericProxy:
    """通用的影子代理（用于 CLIP, VAE）"""
    def __init__(self, group, index):
        self._laoli_group = group
        self._laoli_index = index
        self._laoli_real_obj = None
    
    def _ensure_real(self):
        if self._laoli_real_obj is None:
            real = self._laoli_group.get_real_thing(self._laoli_index)
            self.become_real(real)
        return self._laoli_real_obj

    def become_real(self, real_obj):
        self._laoli_real_obj = real_obj
        self.__dict__.update(real_obj.__dict__)
        try: self.__class__ = real_obj.__class__
        except: pass

    def clone(self):
        if self._laoli_real_obj: return self._laoli_real_obj.clone()
        return ShadowGenericProxy(self._laoli_group, self._laoli_index)

    # 透传常见方法
    def decode(self, samples_in): return self._ensure_real().decode(samples_in)
    def encode(self, pixel_samples): return self._ensure_real().encode(pixel_samples)

    def __getattr__(self, name):
        if name.startswith("_laoli"): raise AttributeError(name)
        real = self._ensure_real()
        return getattr(real, name)

# ==========================================================
# 3. 劫持加载器 (Hooks)
# ==========================================================

# --- Checkpoint Loader 劫持 ---
if not hasattr(comfy.sd, "_laoli_org_load_ckpt"):
    comfy.sd._laoli_org_load_ckpt = comfy.sd.load_checkpoint_guess_config

def _hacked_load_checkpoint(*args, **kwargs):
    # 使用 *args 兼容所有版本
    if Shadow_Config.enabled and Shadow_Config.shadow_mode:
        # 尝试获取文件名用于日志
        try:
            ckpt_path = args[0] if len(args) > 0 else kwargs.get("ckpt_path", "Unknown")
            name = os.path.basename(ckpt_path)
        except: name = "Unknown"

        if Shadow_Config.verbose: print(f"💤 [LaoLi Shadow] 拦截 Checkpoint: {name}")
        
        group = ShadowGroup(name, comfy.sd._laoli_org_load_ckpt, *args, **kwargs)
        
        # 返回影子三剑客 (Model, CLIP, VAE)
        # 注意：如果原始函数返回4个值(clipvision)，这里只返回前3个影子，第4个会丢失。
        # 但通常 CheckpointLoaderSimple 只解包前3个。
        return (ShadowPatcher(group), ShadowGenericProxy(group, 1), ShadowGenericProxy(group, 2))
        
    return comfy.sd._laoli_org_load_ckpt(*args, **kwargs)

if hasattr(comfy.sd, "load_checkpoint_guess_config"):
    comfy.sd.load_checkpoint_guess_config = _hacked_load_checkpoint

# --- ControlNet Loader 劫持 ---
class ShadowControlNet(ModelPatcher):
    def __init__(self, name, loader):
        dummy = torch.nn.Module()
        super().__init__(dummy, torch.device("cpu"), torch.device("cpu"))
        self._laoli_is_shadow = True
        self._laoli_name = name
        self._laoli_loader = loader
        self._laoli_real = None
    
    def summon(self):
        if self._laoli_real: return self._laoli_real
        if Shadow_Config.verbose: print(f"👻 [LaoLi Shadow] 唤醒 ControlNet: {self._laoli_name}")
        self._laoli_real = self._laoli_loader()
        self.__dict__.update(self._laoli_real.__dict__)
        try: self.__class__ = self._laoli_real.__class__
        except: pass
        if hasattr(self, "_laoli_is_shadow"): del self._laoli_is_shadow
        return self._laoli_real

    def copy(self):
        if self._laoli_real: return self._laoli_real.copy()
        return ShadowControlNet(self._laoli_name, self._laoli_loader)
    
    def __getattr__(self, name):
        if name.startswith("_laoli"): raise AttributeError(name)
        real = self.__dict__.get("_laoli_real", None)
        if name == "pinned" and real is None: return set()
        self.summon()
        return getattr(self._laoli_real, name)

if not hasattr(comfy.controlnet, "_laoli_org_load_cn"):
    comfy.controlnet._laoli_org_load_cn = comfy.controlnet.load_controlnet

def _hacked_load_controlnet(*args, **kwargs):
    if Shadow_Config.enabled and Shadow_Config.shadow_mode:
        try:
            ckpt_path = args[0] if len(args) > 0 else kwargs.get("ckpt_path", "Unknown")
            name = os.path.basename(ckpt_path)
        except: name = "Unknown"
        
        if Shadow_Config.verbose: print(f"💤 [LaoLi Shadow] 拦截 ControlNet: {name}")
        return ShadowControlNet(name, lambda: comfy.controlnet._laoli_org_load_cn(*args, **kwargs))
    
    return comfy.controlnet._laoli_org_load_cn(*args, **kwargs)

if hasattr(comfy.controlnet, "load_controlnet"):
    comfy.controlnet.load_controlnet = _hacked_load_controlnet

# ==========================================================
# 4. 显存管理 (The Brain)
# ==========================================================
if not hasattr(mm, "_laoli_original_load_models_gpu"):
    mm._laoli_original_load_models_gpu = mm.load_models_gpu

def _shadow_load_models_gpu(models, memory_required=0, **kwargs):
    if Shadow_Config.enabled:
        try:
            device = mm.get_torch_device()
            # 1. 唤醒所有涉及的影子
            for model in models:
                if getattr(model, "_laoli_is_shadow", False):
                    if hasattr(model, "summon"): model.summon() 
                    if hasattr(model, "_ensure_real"): model._ensure_real()
            
            # 2. 显存策略
            if Shadow_Config.mode == "Ease Mode" and device.type == 'cuda':
                mm.soft_empty_cache()
                
                # 如果是图片模式，且显存要求较高，执行更激进的检查
                if "Image" in Shadow_Config.job_type:
                    try:
                        stats = torch.cuda.get_device_properties(device)
                        total_mem = stats.total_memory
                        reserved = torch.cuda.memory_reserved(device)
                        free_mem = total_mem - reserved
                        
                        needed = memory_required if memory_required > 0 else (1.5 * 1024**3)
                        reserve_bytes = Shadow_Config.vram_reserve_mb * 1024 * 1024
                        
                        # 策略：如果总显存 < 12GB (小显存卡) 或者 剩余空间严重不足
                        is_low_vram_card = total_mem < (12 * 1024**3)
                        is_critical = free_mem < (needed + reserve_bytes)
                        
                        if is_critical:
                            if is_low_vram_card:
                                if Shadow_Config.verbose: print("🧹 [LaoLi Shadow] 小显存保护 -> 深度清理 (Unload All)")
                                mm.unload_all_models()
                                mm.soft_empty_cache()
                                torch.cuda.empty_cache()
                            else:
                                if Shadow_Config.verbose: print("🧹 [LaoLi Shadow] 空间占用比例高 -> 智能腾挪")
                                mm.free_memory(needed + reserve_bytes, device)
                                mm.soft_empty_cache()
                    except: pass

        except Exception as e: 
            print(f"❌ [LaoLi Shadow Error] 显存管理异常: {e}")
    
    return mm._laoli_original_load_models_gpu(models, memory_required=memory_required, **kwargs)

mm.load_models_gpu = _shadow_load_models_gpu

# ==========================================================
# 5. 节点定义 (Nodes)
# ==========================================================

# --- 节点 1: 全局配置 ---
class LaoLi_Shadow_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}), 
                "job_type": (["Image (Aggressive)", "Video (Safe)"], {"default": "Image (Aggressive)"}), 
                "shadow_mode": ("BOOLEAN", {"default": True}),
                "mode": (["Ease Mode", "Monitor Mode"],),
                "ram_reserve_gb": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5}),
                "vram_reserve_mb": ("FLOAT", {"default": 512.0, "min": 0.0, "max": 8192.0, "step": 64.0}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "update_settings"
    CATEGORY = "LaoLi Shadow"
    DESCRIPTION = "影子系统控制台 - 必须连接在工作流中"
    OUTPUT_NODE = True # 标记为输出节点

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan") # 强制每次运行

    def update_settings(self, enable, job_type, shadow_mode, mode, ram_reserve_gb, vram_reserve_mb, verbose):
        Shadow_Config.enabled = enable
        Shadow_Config.job_type = job_type
        Shadow_Config.shadow_mode = shadow_mode
        Shadow_Config.mode = mode
        Shadow_Config.ram_reserve_gb = float(ram_reserve_gb)
        Shadow_Config.vram_reserve_mb = float(vram_reserve_mb)
        Shadow_Config.verbose = verbose
        
        if verbose:
            icon = "🖼️" if "Image" in job_type else "🎥"
            status = "🟢 ON" if enable else "🔴 OFF"
            print(f"👻 [LaoLi Shadow] {status} | {icon} {job_type}")
            
        return ()

# --- 节点 2: 逻辑门 ---
class LaoLi_Flow_Gate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "input_data": (any_type,) },
            "optional": { "wait_for": (any_type,) }
        }
    RETURN_TYPES = (any_type,) 
    FUNCTION = "run"
    CATEGORY = "LaoLi Shadow"
    DESCRIPTION = "流程控制：等待 wait_for 完成后才输出 input_data"
    
    def run(self, input_data, wait_for=None): return (input_data,)

# --- 节点 3: 显存排队优化 ---
class LaoLi_Lineup_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (any_type,),
                "vram_threshold": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 1.0, "step": 0.05}),
                "cleaning_interval": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "strict_mode": ("BOOLEAN", {"default": true}),
            }
        }
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "apply_lineup"
    CATEGORY = "LaoLi Shadow" 
    DESCRIPTION = "Lineup: 在模型层间插入显存检查点"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan") # 强制每次运行

    def apply_lineup(self, model, vram_threshold, cleaning_interval, strict_mode):
        if not Shadow_Config.enabled: return (model,)

        target_model_wrapper = model
        try:
            # 1. 解开影子
            if getattr(model, "_laoli_is_shadow", False):
                if Shadow_Config.verbose: print(f"⚡ [LaoLi Lineup] 唤醒影子模型以注入钩子...")
                if hasattr(model, "_ensure_real"):
                    model._ensure_real()
                    target_model_wrapper = model._laoli_real_obj
            elif hasattr(model, "clone"):
                try: target_model_wrapper = model.clone()
                except: target_model_wrapper = model
            
            # 2. 准备数据
            device = mm.get_torch_device()
            total_vram = 0
            if device.type == 'cuda':
                total_vram = torch.cuda.get_device_properties(device).total_memory
            
            reserve_bytes = Shadow_Config.vram_reserve_mb * 1024 * 1024
            limit_bytes = min(total_vram * vram_threshold, total_vram - reserve_bytes)
            
            # 3. 钩子逻辑
            def smart_hook(module, input):
                if total_vram == 0: return None
                try:
                    if torch.cuda.memory_reserved(device) >= limit_bytes:
                        if strict_mode and "Video" not in Shadow_Config.job_type: 
                            torch.cuda.synchronize()
                        mm.soft_empty_cache()
                except: pass
                return None

            # 4. 寻找层结构
            best_container = self._find_dominant_layer_container(target_model_wrapper)
            if best_container:
                blocks = list(best_container)
                count = 0
                for i, block in enumerate(blocks):
                    if i % cleaning_interval == 0 and hasattr(block, "register_forward_pre_hook"):
                        block.register_forward_pre_hook(smart_hook)
                        count += 1
                if Shadow_Config.verbose:
                    print(f"🚀 [LaoLi Lineup] 优化完毕: 挂载 {count} 个清理哨兵")
            else:
                if Shadow_Config.verbose: print(f"⚠️ [LaoLi Lineup] 无法识别模型结构，跳过优化")

            return (target_model_wrapper,)

        except Exception as e:
            print(f"❌ [LaoLi Lineup] 优化失败: {e}")
            return (model,)

    def _find_dominant_layer_container(self, root_obj):
        real_model = root_obj
        if hasattr(real_model, "model"): real_model = real_model.model
        if hasattr(real_model, "diffusion_model"): real_model = real_model.diffusion_model
        
        best_container = None
        max_len = 0
        try:
            for name, module in real_model.named_modules():
                if isinstance(module, (nn.ModuleList, nn.Sequential)):
                    if len(module) > max_len and len(module) > 4:
                        max_len = len(module)
                        best_container = module
        except: pass
        return best_container

# ==========================================================
# 6. Prompt 预扫描 (保持配置同步)
# ==========================================================
try:
    import server
    if not hasattr(server.PromptServer, "_laoli_original_trigger"):
        server.PromptServer._laoli_original_trigger = server.PromptServer.trigger_computation

    def _shadow_hooked_trigger(self, prompt, id, *args, **kwargs):
        # 默认关闭，等待扫描激活
        Shadow_Config.enabled = False
        try:
            for uid, data in prompt.items():
                if data.get('class_type') == 'LaoLi_Shadow':
                    inputs = data.get('inputs', {})
                    Shadow_Config.enabled = inputs.get('enable', True)
                    Shadow_Config.job_type = inputs.get('job_type', "Image")
                    Shadow_Config.shadow_mode = inputs.get('shadow_mode', True)
                    Shadow_Config.mode = inputs.get('mode', "Ease Mode")
                    Shadow_Config.verbose = inputs.get('verbose', True)
                    # 简单读取数值
                    try: Shadow_Config.vram_reserve_mb = float(inputs.get('vram_reserve_mb', 512.0))
                    except: pass
                    break
        except: pass
        return server.PromptServer._laoli_original_trigger(self, prompt, id, *args, **kwargs)

    server.PromptServer.trigger_computation = _shadow_hooked_trigger
except: pass

# ==========================================================
# 7. 注册节点
# ==========================================================
NODE_CLASS_MAPPINGS = {
    "LaoLi_Shadow": LaoLi_Shadow_Node,
    "LaoLi_Flow_Gate": LaoLi_Flow_Gate,
    "LaoLi_Lineup": LaoLi_Lineup_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LaoLi_Shadow": "👻 老李_影子 (Shadow) ",
    "LaoLi_Flow_Gate": "🚧 老李_逻辑门 (Flow Gate)",
    "LaoLi_Lineup": "🚀 老李_排队 (Lineup)"
}