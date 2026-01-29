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
    enabled = True
    mode = "Ease Mode" 
    shadow_mode = True 
    ram_reserve_gb = 4.0  # 内存保留默认值
    vram_reserve_mb = 1024.0  # 显存保留默认值
    verbose = True
    vram_cushion_gb = 1.0 

# ==========================================================
# 2. 影子系统 (The Shadow Legion)
# ==========================================================
class ShadowGroup:
    def __init__(self, name, loader_func, *args, **kwargs):
        self.name = name
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self.is_loaded = False
        self.cached_model = None
        self.cached_clip = None
        self.cached_vae = None

    def _execute_load(self):
        if self.is_loaded: return
        if Shadow_Config.verbose: print(f"👻 [LaoLi Shadow] 触发 Checkpoint 加载: {self.name} ...")
        
        if psutil:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            if available_gb < Shadow_Config.ram_reserve_gb:
                if Shadow_Config.verbose: print(f"⚠️ [LaoLi Shadow] 剩余内存过低 -> 触发GC")
                gc.collect()

        start_t = time.time()
        out = self.loader_func(*self.args, **self.kwargs)
        self.cached_model = out[0]
        self.cached_clip = out[1]
        self.cached_vae = out[2]
        self.is_loaded = True
        if Shadow_Config.verbose: print(f"✨ [LaoLi Shadow] {self.name} 全部就绪 (耗时 {time.time()-start_t:.2f}s)")

    def get_real_thing(self, mode):
        if not self.is_loaded: self._execute_load()
        if mode == "model": return self.cached_model
        if mode == "clip": return self.cached_clip
        if mode == "vae": return self.cached_vae
        return None

class ShadowInnerModel(torch.nn.Module):
    def __init__(self, parent_patcher):
        super().__init__()
        self._laoli_parent = parent_patcher 
    def __getattr__(self, name):
        if name.startswith("_laoli") or name.startswith("training"): return super().__getattr__(name)
        if Shadow_Config.verbose:
            if name not in ['device', 'dtype']: print(f"⚡ [LaoLi Shadow] Deep Access 触发加载: .model.{name}")
        real_patcher = self._laoli_parent._ensure_real()
        return getattr(real_patcher.model, name)

class ShadowPatcher(ModelPatcher):
    def __init__(self, group, *args, **kwargs):
        dummy = ShadowInnerModel(self) 
        super().__init__(dummy, torch.device("cpu"), torch.device("cpu"))
        self._laoli_group = group
        self._laoli_real_obj = None
        self._laoli_is_shadow = True 

    def _ensure_real(self):
        if self._laoli_real_obj is None:
            real = self._laoli_group.get_real_thing("model")
            self.become_real(real)
        return self._laoli_real_obj

    def become_real(self, real_obj):
        if real_obj is None: return 
        self._laoli_real_obj = real_obj
        self.__dict__.update(real_obj.__dict__)
        try: self.__class__ = real_obj.__class__
        except: pass
        if hasattr(self, "_laoli_is_shadow"): del self._laoli_is_shadow
    
    def copy(self): return self.clone()
    def clone(self, *args, **kwargs):
        if self._laoli_real_obj: return self._laoli_real_obj.clone()
        return ShadowPatcher(self._laoli_group)
    def __getattr__(self, name):
        if name.startswith("_laoli"): raise AttributeError(name)
        real = self.__dict__.get("_laoli_real_obj", None)
        if name == "pinned" and real is None: return set()
        if Shadow_Config.verbose and not real:
            if not name.startswith("__"): print(f"⚡ [LaoLi Shadow] MODEL 触发加载: method '{name}'")
        if real is None: real = self._ensure_real()
        return getattr(real, name)

class ShadowCLIP:
    def __init__(self, group):
        self._laoli_group = group
        self._laoli_real_obj = None
    def _ensure_real(self):
        if self._laoli_real_obj is None:
            real = self._laoli_group.get_real_thing("clip")
            self.become_real(real)
        return self._laoli_real_obj
    def become_real(self, real_obj):
        self._laoli_real_obj = real_obj
        self.__dict__.update(real_obj.__dict__)
        try: self.__class__ = real_obj.__class__
        except: pass
    def clone(self):
        if self._laoli_real_obj: return self._laoli_real_obj.clone()
        return ShadowCLIP(self._laoli_group)
    def copy(self): return self.clone()
    def __getattr__(self, name):
        if name.startswith("_laoli"): raise AttributeError(name)
        if Shadow_Config.verbose and not self._laoli_real_obj:
            print(f"⚡ [LaoLi Shadow] CLIP 触发加载: method '{name}'")
        real = self._ensure_real()
        return getattr(real, name)

class ShadowVAE:
    def __init__(self, group):
        self._laoli_group = group
        self._laoli_real_obj = None
    def _ensure_real(self):
        if self._laoli_real_obj is None:
            real = self._laoli_group.get_real_thing("vae")
            self.become_real(real)
        return self._laoli_real_obj
    def become_real(self, real_obj):
        self._laoli_real_obj = real_obj
        self.__dict__.update(real_obj.__dict__)
        try: self.__class__ = real_obj.__class__
        except: pass
    def decode(self, samples_in): return self._ensure_real().decode(samples_in)
    def encode(self, pixel_samples): return self._ensure_real().encode(pixel_samples)
    def __getattr__(self, name):
        if name.startswith("_laoli"): raise AttributeError(name)
        if Shadow_Config.verbose and not self._laoli_real_obj:
            print(f"⚡ [LaoLi Shadow] VAE 触发加载: method '{name}'")
        real = self._ensure_real()
        return getattr(real, name)

# ==========================================================
# 3. 劫持加载器
# ==========================================================
if not hasattr(comfy.sd, "_laoli_org_load_ckpt"):
    comfy.sd._laoli_org_load_ckpt = comfy.sd.load_checkpoint_guess_config

def _hacked_load_checkpoint(ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
    if Shadow_Config.enabled and Shadow_Config.shadow_mode:
        name = os.path.basename(ckpt_path)
        if Shadow_Config.verbose: print(f"💤 [LaoLi Shadow] 拦截 Checkpoint: {name} -> 建立影子阵列")
        group = ShadowGroup(name, comfy.sd._laoli_org_load_ckpt, ckpt_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=embedding_directory)
        return (ShadowPatcher(group), ShadowCLIP(group), ShadowVAE(group))
    return comfy.sd._laoli_org_load_ckpt(ckpt_path, output_vae, output_clip, embedding_directory)
comfy.sd.load_checkpoint_guess_config = _hacked_load_checkpoint

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
        if Shadow_Config.verbose: print(f"👻 [LaoLi Shadow] 加载 ControlNet: {self._laoli_name}")
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
def _hacked_load_controlnet(ckpt_path):
    if Shadow_Config.enabled and Shadow_Config.shadow_mode:
        name = os.path.basename(ckpt_path)
        if Shadow_Config.verbose: print(f"💤 [LaoLi Shadow] 拦截 ControlNet: {name} -> 影子模式")
        return ShadowControlNet(name, lambda: comfy.controlnet._laoli_org_load_cn(ckpt_path))
    return comfy.controlnet._laoli_org_load_cn(ckpt_path)
comfy.controlnet.load_controlnet = _hacked_load_controlnet

# ==========================================================
# 4. 显存管理与触发器 (Safe VRAM Check)
# ==========================================================
if not hasattr(mm, "_laoli_original_load_models_gpu"):
    mm._laoli_original_load_models_gpu = mm.load_models_gpu

def _shadow_load_models_gpu(models, memory_required=0, **kwargs):
    if Shadow_Config.enabled:
        try:
            device = mm.get_torch_device()
            # 1. 唤醒影子
            for model in models:
                if getattr(model, "_laoli_is_shadow", False):
                    if hasattr(model, "summon"): model.summon() 
                    if hasattr(model, "_ensure_real"): model._ensure_real()
            
            # 2. Ease Mode 显存预检查
            if Shadow_Config.mode == "Ease Mode":
                all_loaded = True
                for model in models:
                    if hasattr(model, "current_device"):
                        if model.current_device != device: all_loaded = False; break
                    else: all_loaded = False; break
                
                if not all_loaded and device.type == 'cuda':
                    try:
                        free_mem, total_mem = torch.cuda.mem_get_info(device)
                    except:
                        stats = torch.cuda.get_device_properties(device)
                        free_mem = stats.total_memory - torch.cuda.memory_reserved(device)
                    
                    needed = memory_required if memory_required > 0 else (1.0 * 1024**3)
                    
                    # 使用 Shadow 节点的全局预留设置
                    reserve_bytes = Shadow_Config.vram_reserve_mb * 1024 * 1024
                    cushion_bytes = Shadow_Config.vram_cushion_gb * 1024**3
                    safe_cushion = max(cushion_bytes, reserve_bytes)

                    if free_mem < (needed + safe_cushion):
                        if Shadow_Config.verbose: 
                            print(f"🧹 [LaoLi Shadow] 显存不足 (真实剩余{free_mem/1024**3:.1f}G | 需保留{Shadow_Config.vram_reserve_mb}MB) -> 强制清理")
                        mm.unload_all_models()
                        mm.soft_empty_cache()
                        if device.type == 'cuda': torch.cuda.empty_cache()

            # 3. 内存(RAM) 检查
            if psutil:
                mem = psutil.virtual_memory()
                available_gb = mem.available / (1024**3)
                if available_gb < Shadow_Config.ram_reserve_gb:
                     if Shadow_Config.verbose: print(f"⚠️ [LaoLi Shadow] 系统内存不足 -> 触发GC")
                     gc.collect()

        except Exception as e: print(f"❌ [LaoLi Shadow Error] {e}")
    
    return mm._laoli_original_load_models_gpu(models, memory_required=memory_required, **kwargs)
mm.load_models_gpu = _shadow_load_models_gpu

# ==========================================================
# 5. 节点定义 (Nodes Definition)
# ==========================================================

class LaoLi_Shadow_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "shadow_mode": ("BOOLEAN", {"default": True}),
                "mode": (["Ease Mode", "Monitor Mode"],),
                "ram_reserve_gb": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5}),
                "vram_reserve_mb": ("FLOAT", {"default": 512.0, "min": 0.0, "max": 8192.0, "step": 64.0, "tooltip": "为系统预留的显存(MB)，防止卡死"}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "update_settings"
    CATEGORY = "LaoLi Shadow"
    DESCRIPTION = "👻 老李_影子 (Shadow) : 全局控制与资源管理"
    
    def update_settings(self, enable, shadow_mode, mode, ram_reserve_gb, vram_reserve_mb, verbose):
        Shadow_Config.enabled = enable
        Shadow_Config.shadow_mode = shadow_mode
        Shadow_Config.mode = mode
        Shadow_Config.ram_reserve_gb = float(ram_reserve_gb)
        Shadow_Config.vram_reserve_mb = float(vram_reserve_mb)
        Shadow_Config.verbose = verbose
        status = "✅ 开启" if enable else "⏸️ 暂停"
        print(f"\n👻 [LaoLi Shadow] {status} | 模式: {mode} | VRAM预留: {vram_reserve_mb}MB")
        return ()

class LaoLi_Flow_Gate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "input_data": (any_type, {"tooltip": "连接任何数据"}) },
            "optional": { "wait_for": (any_type, {"tooltip": "连接先决条件"}) }
        }
    RETURN_TYPES = (any_type,) 
    FUNCTION = "run"
    CATEGORY = "LaoLi Shadow"
    DESCRIPTION = "强行让 ComfyUI 等待 'wait_for' 完成后，才释放 'input_data'。"
    def run(self, input_data, wait_for=None): return (input_data,)

class LaoLi_Lineup_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (any_type,),
                "vram_threshold": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 1.0, "step": 0.05}),
                "cleaning_interval": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "strict_mode": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "apply_lineup"
    CATEGORY = "LaoLi Shadow" 
    DESCRIPTION = "老李 Lineup : 显存排队与深度优化"

    def apply_lineup(self, model, vram_threshold, cleaning_interval, strict_mode):
        target_model_wrapper = model
        try:
            # 1. 影子处理
            if getattr(model, "_laoli_is_shadow", False):
                if Shadow_Config.verbose: print(f"⚡ [LaoLi Lineup] 检测到影子，强制加载真身...")
                model._ensure_real()
                target_model_wrapper = model._laoli_real_obj
            elif hasattr(model, "clone"):
                try: target_model_wrapper = model.clone()
                except: target_model_wrapper = model
            
            # 2. 显存计算 (双重限制)
            device = mm.get_torch_device()
            total_vram = 0
            if device.type == 'cuda':
                total_vram = torch.cuda.get_device_properties(device).total_memory
            
            reserve_bytes = Shadow_Config.vram_reserve_mb * 1024 * 1024
            limit_by_ratio = total_vram * vram_threshold
            limit_by_reserve = total_vram - reserve_bytes
            effective_limit_bytes = min(limit_by_ratio, limit_by_reserve)
            
            if Shadow_Config.verbose and total_vram > 0:
                print(f"🛡️ [LaoLi Lineup] 显存安全线: {effective_limit_bytes/1024**3:.2f} GB (全局预留: {Shadow_Config.vram_reserve_mb}MB)")

            def smart_hook(module, input):
                if total_vram == 0: return None
                current_reserved = torch.cuda.memory_reserved(device)
                if current_reserved >= effective_limit_bytes:
                    if strict_mode: torch.cuda.synchronize() 
                    mm.soft_empty_cache()       
                return None

            # 3. 搜索核心层 (优化版)
            best_container = self._find_dominant_layer_container(target_model_wrapper)

            if best_container is None:
                if Shadow_Config.verbose: print(f"⚠️ [LaoLi Lineup] 扫描完成，未发现可用的层结构 (可能模型已被高度封装)")
                return (target_model_wrapper,)

            blocks = list(best_container)
            mounted_count = 0
            
            for i, block in enumerate(blocks):
                if i % cleaning_interval == 0:
                    block.register_forward_pre_hook(smart_hook)
                    mounted_count += 1
            
            if Shadow_Config.verbose:
                 print(f"🚀 [LaoLi Lineup] 注入成功 | 发现: {len(blocks)}层 | 挂载: {mounted_count}个钩子")

            return (target_model_wrapper,)

        except Exception as e:
            print(f"❌ [LaoLi Lineup Error] {e}")
            return (model,)

    def _find_dominant_layer_container(self, root_obj):
        # 主动剥洋葱逻辑：专门处理 ComfyUI 的 ModelPatcher 和 SD/Flux 结构
        real_model = root_obj
        
        # 1. 剥离 ModelPatcher
        if hasattr(real_model, "model"): 
            real_model = real_model.model
            
        # 2. 剥离 ComfyUI 的 BaseModel 包装 (针对 SD/Flux)
        # 大多数模型的真正层结构在 diffusion_model 属性下
        if hasattr(real_model, "diffusion_model"):
            real_model = real_model.diffusion_model
            
        best_container = None
        max_len = 0
        
        # 3. 直接在剥离后的模型中搜索
        try:
            # 使用 named_modules 搜索所有子模块
            for name, module in real_model.named_modules():
                if isinstance(module, (nn.ModuleList, nn.Sequential)):
                    curr_len = len(module)
                    # 只有长度大于 4 的才被认为是核心计算层 (排除一些小的 embedding list)
                    if curr_len > 4: 
                        if curr_len > max_len:
                            max_len = curr_len
                            best_container = module
        except Exception:
            pass
            
        return best_container

# --- 注册节点 ---
NODE_CLASS_MAPPINGS = {
    "LaoLi_Shadow": LaoLi_Shadow_Node,
    "LaoLi_Flow_Gate": LaoLi_Flow_Gate,
    "LaoLi_Lineup": LaoLi_Lineup_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LaoLi_Shadow": "👻 老李_影子 (Shadow)",
    "LaoLi_Flow_Gate": "🚧 老李_逻辑门 (Flow Gate)",
    "LaoLi_Lineup": "🚀 老李_排队 (Lineup VRAM)"
}