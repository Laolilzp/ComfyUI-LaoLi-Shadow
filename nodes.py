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
    ram_reserve_gb = 4.0
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
# 4. 显存管理与触发器
# ==========================================================
if not hasattr(mm, "_laoli_original_load_models_gpu"):
    mm._laoli_original_load_models_gpu = mm.load_models_gpu
def _shadow_load_models_gpu(models, memory_required=0, **kwargs):
    if Shadow_Config.enabled:
        try:
            device = mm.get_torch_device()
            for model in models:
                if getattr(model, "_laoli_is_shadow", False):
                    if hasattr(model, "summon"): model.summon() 
                    if hasattr(model, "_ensure_real"): model._ensure_real()
            if Shadow_Config.mode == "Ease Mode":
                all_loaded = True
                for model in models:
                    if hasattr(model, "current_device"):
                        if model.current_device != device: all_loaded = False; break
                    else: all_loaded = False; break
                if not all_loaded:
                    if device.type == 'cuda':
                        stats = torch.cuda.get_device_properties(device)
                        free = stats.total_memory - torch.cuda.memory_reserved(device)
                        needed = memory_required if memory_required > 0 else (1.0 * 1024**3)
                        cushion = Shadow_Config.vram_cushion_gb * 1024**3
                        if free < (needed + cushion):
                            if Shadow_Config.verbose: print(f"🧹 [LaoLi Shadow] 显存不足 (余{free/1024**3:.1f}G) -> 清理")
                            mm.unload_all_models()
                            mm.soft_empty_cache()
                            if device.type == 'cuda': torch.cuda.empty_cache()
            if psutil:
                mem = psutil.virtual_memory()
                available_gb = mem.available / (1024**3)
                if available_gb < Shadow_Config.ram_reserve_gb:
                     if Shadow_Config.verbose: print(f"⚠️ [LaoLi Shadow] 剩余内存过低 -> GC")
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
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "update_settings"
    CATEGORY = "LaoLi Shadow"
    DESCRIPTION = "👻 老李_影子 (Shadow)  "
    def update_settings(self, enable, shadow_mode, mode, ram_reserve_gb, verbose):
        Shadow_Config.enabled = enable
        Shadow_Config.shadow_mode = shadow_mode
        Shadow_Config.mode = mode
        Shadow_Config.ram_reserve_gb = float(ram_reserve_gb)
        Shadow_Config.verbose = verbose
        status = "✅ 开启" if enable else "⏸️ 暂停"
        print(f"\n👻 [LaoLi Shadow] {status} | 模式: {shadow_mode} | 内存保留: {ram_reserve_gb}GB")
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
    DESCRIPTION = "老李 Lineup : 影子穿透+全局算法。"

    def apply_lineup(self, model, vram_threshold, cleaning_interval, strict_mode):
        target_model_wrapper = model
        try:
            # 1. 如果是影子，必须强制现身，否则扫描不到任何层
            if getattr(model, "_laoli_is_shadow", False):
                if Shadow_Config.verbose: print(f"⚡ [LaoLi Lineup] 检测到影子，正在强制加载真身以进行优化...")
                model._ensure_real() # 强制读盘
                target_model_wrapper = model._laoli_real_obj # 拿到真身(ModelPatcher)
            elif hasattr(model, "clone"):
                try: target_model_wrapper = model.clone()
                except: target_model_wrapper = model
            
            # 2. 准备钩子
            device = mm.get_torch_device()
            total_vram = 0
            if device.type == 'cuda':
                total_vram = torch.cuda.get_device_properties(device).total_memory
            
            def smart_hook(module, input):
                if total_vram == 0: return None
                current_reserved = torch.cuda.memory_reserved(device)
                usage_ratio = current_reserved / total_vram
                if usage_ratio >= vram_threshold:
                    if strict_mode: torch.cuda.synchronize() 
                    mm.soft_empty_cache()       
                return None

            # 3. 全局算法 (扫描真身)
            best_container = self._find_dominant_layer_container(target_model_wrapper)

            if best_container is None:
                if Shadow_Config.verbose: 
                    print(f"⚠️ [LaoLi Lineup] 扫描失败: {type(target_model_wrapper).__name__} 内部没有发现任何层列表。")
                # 即使失败也返回真身(或克隆体)，不要返回影子，否则采样器可能不认
                return (target_model_wrapper,)

            blocks = list(best_container)
            mounted_count = 0
            
            for i, block in enumerate(blocks):
                if i % cleaning_interval == 0:
                    block.register_forward_pre_hook(smart_hook)
                    mounted_count += 1

            if Shadow_Config.verbose:
                print(f"🚀 [LaoLi Lineup] 注入成功 | 目标: {len(blocks)}层结构 | 挂载: {mounted_count}层 | 阈值: {int(vram_threshold*100)}%")
            
            return (target_model_wrapper,)

        except Exception as e:
            print(f"❌ [LaoLi Lineup Error] {e}")
            return (model,)

    def _find_dominant_layer_container(self, root_obj):
        best_container = None
        max_len = 0
        
        # 定义搜索生成器，自动处理 ModelPatcher 和 Module
        def iter_modules(obj):
            if isinstance(obj, ModelPatcher):
                # 确保这里拿到的是真的 model
                yield from obj.model.named_modules()
            elif isinstance(obj, torch.nn.Module):
                yield from obj.named_modules()
            else:
                # 鸭子类型尝试
                if hasattr(obj, "model") and isinstance(obj.model, torch.nn.Module):
                    yield from obj.model.named_modules()

        for name, module in iter_modules(root_obj):
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                curr_len = len(module)
                # Qwen/Wan 的层数通常 > 4
                if curr_len > 4: 
                    if curr_len > max_len:
                        max_len = curr_len
                        best_container = module
        
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