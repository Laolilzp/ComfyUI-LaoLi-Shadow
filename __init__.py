import sys
import importlib
import traceback

# 1. 依赖检查
def check_dependency(package_name):
    try:
        importlib.import_module(package_name)
    except ImportError:
        print(f"\033[33m⚠️ [LaoLi Shadow] 缺少依赖库: {package_name}，请手动安装: pip install {package_name}\033[0m")

check_dependency("psutil")

# 2. 尝试导入节点
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    print("\033[34m[LaoLi Shadow] \033[0m影子系统已就绪")
except Exception as e:
    print(f"\033[31m❌ [LaoLi Shadow] 插件加载失败！错误详情如下：\033[0m")
    traceback.print_exc()  # 打印完整的错误堆栈，这非常重要
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = []