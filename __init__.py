import sys
import subprocess
import importlib

def check_and_install(package_name):
    try:
        importlib.import_module(package_name)
    except ImportError:
        print(f"\033[33m[LaoLi Shadow] 正在自动安装依赖: {package_name} ...\033[0m")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"\033[32m[LaoLi Shadow] {package_name} 安装成功！\033[0m")
        except Exception as e:
            print(f"\033[31m[LaoLi Shadow] 自动安装失败: {e}\n请手动运行: pip install {package_name}\033[0m")

check_and_install("psutil")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("\033[34m[LaoLi Shadow] \033[0m影子系统已激活：ControlNet 0秒启动 | 智能显存共存")