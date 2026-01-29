# ComfyUI-LaoLi-Shadow (影子)

**极速启动 · 智能显存管理 · 深度模型优化**

ComfyUI-LaoLi-Shadow 是一个专为 ComfyUI 设计的底层性能优化插件。它通过“影子加载”技术实现工作流的秒级启动，并通过“显存排队”机制和模型动态卸载防止大模型采样时爆显存。

核心特性：
- **全自动影子模式**：加载 Checkpoint/ControlNet 时不读硬盘、不占显存，只有在真正采样时才触发加载。
- **智能显存共存**：自动卸载未使用的模型，让 VLM、LLM 和绘图模型在小显存显卡上也能流畅协作。
- **通用模型支持**：内置全局算法 (Dominant Layer Search)，完美支持 SD1.5/SDXL/Flux/WanVideo/Qwen 以及各种复杂的 Wrapper 封装模型。

---
<img width="3373" height="1420" alt="image" src="https://github.com/user-attachments/assets/c0e95dae-5dfc-47d3-9563-d5e875b7a69c" />



## 🛠️ 节点说明 (Nodes)

插件包含三个核心节点，分别负责**全局控制**、**推理优化**和**流程控制**。

### 1. 👻 老李_影子 (Shadow)
**【全局控制器】**
无需连接任何连线，只需添加到工作流中即可生效。它接管了 ComfyUI 的底层模型加载逻辑。

*   **参数详解：**
    *   `enable` (启用): 插件总开关。关闭后恢复 ComfyUI 原生行为。
    *   `shadow_mode` (影子模式): **核心功能**。
        *   `True`: 开启延迟加载。启动工作流时，硬盘读取和显存占用均为 0。
        *   `False`: 传统加载模式。
    *   `mode` (运行模式):
        *   `Ease Mode` (安逸模式/推荐): 智能管理显存。当显存不足时，自动卸载当前不参与计算的模型（如跑完反推后自动卸载 VLM）。
        *   `Monitor Mode` (监控模式): 仅监控显存，不主动执行卸载策略（适合 24G+ 大显存用户）。
    *   `ram_reserve_gb` (内存保留): **防卡死安全气囊**。
        *   当系统物理内存（RAM）剩余量低于此数值（例如 4.0GB）时，强制触发垃圾回收（GC），防止电脑因内存耗尽而死机。
    *   `verbose` (日志): 是否在控制台打印详细的运行日志（建议开启）。

---

### 2. 🚀 老李_排队 (Lineup VRAM)
**【推理显存保姆】**
这是 V15 版本最强大的节点。它利用“全局算法”深入扫描模型内部，在每一层计算前检查显存。

*   **节点作用：**
    *   防止采样过程中（KSampler）因为显存瞬间峰值导致的 OOM（爆显存）。
    *   支持 **万能连接**：无论是标准的 Checkpoint，还是深层封装的 Qwen/WanVideo/Flux 包装器，都能自动识别核心计算层。
    *   支持 **影子穿透**：如果连接的是“影子”模型，它会强制加载真身并进行优化。

*   **连接说明：**
    *   **输入 (model)**: 连接大模型加载器（Load Checkpoint）或任何模型处理节点的输出。
    *   **输出 (optimized_model)**: 连接到 KSampler 或其他使用模型的节点。

*   **参数详解：**
    *   `vram_threshold` (显存阈值): **警戒线**（默认 0.85）。
        *   含义：当显存占用达到总显存的 85% 时，触发清理。
        *   建议：8G-12G 显存设为 `0.80`；16G-24G 显存设为 `0.85` 或 `0.90`。
    *   `cleaning_interval` (清理间隔):
        *   `1`: 每计算 1 层就检查一次显存（最安全，推荐）。
        *   `2`: 每 2 层检查一次（稍快，风险稍高）。
    *   `strict_mode` (严格模式):
        *   `True` (推荐): 发现显存超标时，强制暂停 GPU (Synchronize) 进行清理。**防崩效果最好**。
        *   `False`: 仅发送清理请求，不强制暂停。速度略快，但可能拦不住显存激增。

---

### 3. 🚧 老李_逻辑门 (Flow Gate)
**【流程红绿灯】**
强制控制 ComfyUI 的执行顺序，解决并行任务导致的资源冲突。

*   **节点作用：**
    *   强制要求 ComfyUI 必须先执行完 B 任务，才能开始 A 任务。
    *   **典型场景**：先跑 VLM (图像反推) 拿到提示词，跑完释放显存后，再加载 ControlNet 模型进行绘图。

*   **连接说明：**
    *   **input_data (输入)**: 连接你想要**延迟**执行的数据（例如：ControlNet 模型、Checkpoint 模型）。
    *   **wait_for (等待)**: 连接必须**先执行**的数据（例如：VLM 节点输出的 STRING 提示词）。
    *   **输出**: 连接到原本 `input_data` 应该去的地方。

---

## 🔗 连接示例 (Wiring Guide)

### 场景 A：常规绘图防爆显存
最简单的用法，防止 Flux 或 Qwen 爆显存。

```mermaid
[Load Checkpoint] ===> [🚀 老李_排队] ===> [KSampler]
                           ^
                    (设置阈值 0.85)
```

### 场景 B：VLM 反推 + ControlNet (极致显存优化)
确保 VLM 跑完并释放显存后，才加载 ControlNet。

```mermaid
[Load Image] ---------------------+
      |                           |
      v                           v
[VLM 节点] ===(输出提示词String)==> [🚧 老李_逻辑门 (wait_for)]
                                          ^
[Load ControlNet] ==(模型Model)==> [🚧 老李_逻辑门 (input_data)] 
                                          |
                                          v
                                  [ControlNet Apply]
```

---

## 📥 安装 (Installation)

1.  进入 ComfyUI 的 `custom_nodes` 目录。
2.  克隆本仓库：
    ```bash
    git clone https://github.com/YourName/ComfyUI-LaoLi-Shadow.git
    ```
3.  重启 ComfyUI。
