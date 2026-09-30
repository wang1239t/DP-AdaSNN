# DP-AdaSNN

DP-AdaSNN 核心实现，包含自适应阈值、原版时间曲面驱动损失（TSD）、SpikeYOLO-P2，以及 GEN1/DSEC 的训练、独立测试和事件检测入口。

[English](README.md) · [许可证](LICENSE) · [第三方来源](THIRD_PARTY_NOTICES.md)

## 环境与安装

使用 Python 3.11，并根据 CPU/CUDA 平台安装匹配的 PyTorch 与 torchvision。本版在 PyTorch 2.7.0、torchvision 0.22.0、SpikingJelly 0.0.0.0.14 环境检查。

```bash
python -m pip install -r requirements.txt
```

所有命令在仓库根目录运行。仓库内的 `ultralytics/` 已修改，不能用 PyPI 版本替代。入口不会自动安装依赖。完整模型 `.pt` 使用 Python 反序列化，只加载可信检查点，例如自己训练生成的文件。

## 数据与配置

数据需自行获取：[GEN1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)、[DSEC](https://dsec.ifi.uzh.ch/)。本版读取已经按时间窗口准备好的事件和 YOLO 标签，不包含原始数据转换工具及数据本身。

目录格式为 `data/gen1/{train,val,test}/events/*.npy` 和对应的 `labels/*.txt`，DSEC 同理。事件数组为 `N×4`，列顺序 `[x,y,timestamp,polarity]`，时间戳按升序排列、单位为微秒，极性为 0/1。配置中的列重排转换为内部 `[x,y,polarity,timestamp]`。标签每行是 `class_id center_x center_y width height`，坐标归一化至 0～1。GEN1 为 car/pedestrian，DSEC 为数据 YAML 中的七类；保留原始 train/val/test 划分。

将数据 YAML 的 `path` 占位值改为数据实际**绝对路径**。测试和检测 YAML 的 `model` 指向本版训练的检查点；检测的 `source` 指向单个 `.npy` 或事件目录，目录内文件名主干必须唯一。

## 运行

```bash
python train.py --cfg ultralytics/cfg/experiments/gen1_train.yaml
python test.py --cfg ultralytics/cfg/experiments/gen1_test.yaml
python detect.py --cfg ultralytics/cfg/experiments/gen1_detect.yaml
```

将 `gen1` 改为 `dsec` 即可切换数据集。测试默认 `split: test`，训练使用 val。检测逐文件处理，只保存常规预测框图片和可选 YOLO 标签；输入计数的显示映射不会参与模型计算。

训练模板保留双卡配置，GEN1 batch=16、DSEC batch=8。按机器修改 `cuda_visible_devices`、`device`、路径和输出名称；复现时保持科学参数不变。batch 必须为明确的正整数。CPU 检查设置 `device: cpu`、`cuda_visible_devices: null`。已有输出目录由 Ultralytics 自动递增，不覆盖。

## 方法冻结

| 参数 | GEN1 | DSEC |
|---|---|---|
| 检测器 | s，P2/P3/P4/P5 | s，P2/P3/P4/P5 |
| 原始事件尺寸 | 240×304，缩放/填充至 320 | 480×640，no-pad |
| 输入步数 / 前端槽数 | 6 / 4 | 4 / 3 |
| 输入计数 cap | 512 | 无 |
| 时间曲面 tau | 50,000 µs | 18,000 µs |
| vreset / 输出 clamp | 0 / 24 | 0 / 24 |
| SAT / 对数压缩 | 关闭 / 关闭 | 关闭 / 关闭 |
| AdaVth / 原版 TSD | 启用 / 启用 | 启用 / 启用 |

核心计算和配置从冻结研究版本提取，保留原版 TSD 公式、时间反转、物理缩放及损失权重。合成数据检查不能代表真实数据集指标，本版不发布未经核验的结果声明。

## 开源边界

保留核心方法、运行必需的检测框架、GEN1/DSEC 数据链路、六套入口模板、常规 loss 与 P/R/mAP。

不包含内部状态/发放率统计 hook、物理/能耗报告、效率 profiler、审计捕获、前端研究可视化、外部日志平台回调、后续研究变体、其他数据集、实验管理、权重或历史结果。私有监控 API 与诊断记录参数已移除。

历史研究检查点若包含已移除的类，不保证直接加载；优先在本版训练，或明确迁移兼容的 state_dict。许可证为 AGPL-3.0，来源声明见第三方说明。

## 小规模回归检查

```bash
python -m unittest discover -s tests -v
```

Windows 入口保留原项目的进程级 OpenMP 兼容设置，不修改已安装环境。GPU 预热使用五维事件张量，AMP 检查使用本地模型和小规模合成输入。

本次验收包含 CPU 合成训练/测试/检测、冻结源逐值对照，以及单卡 64×96 输入的推理和 AMP 接口检查；不涵盖 CUDA 训练、多卡训练及完整真实数据集结果。
