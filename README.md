# BrickGPT LoRA 微调与上下文学习

实验 A 使用三个完整搭建示例，实验 B 使用三个局部合并示例。两组共享 36 个训练尺寸组，4×4 留作评测。

## 实验报告与结果

- [结果目录说明](results/README.md)
- [实验总报告](results/brickgpt_lora_icl_experiment_report.md)和[早期纯 ICL 报告](results/icl_report/experiment_report.md)
- [最终 30 个用例的 A/B 对照](results/icl_lora_train_expanded_v1/comparison.json)
- [可视化图集](results/GALLERY.md)展示相同 10 个 4×4 任务的 A/B 输出，共 20 张图

## 本地环境

使用 Python 3.12 和支持 BF16 的 CUDA 版 PyTorch。将上游 BrickGPT 切换到实验使用的版本。

```powershell
git clone https://github.com/AvaLovelace1/BrickGPT.git
git -C BrickGPT checkout 9c53c7cb65f9de0df89562cc74b2d2e270fbcab4
python -m pip install -r requirements-stage1.txt
python -m pip install "numpy<2" matplotlib accelerate networkx
```

将 Llama-3.2-1B-Instruct 权重放在 `temp/models/Llama-3.2-1B-Instruct/`，将原始 BrickGPT adapter 和 tokenizer 放在 `temp/models/BrickGPT/`。脚本也支持从已配置的本地 Hugging Face 缓存读取 Llama 权重。

## 数据和运行

从 Google Drive 下载所需 ZIP 并解压到项目的 `data/`。压缩包自带对应的子目录。

| 压缩包 | 解压后的目录 | 内容 |
|---|---|---|
| `brickgpt_icl_ab_v1.zip` | `data/brickgpt_icl_ab_v1/` | A/B 样本、固定评测和数据核验报告 |
| `stabletext2brick_raw_v1.zip` | `data/stabletext2brick_raw_v1/` | 七个原始 Parquet 分片和来源信息 |
| `brickgpt_icl_ab_rebuild_inputs_v1.zip` | `data/brickgpt_icl_ab_rebuild_inputs_v1/` | 标准化结构和固定测试目标清单 |

训练和评测只需第一个包。重建样本时，解压重建输入包，再运行 `python scripts/prepare_context_expanded.py`，输出到 `data/brickgpt_icl_ab_v1/`。原始数据包可用于查阅 StableText2Brick 数据。

```powershell
Expand-Archive -LiteralPath .\brickgpt_icl_ab_v1.zip -DestinationPath .\data
```

Google Drive 下载链接待补充。可用随包提供的 `SHA256SUMS.txt` 校验下载文件。

运行 `python -u scripts/train_lora_expanded.py`，程序依次完成 A、B 的训练与评测。进度和 checkpoint 保存到 `output/icl_lora_train_expanded_v1/`。

已记录评测的生成语法约束允许边长 1–8。训练数据还包含边长为 9 的尺寸组。

## 微调权重

最终 adapter 分别位于训练输出目录下的 `A/checkpoints/step_2250/` 和 `B/checkpoints/step_2250/`。发布文件包括 `adapter_config.json`、`adapter_model.safetensors` 和对应的 tokenizer。使用时先将原始 BrickGPT 合并到 Llama 基座，再加载实验 adapter。续训需要另外保留训练状态。

数据、缓存和权重通过 Git 忽略规则排除。权重的 Google Drive 链接待补充。
