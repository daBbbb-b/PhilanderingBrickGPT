# 实验结果目录

实验 A 使用三个完整搭建示例。实验 B 使用三个局部合并示例。两组使用相同的 36 个训练尺寸组、3,600 条训练样本和固定测试任务，各训练五轮。4×4 留作生成测试。

## 36 尺寸实验的两个目录

- **A/B 共用的数据准备**在 [icl_lora_context_expanded_v1](icl_lora_context_expanded_v1)。[数据报告](icl_lora_context_expanded_v1/DATA_REPORT.md)记录同一批 3,600 条训练样本如何筛选和划分，以及各尺寸的几何核验结果。实际训练样本随数据包单独提供。
- **A/B 各自的评测结果**在 [icl_lora_train_expanded_v1](icl_lora_train_expanded_v1)。[A](icl_lora_train_expanded_v1/A) 和 [B](icl_lora_train_expanded_v1/B) 各自保存五轮固定测试和最终 30 个 4×4 用例的结果。[最终对照](icl_lora_train_expanded_v1/comparison.json)可直接比较两组的提议、接受和最终使用情况。

## 完整目录索引

| 文件夹 | 实验与条件 | 内容 |
|---|---|---|
| [icl_report](icl_report) | 早期纯 ICL，使用未加实验 LoRA 的 BrickGPT，比较不同 4×4 提示策略 | 实验报告 Markdown 和 PDF |
| [icl_lora_eval_600_v1](icl_lora_eval_600_v1) | 三尺寸 LoRA，600 条训练样本、三轮。1×3 比较微调前后，其余尺寸只测微调后 | 生成测试汇总 |
| [icl_lora_context_seven_consecutive_v2](icl_lora_context_seven_consecutive_v2) | 七尺寸数据准备阶段回查上一版三尺寸模型的 10 个 4×4 用例 | 逐例候选诊断，记录 4×4 提议及各类拒绝 |
| [icl_lora_train_seven_consecutive_v1](icl_lora_train_seven_consecutive_v1) | 七尺寸 LoRA，1,400 条训练样本、三轮。每轮固定测 5 个 2×3 和 10 个 4×4 | 正式逐轮汇总在 `epoch_tests/`，`smoke_epoch_tests/` 为启动预检 |
| [icl_lora_eval_epoch2_4x4_30](icl_lora_eval_epoch2_4x4_30) | 七尺寸模型第 2 轮 checkpoint，固定 30 个 4×4 任务 | 提议、接受和最终使用汇总 |
| [icl_lora_eval_final_4x4_30](icl_lora_eval_final_4x4_30) | 七尺寸模型第 3 轮 checkpoint，同一批 30 个 4×4 任务 | 与第 2 轮同口径的汇总 |
| [icl_lora_context_expanded_v1](icl_lora_context_expanded_v1) | 36 尺寸 A/B 共用数据，按替换占比取前 500，再选搭建序列最短的 250 个结构 | 各组候选数、划分规模和几何核验结果 |
| [icl_lora_train_expanded_v1](icl_lora_train_expanded_v1) | A/B 各训练五轮。每轮固定测 5 个 2×3、5 个 3×8、10 个 4×4；第 5 轮 checkpoint 另测 30 个 4×4 | `A/` 和 `B/` 各含逐轮与最终汇总，`comparison.json` 为最终对照 |
| [visualizations](visualizations) | 最终 4×4 测试的前 10 个固定用例，A、B 各 10 张 | [并排图集](GALLERY.md) |

[实验总报告](brickgpt_lora_icl_experiment_report.md)包含数据构建、两版提示词和跨阶段结果。测试汇总分别记录出现过目标砖提议的任务数、提议总数、接受与拒绝次数、最终保留目标砖的任务数，以及正常结束且几何合法的任务数。
