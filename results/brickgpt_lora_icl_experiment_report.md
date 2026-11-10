# BrickGPT 新积木尺寸的上下文学习实验

## 1. 实验思路

给 BrickGPT 的原始砖库每次增加一个尺寸组，通过三个 ICL 示例说明该尺寸怎么用。训练时让模型学习使用若干新增尺寸，生成测试时用训练未见的 4×4 检查能否迁移。评估以生成过程中是否尝试放置目标砖、成品是否保留目标砖为主，任务仍然是按描述生成完整结构。

早期的纯 ICL 实验尝试了完整示例、替换示例、预算约束和局部填充等策略。在此前的 45 次椅子序列测试和后续 105 次策略测试中，均未观察到完整且正确使用 4×4 的椅子。后续因此加入 LoRA 微调，再逐步增加训练尺寸和改变示例形式。[早期纯 ICL 实验报告](icl_report/experiment_report.md)

## 2. 数据构建与实验设置

| 阶段 | 新增训练尺寸组，旋转归为同组 | 结构筛选与改装 | 实际训练 |
|---|---|---|---:|
| 三尺寸 | 1×3、2×3、2×8 | 1×6 或 2×6 拆为两块，平行紧邻的 1×8 合并。每组先按替换占比取前 1,200，再取搭建顺序字符数最短的 600，按 480/60/60 划分 | 3×200＝600 条，3 轮 |
| 七尺寸 | 原三组加 6×6、3×4、3×3、3×8 | 保留 1×3、2×3 的拆分。合并类要求原砖同层、步骤连续并恰好铺满矩形。每组取占比前 500，再取最短 250，按 200/25/25 划分 | 7×200＝1,400 条，3 轮 |
| 三十六尺寸与三十六尺寸短提示示例 | 边长 1–9，旋转去重后共 45 组，扣除原砖库 8 组和留测的 4×4，共 36 组 | 所有训练组使用同层、连续步骤的整砖合并。每组取占比前 500，再取最短 250，按 200/25/25 划分，从训练候选中抽 100 个目标结构 | 每组实验 36×100＝3,600 条，5 轮 |

替换占比按被替换原砖数除以原结构砖数计算。数据来自 StableText2Brick。以结构 ID 划分，同一结构的多条 caption 和改装版本放在同一分区。每条实际训练样本固定一个目标 caption，训练条数按目标结构与尺寸组计。每条 prompt 内三个示例和目标对应四个不同结构 ID。

扩展数据扫描了 47,389 个结构，核验了 487,579 个候选改装版本和 1,207,361 处合并。合并砖的起点由原砖覆盖格的最小坐标确定，逐处校验新砖覆盖格与原砖并集相同，并核验完整改装结构无碰撞、无越界。扩展阶段每个实验另有 900 条验证和 900 条常规测试样本准备数据。训练中的验证 loss 实际固定使用 5 条，逐轮生成和最终生成使用单独锁定的测试清单。[扩展数据报告](icl_lora_context_expanded_v1/DATA_REPORT.md)

训练从同一冻结的 BrickGPT 起点添加新 LoRA，基座为 Llama-3.2-1B-Instruct。使用 RTX 4060 8GB、BF16、rank 16、学习率 1e-4、单条输入和梯度累积 8，仅监督最后的完整搭建答案。扩展阶段每轮 450 次更新，每个实验共 2,250 次，轮末与总进度每 20% 的保存点重合。

## 3. 上下文怎么构建

三尺寸、七尺寸和三十六尺寸 使用官方 few-shot 模板。允许尺寸为原始砖库加本条任务的一个尺寸组，最终任务明确要求使用该组新砖。三个上下文示例的输入是各自的 caption 和新尺寸要求，输出是对应的完整改装搭建顺序。

三十六尺寸短提示示例 保留同样的目标、caption、完整答案和三个示例结构，把每个示例改为一次局部合并。使用固定种子从该结构通过几何核验的合并记录中随机抽取一次。输入列出原砖的尺寸、坐标和顺序，输出给出替换后的单块新砖。

#### 三十六尺寸（A）

```text
Allowed brick dimensions are {原始砖库尺寸}, {本次新增尺寸组}.

### Input:
{示例 1 的 caption 和使用新砖要求}
### Output:
{示例 1 改装后的完整搭建顺序}

### Input:
{示例 2 的 caption 和使用新砖要求}
### Output:
{示例 2 改装后的完整搭建顺序}

### Input:
{示例 3 的 caption 和使用新砖要求}
### Output:
{示例 3 改装后的完整搭建顺序}

### Input:
{目标任务的 caption 和使用新砖要求}
### Output:
{目标结构改装后的完整搭建顺序，仅训练时提供}
```

#### 三十六尺寸短提示示例（B）

```text
Allowed brick dimensions are {原始砖库尺寸}, {本次新增尺寸组}.

### Input:
{示例 1 的使用新砖要求和待替换原砖指令}
### Output:
{示例 1 的单块替换砖尺寸和坐标}

### Input:
{示例 2 的使用新砖要求和待替换原砖指令}
### Output:
{示例 2 的单块替换砖尺寸和坐标}

### Input:
{示例 3 的使用新砖要求和待替换原砖指令}
### Output:
{示例 3 的单块替换砖尺寸和坐标}

### Input:
{目标任务的 caption 和使用新砖要求}
### Output:
{目标结构改装后的完整搭建顺序，仅训练时提供}
```

#### 实际测试提示词对照

以下是最终 30 个 4×4 测试用例中 `4x4_case_20` 的两版完整用户提示词。两版使用同一个目标任务和同样的三个示例结构。

**三十六尺寸（A）**

```text
Create a LEGO model of the input. Format your response as a list of bricks: <brick dimensions> <brick position>, where the brick position is (x,y,z).
Allowed brick dimensions are 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2, 4x4.
All bricks are 1 unit tall.

### Input:
The bench is elongated with a flat upper surface and staggered lower supports. Use the newly added 4x4 bricks.

Each line of your output should be a LEGO brick in the format `<brick dimensions> <brick position>`. For example:
2x4 (2,1,0)
DO NOT output any other text. Only output LEGO bricks. The first brick should have a z-coordinate of 0.

Here are some example LEGO models:

### Input:
Vessel with a rectangular core, topped by tiered, angular layers. Use the newly added 4x4 bricks.

### Output:
1x4 (3,15,0)
1x8 (3,7,0)
1x1 (2,7,0)
2x4 (2,3,0)
2x1 (1,18,0)
2x4 (1,14,0)
2x6 (1,8,0)
1x6 (4,14,1)
1x2 (4,11,1)
2x1 (3,13,1)
2x6 (3,5,1)
1x2 (3,3,1)
2x6 (2,14,1)
1x1 (2,13,1)
1x1 (2,10,1)
1x2 (2,2,1)
2x6 (1,4,1)
2x1 (0,19,1)
2x6 (0,13,1)
4x2 (0,11,1)
2x1 (0,10,1)
1x1 (0,9,1)
1x2 (0,7,1)
1x1 (4,8,2)
1x2 (4,6,2)
1x1 (3,19,2)
2x1 (3,18,2)
2x4 (3,14,2)
1x1 (3,6,2)
4x2 (2,10,2)
4x1 (2,9,2)
2x2 (2,7,2)
2x2 (2,3,2)
1x1 (2,2,2)
2x6 (1,14,2)
4x1 (1,5,2)
1x1 (1,4,2)
1x1 (0,18,2)
1x4 (0,14,2)
6x2 (0,12,2)
2x6 (0,6,2)
2x6 (3,9,3)
1x1 (2,14,3)
1x1 (2,11,3)
1x4 (1,11,3)
2x2 (1,9,3)
4x4 (1,11,4)


### Input:
An elongated bus, rectangular with a stepped front, smooth top, and vertical sides. Use the newly added 4x4 bricks.

### Output:
2x1 (3,16,0)
2x2 (3,14,0)
2x2 (3,5,0)
1x2 (2,3,0)
2x1 (0,16,0)
2x2 (0,14,0)
2x2 (0,5,0)
1x2 (4,18,1)
1x8 (4,10,1)
1x6 (4,4,1)
1x2 (2,2,1)
4x1 (0,19,1)
4x4 (0,15,1)
4x2 (0,13,1)
4x1 (0,12,1)
4x4 (0,8,1)
4x4 (0,4,1)
1x6 (4,5,2)
1x1 (4,4,2)
2x6 (3,14,2)
2x1 (3,13,2)
2x2 (3,11,2)
1x8 (3,3,2)
1x4 (2,16,2)
1x1 (2,15,2)
1x8 (2,7,2)
1x1 (2,5,2)
1x1 (2,2,2)
2x2 (1,3,2)
2x2 (0,18,2)
2x6 (0,12,2)
2x1 (0,11,2)
2x6 (0,5,2)
1x1 (0,4,2)
1x2 (4,18,3)
1x8 (4,10,3)
2x6 (3,4,3)
2x6 (2,10,3)
1x1 (2,3,3)
2x6 (1,4,3)
4x4 (0,16,3)
2x6 (0,10,3)
1x4 (0,6,3)
1x1 (0,4,3)
1x4 (4,15,4)
1x8 (4,7,4)
4x1 (1,19,4)
1x8 (0,12,4)
1x2 (0,10,4)
4x1 (0,9,4)
4x2 (0,7,4)
2x6 (3,14,5)
2x6 (3,8,5)
2x1 (3,7,5)
1x4 (2,16,5)
1x8 (2,8,5)
1x1 (2,7,5)
2x6 (0,14,5)
2x1 (0,13,5)
2x6 (0,7,5)


### Input:
A small couch with sturdy base, flat seating area, and elevated backrest. Use the newly added 4x4 bricks.

### Output:
2x1 (16,19,0)
2x6 (16,11,0)
2x6 (14,11,0)
6x2 (12,17,0)
2x6 (12,11,0)
2x6 (10,11,0)
8x1 (8,19,0)
2x6 (8,11,0)
6x2 (6,17,0)
2x6 (6,11,0)
2x6 (4,11,0)
2x6 (2,11,0)
8x1 (0,19,0)
6x2 (0,17,0)
2x6 (0,11,0)
1x8 (17,12,1)
1x1 (17,11,1)
8x1 (9,19,1)
8x1 (9,11,1)
8x1 (1,19,1)
8x1 (1,11,1)
1x8 (0,12,1)
1x1 (0,11,1)
1x8 (17,11,2)
2x1 (16,19,2)
1x1 (16,11,2)
6x1 (10,17,2)
6x2 (10,15,2)
6x2 (10,13,2)
6x2 (10,11,2)
8x1 (8,19,2)
6x2 (4,15,2)
6x2 (4,13,2)
6x2 (4,11,2)
8x1 (2,17,2)
2x6 (2,11,2)
1x1 (1,11,2)
8x1 (0,19,2)
1x8 (0,11,2)
1x2 (17,18,3)
2x6 (16,11,3)
1x6 (15,11,3)
2x6 (13,11,3)
6x1 (12,17,3)
6x2 (11,18,3)
1x1 (11,17,3)
2x6 (11,11,3)
2x6 (9,14,3)
4x4 (5,14,3)
6x2 (5,12,3)
6x2 (3,18,3)
2x6 (3,12,3)
8x1 (3,11,3)
1x8 (2,12,3)
1x1 (2,11,3)
2x2 (0,18,3)
2x1 (0,17,3)
2x6 (0,11,3)
1x6 (17,12,4)
1x1 (17,11,4)
2x1 (15,15,4)
2x4 (15,11,4)
6x2 (12,18,4)
6x2 (11,16,4)
2x2 (10,18,4)
1x2 (10,16,4)
2x4 (8,16,4)
1x2 (7,17,4)
1x1 (7,16,4)
4x1 (3,16,4)
6x2 (1,17,4)
2x6 (1,11,4)
8x1 (0,19,4)
1x8 (0,11,4)
4x2 (12,16,5)
6x2 (6,16,5)
4x2 (2,16,5)
2x2 (14,16,6)
6x2 (8,16,6)
6x2 (2,16,6)
6x2 (9,16,7)
6x2 (3,16,7)
1x2 (2,16,7)


Do NOT copy the examples, but create your own LEGO model for the following input.

### Input:
The bench is elongated with a flat upper surface and staggered lower supports. Use the newly added 4x4 bricks.

### Output:
```

**三十六尺寸短提示示例（B）**

```text
Create a LEGO model of the input. Format your response as a list of bricks: <brick dimensions> <brick position>, where the brick position is (x,y,z).
Allowed brick dimensions are 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2, 4x4.
All bricks are 1 unit tall.

### Input:
The bench is elongated with a flat upper surface and staggered lower supports. Use the newly added 4x4 bricks.

Each line of your output should be a LEGO brick in the format `<brick dimensions> <brick position>`. For example:
2x4 (2,1,0)
DO NOT output any other text. Only output LEGO bricks. The first brick should have a z-coordinate of 0.

The examples show how consecutive bricks can be replaced by one new brick while covering exactly the same cells. For the final task, generate the full assembly sequence.

### Input:
Use the new 4x4 brick to replace these consecutive bricks.
2x4 (3,11,4)
2x4 (1,11,4)

### Output:
4x4 (1,11,4)

### Input:
Use the new 4x4 brick to replace these consecutive bricks.
4x1 (0,19,3)
4x2 (0,17,3)
4x1 (0,16,3)

### Output:
4x4 (0,16,3)

### Input:
Use the new 4x4 brick to replace these consecutive bricks.
4x2 (5,16,3)
4x2 (5,14,3)

### Output:
4x4 (5,14,3)

Do NOT copy the examples, but create your own LEGO model for the following input.

### Input:
The bench is elongated with a flat upper surface and staggered lower supports. Use the newly added 4x4 bricks.

### Output:
```

训练、验证与生成测试均采用各自对应的示例格式。A、B 每轮用相同的 5 个 2×3、5 个 3×8 和 10 个 4×4 任务，最终再以第 5 轮 checkpoint 测相同的 30 个 4×4 任务。固定测试清单和两版提示词随 A/B 数据包提供。

## 4. 实验结果

提议指模型生成一条目标尺寸的候选放置指令，随后接受或拒绝。一个任务可产生多次提议。成品含目标砖也包括生成中止时保存的部分输出。“正常完成且含目标砖”要求生成正常结束、几何合法且保留目标砖。评测未要求参考形状完全一致，也未执行物理稳定性检查。

### 三尺寸与七尺寸阶段

三尺寸模型在已见尺寸上可以用新砖。1×3 从微调前 0/10 提升到微调后 10/10，正常完成且含新砖为 5/10。微调后 2×3 为 9/10，2×8 为 3/9，后者只有 9 个已完成用例。**4×4 在 10 个测试任务中没有出现一次提议**，225 次候选拒绝全部来自其他尺寸。[三尺寸结果](icl_lora_eval_600_v1/RESULTS.md) · [4×4 候选诊断](icl_lora_context_seven_consecutive_v2/diagnostic_4x4.json)

七尺寸模型在固定的 5 个 2×3 测试中，三轮均为 5/5 成品含新砖。固定的 10 个 4×4 测试中，三轮分别提议 0、6、1 次，全部未被接受，三轮成品均为 0/10。扩展到 30 例后，第二、三轮 checkpoint 都各有 1 例成品含 4×4。[七尺寸逐轮记录](icl_lora_train_seven_consecutive_v1/epoch_tests)

### 30 个 4×4 测试目标的阶段对照

| 模型 | 提议过的任务 | 提议总次数 | 接受 / 拒绝 | 成品含 4×4 | 正常完成且含 4×4 |
|---|---:|---:|---:|---:|---:|
| 七尺寸，第 2 轮 | 3/30 | 8 | 1 / 7 | 1/30 | 1/30 |
| 七尺寸，第 3 轮 | 2/30 | 2 | 1 / 1 | 1/30 | 0/30 |
| 三十六尺寸，第 5 轮 | 21/30 | 44 | 41 / 3 | 20/30 | 17/30 |
| 三十六尺寸短提示示例，第 5 轮 | 23/30 | 96 | 64 / 32 | 23/30 | 21/30 |

[七尺寸第 2 轮](icl_lora_eval_epoch2_4x4_30/summary.json) · [七尺寸第 3 轮](icl_lora_eval_final_4x4_30/summary.json) · [三十六尺寸](icl_lora_train_expanded_v1/A/final_4x4/summary.json) · [三十六尺寸短提示示例](icl_lora_train_expanded_v1/B/final_4x4/summary.json)

30 例对照复用了目标用例。扩展阶段同时改变了训练尺寸数、样本量、轮次和部分 ICL 示例，阶段差异应作为整套方案的改进来解释。三十六尺寸与三十六尺寸短提示示例 共用样本和测试任务，可以直接对比两种示例形式在本次运行中的表现。

### 三十六尺寸与三十六尺寸短提示示例 的逐轮固定测试

表中统计成品含目标砖的任务数。

#### 三十六尺寸（A）

| 轮次 | 2×3（5 例） | 3×8（5 例） | 4×4（10 例） |
|---|---:|---:|---:|
| 1 | 1/5 | 1/5 | 5/10 |
| 2 | 3/5 | 4/5 | 6/10 |
| 3 | 3/5 | 4/5 | 5/10 |
| 4 | 3/5 | 4/5 | 5/10 |
| 5 | 3/5 | 3/5 | 5/10 |

[三十六尺寸逐轮评测](icl_lora_train_expanded_v1/A/epoch_tests)

#### 三十六尺寸短提示示例（B）

| 轮次 | 2×3（5 例） | 3×8（5 例） | 4×4（10 例） |
|---|---:|---:|---:|
| 1 | 3/5 | 2/5 | 5/10 |
| 2 | 5/5 | 2/5 | 7/10 |
| 3 | 5/5 | 4/5 | 7/10 |
| 4 | 5/5 | 2/5 | 6/10 |
| 5 | 5/5 | 3/5 | 8/10 |

[三十六尺寸短提示示例逐轮评测](icl_lora_train_expanded_v1/B/epoch_tests)

最终 30 例中，B 比 A 多 3 例成品含 4×4，多 4 例达到正常完成标准。B 的 4×4 提议次数从 44 增到 96，单次接受率从 93% 降到 67%。局部合并示例在本次测试中提高了 4×4 的尝试频率和最终使用覆盖率。已见尺寸上，B 对 2×3 的优势较稳定，3×8 的结果随轮次波动。
