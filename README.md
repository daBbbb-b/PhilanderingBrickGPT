# BrickGPT LoRA and in-context learning

Experiment A uses three complete assembly demonstrations. Experiment B uses three local merge demonstrations. Both use the same 36 training size groups and hold out 4×4 for evaluation.

## Experiment reports and results

- [Full experiment report](results/brickgpt_lora_icl_experiment_report.md)
- [Early ICL report](results/icl_report/experiment_report.md)
- [Final 30-case A/B comparison](results/icl_lora_train_expanded_v1/comparison.json)
- [Visualization gallery](results/GALLERY.md) with ten outputs from five matched 4×4 tasks

## Local setup

Use Python 3.12 and a CUDA-enabled PyTorch installation with BF16 support. Clone the upstream dependency at the revision used in the experiments.

```powershell
git clone https://github.com/AvaLovelace1/BrickGPT.git
git -C BrickGPT checkout 9c53c7cb65f9de0df89562cc74b2d2e270fbcab4
python -m pip install -r requirements-stage1.txt
python -m pip install "numpy<2" matplotlib accelerate networkx
```

Place the Llama-3.2-1B-Instruct checkpoint in `temp/models/Llama-3.2-1B-Instruct/` and the original BrickGPT adapter and tokenizer in `temp/models/BrickGPT/`. The runner also accepts a locally cached Llama checkpoint in its configured Hugging Face cache.

## Data and execution

Download the required ZIP files from Google Drive and extract them into the project's `data/` directory. Each ZIP includes its own top-level directory.

| Archive | Location after extraction | Contents |
|---|---|---|
| `brickgpt_icl_ab_v1.zip` | `data/brickgpt_icl_ab_v1/` | Prepared A/B samples, fixed evaluations, and data audit |
| `stabletext2brick_raw_v1.zip` | `data/stabletext2brick_raw_v1/` | Seven original StableText2Brick Parquet shards and source metadata |
| `brickgpt_icl_ab_rebuild_inputs_v1.zip` | `data/brickgpt_icl_ab_rebuild_inputs_v1/` | Normalized source structures and fixed target manifests |

Only the prepared archive is required to train and evaluate. To regenerate it locally, extract the rebuild inputs archive and run `python scripts/prepare_context_expanded.py`. This writes the prepared files to `data/brickgpt_icl_ab_v1/`. The raw archive is for inspecting the original dataset.

```powershell
Expand-Archive -LiteralPath .\brickgpt_icl_ab_v1.zip -DestinationPath .\data
```

Google Drive links are pending. Compare downloads with the accompanying `SHA256SUMS.txt`.

Run `python -u scripts/train_lora_expanded.py`. It trains and evaluates A, then B, and writes progress and checkpoints under `output/icl_lora_train_expanded_v1/`.

The generation syntax mask matches the recorded evaluations and permits brick dimensions 1–8. Training data also includes size groups with a side of 9.

## Trained adapters

The final adapters are stored locally in `A/checkpoints/step_2250/` and `B/checkpoints/step_2250/` under the training output directory. Distribute each PEFT adapter as `adapter_config.json` and `adapter_model.safetensors`. Keep the matching tokenizer files with the release if consumers need them. The adapter expects the original BrickGPT merged into the Llama base. The training state is only needed for resuming training.

Google Drive links will be added after the adapters are uploaded. Data, caches, and weights are excluded from Git.
