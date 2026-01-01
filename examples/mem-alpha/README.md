# AReaL lightweight reproduction of MEM-$\alpha$

## Adaptation Notes

Adapted from https://github.com/wangyu-ustc/Mem-alpha. The original setup uses
`n_samples=8` and a Qwen3-32B external LLM; our configuration uses `n_samples=4` and a
Qwen3-14B external LLM to keep the reproduction lightweight. We also cap per-turn
`context_length` at 16384 tokens; if you have more compute, consider increasing it. On
5x H20 GPUs for training and 4x H20 GPUs for eval, expect ~20 hours for one training
epoch and ~2 hours for eval.

## Data

https://huggingface.co/datasets/YuWangX/Memalpha/tree/main

## Environment (uv)

```bash
uv venv .venv
source .venv/bin/activate
# Follow https://inclusionai.github.io/AReaL/tutorial/installation.html to install AReaL **then** run the following command:
uv pip install -r examples/mem-alpha/requirements.txt
```

## Train

5-GPU layout (example: 14B external + 4B rollout, matches `allocation_mode`):

- external judge: 3 GPUs
- rollout (sglang): 1 GPU
- train (fsdp): 1 GPU

1. Launch the external SGLang server (LLM judge / QA). Keep the external engine's
   `experiment_name` aligned with `external_engine.experiment_name`, and set the
   `trial_name` to match the training run.

```bash
CUDA_VISIBLE_DEVICES=2,3,4 python3 -m areal.launcher.local --config examples/mem-alpha/config.yaml \
  allocation_mode=sglang:d3p1t1 \
  experiment_name=external-engine \
  trial_name=<TRIAL_NAME> \
  actor.path=<MODEL_PATH> # Qwen3-14B
```


2. Run training:

Example CLI:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m areal.launcher.local examples/mem-alpha/train.py \
  --config examples/mem-alpha/config.yaml \
  experiment_name=mem-alpha \
  trial_name=<TRIAL_NAME> \
  actor.path=<MODEL_PATH> \ # Qwen3-4B
  gconfig.agent.adv_estimator=grpo \
  train_dataset.path=<DATA_DIR>
```

## Eval

1 rollout + 3 external layout (example):

- external judge: 3 GPUs
- rollout (sglang): 1 GPU

Start the external engine:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m areal.launcher.local --config examples/mem-alpha/config_eval.yaml \
  allocation_mode=sglang:d3p1t1 \
  experiment_name=external-engine \
  trial_name=<TRIAL_NAME_EVAL> \
  actor.path=<MODEL_PATH> # Qwen3-14B
```

Then run eval:

```bash
CUDA_VISIBLE_DEVICES=3 python3 -m areal.launcher.local examples/mem-alpha/eval.py \
  --config examples/mem-alpha/config_eval.yaml \
  allocation_mode=sglang:d1+eval \
  actor.path=<CHECKPOINT_PATH> \
  valid_dataset.path=<DATA_DIR> \
  trial_name=<TRIAL_NAME_EVAL>
```
