# Qwen3.6-35B-A3B L8 baseline

A reduced-model baseline that keeps **the vendored recipe's own parallel block**
instead of a re-partitioned one:

| | vendored `train_online.yaml` | this baseline |
|---|---|---|
| `accelerator.tp/cp/ep/pp` | 1 / 2 / 2 / 1 | unchanged |
| `fsdp_config.dp_shard / edp_shard` | 8 / 4 | unchanged |
| `activation_checkpoint.mode` | full | unchanged |
| `dataset.data_transform.max_seq_len` | 4096 | unchanged |
| `training.train_iters` | 100 | unchanged |
| `plan_overrides` | 3 | unchanged |
| model / data / output paths | author's paths | this deployment |
| `debug.check_nan_inf` | false | false |

The point of the file is that a reviewer can diff it against
`hyper_parallel/models/qwen3_6_moe/recipes/train_online.yaml` and see only path
substitutions plus `model.load_base_model: true` (which is already the
`from_pretrained` default) and `training.logging_steps: 1`. The optimizer, LR
schedule, loss, precision, determinism, data pipeline and every `plan_overrides`
entry are inherited, not re-chosen.

## Why EP2 + CP2 still needs eight dies

`dp_shard` is the FSDP shard group that absorbs `cp`/`ep`, so the layout has to
satisfy two equalities (`hyper_parallel/distributed/mesh.py`):

```text
dp_size = world // (tp * cp * pp)                 = 8 // 2   = 4
dp_replicate * dp_shard == dp_size * cp            = 1 * 8 = 4 * 2   ✓
expert domain = dp * cp * tp = 8 = EP2 * EDP4                          ✓
```

Same eight dies, three views: `(dp=4, cp=2, tp=1)`, one flattened 8-way FSDP
shard, and `(edp_shard=4, ep=2)`. Sixteen dies would fail the second equality
(`8 != 16`), so this is not a bigger-cluster configuration.

## Dataset

`open-r1/OpenR1-Math-220k`, config `default`, split `train`, **shard 0 of 10**
(`train-00000-of-00010.parquet`, 213.6 MB, 9,374 rows).

```text
sha256 ccc3a95e4875dc4cff9bec2f0fb1c93e4c9cbb24fb24853fd0b96c416e1a263c
```

It carries a `messages` column with `role`/`content`, so `data_type: sharegpt`,
the chat template and the assistant-only loss mask all stay as the vendored
recipe defines them. Rendered lengths with this model's tokenizer
(`enable_thinking=true`, `reasoning_effort=medium`, `preserve_thinking=true`,
300 samples): p10 2161, **p50 4882**, p90 11873, max 18569, mean 6063 tokens —
**57.7 % of samples reach the 4096 cap**, so the sequence length is actually
exercised rather than nominal.

## Measured status

A three-step smoke on this parallel block (same model slice, fused GDN,
`recompute=full`, seq 4096) fits and runs: peak 24.52 GB of 32 GB, loss
11.4747 → 10.3331 → 9.0704, step time 88.2 s → 42.7 s → 29.2 s. The step time
was still falling, which is what a per-shape kernel autotune looks like, so **no
throughput number is claimed**.

That smoke used the previous revision of the vendored recipe (five
`plan_overrides`); the file here is generated from the current one (three
`plan_overrides`, EP handled inside the module families). **Re-run before
quoting numbers.**

## Run

```bash
MODEL_DIR=/path/to/model-l8-real \
DATA_PATH=/path/to/openr1-math-220k-train-00000.parquet \
TRAIN_ITERS=30 \
PYTHON=/path/to/venv/bin/python \
bash examples/qwen3_6_moe/baseline/launch_baseline.sh
```

Any leftover arguments are forwarded to `scripts.train_lm` as dotted overrides,
so a one-off experiment does not need to edit the YAML.

## Open

- No 30-step (bench window 6–30) or 100-step repeated run yet, so there is no
  median, no cross-run CV and no stability verdict.
- CP at 4096 tokens is what makes the block defensible; at the synthetic
  ~849-token workload the same block was 66–69 % slower than `EP8`, so the
  comparison only means something on long samples.
