# Using a New Model

Any model `transformers` can load with `AutoModelForCausalLM` works. You select it with
`--model`, which takes a **HuggingFace repo id** or a **local path**.

## Any HuggingFace model

```bash
# a small model, fine on CPU
cap --device cpu run --model gpt2 --type factuality \
  --data mydata.csv --text-column srcs --label-column label \
  --output runs/gpt2

# a larger instruct model on GPU
cap --device cuda run --model Qwen/Qwen3-8B --type faithfulness \
  --data mydata.csv --text-column srcs --label-column label \
  --output runs/qwen
```

A local checkout works the same way:

```bash
cap --device cuda run --model /models/my-finetune --type factuality ...
```

## Choosing the device

`--device` accepts `cpu`, `cuda`, `mps`, or `auto` (the default, which picks CUDA when
available and CPU otherwise). It is a **global** flag, so it comes before the subcommand.
Rough bf16 VRAM: Qwen3-8B ≈ 16 GB, Qwen3-14B ≈ 28 GB; small models like `gpt2` or
`google/gemma-2-2b` run on CPU or a few GB of GPU.

## When to use `--trust-remote-code`

Some repos ship custom modelling code that must execute for the model to load. CAP keeps
this **disabled by default**. If loading fails with a message about `trust_remote_code`,
and you trust the publisher, enable it:

```bash
cap --trust-remote-code --device cuda run --model some-org/custom-arch --type factuality ...
```

Never enable it for a model you do not trust — it runs arbitrary Python from the repo on
your machine.

## Notes

- CAP hooks **every layer** and records the last-token activation, so capture time and
  `activations.h5` size scale with model depth and prompt count. Start with `--n-samples`
  to cap rows per group while you iterate.
- The prompt template matters: an instruct model may respond better to `faithfulness` than
  `factuality`. Try a `--prompt` of your own if neither fits — it just needs `{text}`.
