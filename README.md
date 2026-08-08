# Kaggle Accelerator Inference

Run Hugging Face LLMs on Kaggle GPUs or TPU v5e-8, keep the loaded model alive between cells, stream output, and log hardware/generation metrics to CSV.

## What It Supports

- `llama.cpp`: GGUF file URLs, local GGUF files, or a repository plus `--filename`.
- `vllm`: standard Hugging Face model repositories supported by vLLM.
- `sglang`: standard Hugging Face model repositories supported by SGLang or SGL-JAX.
- Automatic GPU/TPU detection. TPU runs use all eight v5e chips by default; GPU runs use all detected GPUs.
- Automatic safe context selection from model architecture, weight size, device memory, and model limits.
- Automatic maximum output budget from `max_context_window - input_tokens`.
- Explicit `--context`, `--max-tokens`, and `--devices` values act as caps.
- Persistent model server reuse across repeated `!kgpu run` cells.
- Live max-context, input-token, output-token, output-limit, accelerator, CPU, and RAM statistics.
- CSV history at `/root/kaggle-gpu-inference/logs/inference_runs.csv`, mirrored to `/kaggle/working/inference_runs.csv`.
- Preflight checks reject model weights or requested contexts that exceed safe RAM/VRAM estimates before a new server loads.

Compute throughput and GPU memory throughput are estimates based on NVML utilization multiplied by Tesla T4 peak FP16 tensor throughput (65 TFLOPS) and memory bandwidth (320 GB/s). NVML does not expose measured FLOP counts. GPU-CPU throughput uses measured NVML PCIe RX/TX counters when the driver supports them.

## Kaggle Setup

Enable an accelerator and Internet in the Kaggle notebook settings. For a GPU notebook:

```bash
!git clone https://github.com/YLiu95/kaggle-gpu-inference.git /root/kaggle-gpu-inference
!pip install -e /root/kaggle-gpu-inference
!kgpu setup --engine llama.cpp
```

`kgpu` automatically reads `HF_TOKEN` from Kaggle Secrets when needed. It never writes the token to the repository or logs.

GPU setup never imports JAX or performs TPU setup. TPU setup never initializes NVML, compiles CUDA code, or exports CUDA device selection.

## Kaggle TPU v5e-8

Install only the TPU engine you intend to use. vLLM TPU requires Python 3.11+; SGL-JAX requires Python 3.12+.

```bash
!pip install -e /root/kaggle-gpu-inference
!kgpu setup --engine vllm --tpu
```

```bash
!kgpu run "https://huggingface.co/google/gemma-4-26B-A4B" \
  --engine vllm \
  --tpu \
  --temperature 0.7 \
  --thinking true \
  --prompt "Design a Python pairs-trading backtest for PEP and KO."
```

For SGL-JAX:

```bash
!kgpu setup --engine sglang --tpu
!kgpu run "Qwen/Qwen3-8B" --engine sglang --tpu --prompt "Explain TPU tensor parallelism."
```

`--tpu` is optional when TPU runtime markers are visible, and is useful in tunneled sessions that hide those markers. Both TPU engines set tensor parallel size to all detected chips (eight on v5e-8). SGLang TPU launches through `sgl_jax.launch_server`; it does not install or launch the CUDA SGLang runtime.

The llama.cpp build uses two CPU jobs to stay within Kaggle's RAM limit. Model downloads use the Hugging Face cache under `/root/kaggle-gpu-inference/models` and are not repeated when already complete.

## Run A GGUF Model

The URL may be copied directly from a Hugging Face file page:

```bash
!kgpu run "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/main/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --engine llama.cpp \
  --gpus 2 \
  --context 4096 \
  --max-tokens 256 \
  --temperature 0.7 \
  --prompt "Explain mixture-of-experts routing in plain English."
```

Run another prompt with the same settings to reuse the model already in VRAM:

```bash
!kgpu run "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/main/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" --engine llama.cpp --gpus 2 --context 4096 --prompt "Now give a concrete example."
```

Thinking mode is disabled by default so Qwen3-family models return an answer within the output budget. Use `--thinking true` (or the `--thinking` shorthand) to show reasoning and response tokens in separate streaming sections. Use `--thinking false` or `--no-thinking` to disable it. Increase `--max-tokens` because reasoning consumes that same budget. Reasoning tokens are included in output-token speed metrics and CSV `output_length_*` totals, with separate `reasoning_*` and `response_*` columns retained for analysis.

```bash
!kgpu run "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-UD-Q8_K_XL.gguf" \
  --engine llama.cpp \
  --gpus 1 \
  --context 4096 \
  --max-tokens 1024 \
  --thinking true \
  --prompt "How many r letters are in the word raspberry?"
```

## MTP Speculative Decoding

For llama.cpp-compatible MTP models, enable multi-token prediction with `--spec-type draft-mtp`. Unsloth recommends starting with `--spec-draft-n-max 2`; values from 1 through 6 can be benchmarked for the specific model and GPU. MTP requires roughly 2 GB of additional RAM/VRAM headroom; preflight reserves system-memory headroom plus GPU workspace before launch.

```bash
!kgpu run "https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/blob/main/Qwen3.5-9B-UD-Q8_K_XL.gguf" \
  --engine llama.cpp \
  --gpus 1 \
  --context 8192 \
  --max-tokens 8192 \
  --thinking true \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --prompt "Design a Python backtest for a PEP/KO pairs-trading strategy."
```

Changing either speculative-decoding option restarts the managed server once; subsequent commands with the same model and settings reuse it. MTP configuration is recorded in the CSV columns `spec_type`, `spec_draft_n_max`, and `mtp_enabled`.

Use one GPU with `--devices 1`. Other GPUs remain available, but the model must fit in approximately 90% of the selected device memory.

## vLLM And SGLang

These engines load repository-format models, not GGUF files. Install only the engine needed because their dependency sets are large:

```bash
!kgpu setup --engine vllm
!kgpu run "Qwen/Qwen2.5-7B-Instruct" --engine vllm --devices 2 --context 4096 --prompt "Write a CUDA learning plan."
```

```bash
!kgpu setup --engine sglang
!kgpu run "Qwen/Qwen2.5-7B-Instruct" --engine sglang --devices 2 --context 4096 --prompt "Write a CUDA learning plan."
```

Package and CUDA compatibility changes quickly for vLLM/SGLang. Installing both in one Kaggle environment is discouraged because their pinned dependencies can conflict.

## Control Commands

```bash
!kgpu status
!kgpu clear-vram
```

`clear-vram` stops the server managed by this tool. It does not kill unrelated GPU processes.

## Output And Logs

The top live section contains model/engine and accelerator details, maximum context window, input tokens, real-time output tokens, calculated maximum output tokens, TTFT, token speed, accelerator memory, CPU, and RAM. Streaming text appears in the middle. A fixed summary appears at completion.

Each CSV row includes the model, engine, accelerator type/count, selected context and output limits, prompt/output text and token counts, TTFT, token-speed statistics, and sampled hardware metrics. Context calculation uses Hugging Face or GGUF architecture metadata and falls back to 4,096 only when metadata is unavailable.

## Development

```bash
cd /root/kaggle-gpu-inference
pip install -e '.[test]'
pytest -q
```