---
license: mit
library_name: exllamav3
tags:
  - exllamav3
  - exl3
---

# exllamav3 with half-bit rates (4.5 to 7.5 bpw)

A fork of [turboderp's exllamav3](https://github.com/turboderp-org/exllamav3) v1.6.6, not affiliated with upstream.

Whole-bit steps are a blunt tool when you're trying to fit a big MoE model into a fixed amount of VRAM and RAM.
Upstream has half-bit EXL3 rates up to 3.5 bpw; this fork carries them on to 4.5, 5.5, 6.5 and 7.5, across the
whole path: quantizer, GPU kernels and the CPU expert worker. It also speeds up the fused MoE decode kernel at
5-6 bpw (integer rates included) and makes conversion roughly twice as fast.

**Compatibility:** models that use 4.5-7.5 bpw need this fork; stock exllamav3 can't load them. Everything upstream
loads still loads and runs the same here.

## Install

```bash
git clone https://github.com/hunterxma/exllamav3-half-bit exllamav3 && bash exllamav3/install.sh
```

You need Linux with an NVIDIA GPU (compute 7.5+) and a driver for CUDA 12.6 or newer (12.8+ for RTX 50-series),
`git` and a C++ compiler (`sudo apt install git build-essential`), and about 12 GB of disk. For models bigger than
your VRAM you also want an AVX2 CPU and enough free RAM for the experts that run on it.

No CUDA toolkit needed. The script picks the right PyTorch and CUDA compiler for your driver, builds for your GPU
only and runs a self-test over every bitrate. Expect 15-40 minutes, almost all of it compiling. Then use
`./venv/bin/python` (or `source venv/bin/activate`).

<details>
<summary>Installer options and tested systems</summary>

| Variable | Effect |
|---|---|
| `CUDA_HOME` | use this CUDA toolkit |
| `EXL3_CUDA=12.8` | force a PyTorch CUDA family (12.6 / 12.8 / 12.9 / 13.0) |
| `EXL3_PIP_CUDA=1` | use the pip CUDA compiler even if a toolkit exists |
| `EXL3_ARCH="8.6;12.0"` | build for these architectures |
| `MAX_JOBS` | parallel compile jobs (default: by RAM) |
| `TORCH_SPEC`, `EXL3_VENV` | PyTorch version, venv location |
| `EXL3_NO_SELFTEST=1` | build only |

Clean-environment installs (empty home directory, no CUDA toolkit), RTX 5060 Ti (sm_120), driver 595:

| System | Installer picked | Build |
|---|---|---|
| Ubuntu, glibc 2.43, g++ 15, apt `nvcc` 12.4 only | PyTorch cu130 + pip CUDA 13.2 | 14 min (idle CPU, 6 jobs) |
| Ubuntu 24.04 container, g++ 13, no toolkit (earlier version) | PyTorch cu128 + CUDA 12.9.2 redist | 22 min |

CUDA 12.x headers don't compile against glibc 2.41+; the installer tells you, and the fix is a driver for CUDA 13
(R580+). Every architecture from sm_75 to sm_121 has compiled; only sm_120 has been run.

</details>

## Use it

Convert at a half-bit rate (same converter; `-b` now also takes 4.5, 5.5, 6.5 and 7.5):

```bash
python convert.py -i /path/to/hf-model -o /path/to/out -w /path/to/work -b 4.5
```

Or give each tensor its own rate with a recipe (later lines override earlier ones, wildcards allowed):

```yaml
# recipe.yaml  ->  python convert.py -i ... -o ... -w ... --recipe recipe.yaml
tensors:
  "*.mlp.experts.*.down_proj": 5
  "*.mlp.experts.*.gate_proj": 4.5
  "*.mlp.experts.*.up_proj": 4.5
  "*.self_attn.*": 8
```

Run it like any EXL3 model, e.g. `python examples/chat.py -m /path/to/out -mode qwen35`. If the model is larger
than your VRAM, `EXL3_MOE_CPU_SPLIT=N` runs N experts of every MoE layer on the CPU.

For the settings the benchmarks below use, give the installer the model: `MODEL=/path/to/model bash install.sh`. It
fits the model to your GPUs (all of them, fewest experts on the CPU, MTP drafts to match) and writes `run.sh` (chat)
and `serve.sh` (OpenAI-compatible server).

## Does half a bit help?

Qwen3.6-35B-A3B, KLD against BF16 on the same 30,149-token held-out set:

| Build | Rates | Size | KLD mean | Same top-1 |
|---|---|---|---|---|
| integer rates only | 4-8 | 24.61 GB | 0.0166 | 97.4% |
| half-bit rates, 4.5 in the middle layers | 4.5-8 | 23.87 GB | 0.0157 | 97.2% |
| half-bit rates + rate tiers (our released model) | 4.5-8 | 24.51 GB | 0.0156 | 97.4% |

The half-bit builds come out smaller and closer to BF16. The allocations differ in more than just the half steps,
so treat this as "half bits make better allocations reachable", not a pure A/B of the rate itself.

## What's changed

| Area | Change | Checked against |
|---|---|---|
| Format, GEMM | 4.5-7.5 bpw in packing, reconstruct, GEMM and MoE kernels | 601 kernel tests |
| Decode MoE kernel | 2 blocks per SM up to 6 bpw | identical output to the previous kernel |
| Quantizer | register-resident half-rate trellis search | bit-identical to the reference kernel |
| Converter | routed-token sensitivity, no host syncs, faster g-scale search; `EXL3_CONVERT_REUSE=<build>` copies unchanged tensors from an earlier build | fp32-identical statistics |
| CPU experts | half rates on AVX2, AVX-512 and the scalar path; packed planar layout + prefetch on AVX2 | within 1.3% of the GPU result; bit-exact A/B |
| CPU placement | `EXL3_MOE_CPU_SPLIT_MAP` sets the split per layer; a layer stored at two rates puts the smaller group on the CPU | identical output |
| MTP drafting | `-dpm` confidence threshold, sparse draft head (`-sdh`), `EXL3_MOE_CPU_SPLIT_DRAFT`, optional speculative sampling (`-ssp`) | same tokens / same output distribution |
| GPU rate tier | a layer stored at two rates runs on the GPU in one fused decode launch sequence: one run table and rotation, a gate/up launch per rate, one per-token reduction (default when the layer takes no CPU split; `EXL3_MOE_TIER_GPU=1` forces it, `0` disables it) | KLD 0.0007 vs the CPU-split path; 10x closer to an fp32 reference than the dense path |
| Decode MoE kernel | one completion arrival per block for all its rows (was one fence and up to three barriers per row); the down stage always on the wide tile | identical output, 267 MoE kernel tests on sm_86 and sm_120 |
| Decode MoE path | a two-rate layer's gate/up in one launch; the shared expert on a side stream joined before the down stage; router for 2-8 rows uses the single-row FMA GEMV per row (a token routes the same in an MTP verify round as alone) | bit-identical output (router: identical to single-row routing); same kernel test counts on sm_86 and sm_120 |
| Decode attention | split kernel at one pipeline stage for head_dim 256 on CUDA (`EXL3_ATTN_HD256_1STAGE=0` restores two): 1.5-1.8x faster kernel at 32k-96k context | same math; outputs differ only at fp16 rounding (~1e-4), as with any split-count change |
| Build | `setup.py` builds the extension for every GPU in the system (`TORCH_CUDA_ARCH_LIST` overrides) | |
| Misc | loader respects `model.safetensors.index.json`; idle cached memory is freed before CUDA-graph capture (avoids OOM on a full 16 GB card) | |

Run the tests with `pytest tests/kernels/quant tests/kernels/moe tests/moe_cpu`.

## Benchmarks

RTX 5060 Ti 16 GB, Ryzen 5 3600 (6 cores, AVX2), batch size 1, unless the section says two GPUs. Raw output in
[`doc/half-bit-bench/`](doc/half-bit-bench).

**GPU decode MoE kernel**, µs per layer call (Qwen3.6-35B-A3B expert shape, one token, CUDA graphs,
`science/moe_coop_bench.py --graph`):

| bpw | 4.0 | 4.5 | 5.0 | 5.5 | 6.0 | 6.5 | 7.0 | 7.5 | 8.0 |
|---|---|---|---|---|---|---|---|---|---|
| before | 60.1 | 66.5 | 81.2 | 101.3 | 84.5 | 106.2 | 107.8 | 110.8 | 107.3 |
| after | 60.1 | 67.4 | **70.2** | **75.3** | **74.0** | 103.1 | 105.7 | 108.4 | 106.1 |

For integer rates, "before" is upstream's kernel; for half rates it's this fork before the kernel change. 4.5 is
within noise; the gains are at 5-6 bpw.

**CPU expert worker**, µs per call (104 experts in RAM, one token routed to 8 of them, 6 threads, AVX2,
`science/moe_cpu_bench.py --swizzle`):

| bpw | 4 | 4.5 | 5 | 5.5 | 6 | 6.5 | 7 | 7.5 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| packed | 516 | 642 | 553 | 610 | 616 | 660 | 736 | 742 | 717 |
| unpacked | | 1191 | | 1376 | | 1572 | | 1910 | |

5.5 and 6.5 cost about the same as their integer neighbours. 4.5 is the exception: on the CPU it's slower than 5,
so if experts end up on the CPU anyway, store them at 5.

**Conversion:** the half-rate trellis search runs 1.35-2.1x faster than the reference kernel, and a full
Qwen3.6-35B-A3B layer converts in about 200 s instead of 414 s.

**End to end**, same model and settings, only the kernels changed (24.6 GB integer-rate Qwen3.6-35B-A3B, 104 of
256 experts per layer on the CPU, MTP with 4 drafts, 32k context, 8-bit cache):

| | Decode | Prompt | Peak VRAM |
|---|---|---|---|
| upstream kernels | 75.7 tok/s | 368 tok/s | 15.25 GB |
| this fork | **84.5 tok/s** (+11.6%) | 374 tok/s | 15.31 GB |

Our released half-bit model reaches 109.8 tok/s on the same card, but most of that gain comes from tuned MTP
settings.

**Two GPUs** (RTX 3060 12 GB + RTX 5060 Ti 16 GB, Ryzen 5 3600, batch size 1). Our released half-bit model (24.51 GB,
Qwen3.6-35B-A3B) fits whole in the two cards. Decode at 32k context, 8-bit cache; each step adds to the one above:

| Setting | Decode |
|---|---|
| one RTX 5060 Ti, 104 experts per layer on the CPU, MTP 4 drafts | 114.7 tok/s |
| both GPUs, 4 experts per layer on the CPU | 126 tok/s |
| every expert on the GPUs (GPU rate tier) | 151-160 tok/s |
| MTP 6 drafts instead of 4 | 170 tok/s |
| more layers on the faster card, +150 MHz core on the 3060 | 180 tok/s |
| one completion arrival per block in the decode MoE kernel | 186 tok/s |
| both rates in one launch sequence | +1.5% |
| wide tile for the MoE down stage | 190 tok/s |
| both gate/up rates in one launch, exact multi-row router, shared expert on a side stream | 188-194 tok/s (+3-4% at the same draft acceptance) |
| split-decode attention at one pipeline stage (head_dim 256) | same at short context; 138 -> 150 tok/s at 64k context (+9%) |

Clocks: 5060 Ti +350 MHz core / +5200 MHz memory, 3060 +150 / +2000, power capped at 170 W and 140 W. At stock
clocks the same setting decodes 173 tok/s.

At 96k context it decodes 150-180 tok/s and reads a 92k-token prompt at 1,514 tok/s. Peak VRAM at 96k: 10.2 of
12 GiB on the 3060, 15.0 of 15.9 GiB on the 5060 Ti.

Decode on these cards is limited by the trellis arithmetic, not memory bandwidth: a 31% lower core clock made the
MoE kernels 40-51% slower.

`MODEL=/path/to/model bash install.sh` picks this setting on its own. A fresh clone on this machine decodes 187-192
tok/s.

## Limitations

- Half rates need the mul1 codebook.
- Based on exllamav3 v1.6.6; newer upstream versions need the changes rebased.
- Only two GPUs (RTX 5060 Ti, sm_120, and RTX 3060, sm_86, alone and together) and one CPU (Ryzen 5 3600, AVX2)
  have actually been run. Everything else compiles but is untested.
- Linux only, no ROCm build of the half-rate kernels.

Found a problem? Open an issue here with your GPU, driver version and the installer output.

---

Everything below this section is upstream's README.

---


<p align="center">
  <img src="doc/logo.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

[Installation](#installation) · [Supported models](#architecture-support) · [Examples](#examples) · [Quantization](#exl3-quantization) · [Community](#community)

ExLlamaV3 is an inference library for running local LLMs on modern consumer GPUs, with flexible quantization and parallel inference.

- **Quantization** - [EXL3](#exl3-quantization), based on QTIP, plus 2–8 bit cache quantization.
- **Parallel inference** - Flexible tensor-parallel and expert-parallel inference for consumer hardware setups.
- **CPU offloading** - Allows large MoE models to run with limited GPU resources. AVX2 and AVX512 support.  
- **Generation** - Continuous, dynamic batching, speculative decoding, multimodal support.
- **Integrations** - Broad [HF model support](#architecture-support), a [Transformers plugin](examples/transformers_integration.py), and an OpenAI-compatible API via [TabbyAPI](https://github.com/theroyallab/tabbyAPI/).

> [!TIP]
> **Looking for a server?** [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) is the official and recommended backend server. It provides an OpenAI-compatible API for local or remote inference, HF model downloading, embedding model support, and HF Jinja2 chat templates. Its startup script manages and installs prerequisites to help you get started.

<p align="center">
  <img src="doc/qb_kld.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

## Installation

Start by making sure you have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed (CUDA 12.4 or later) since the Torch dependency is not automatically handled by `pip`. Then pick a method below:

### Prebuilt wheel · recommended

Pick a wheel from the [releases page](https://github.com/turboderp-org/exllamav3/releases), then e.g.:

```sh
pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.8.0-cp313-cp313-linux_x86_64.whl
```

### Install from PyPI

```sh
pip install exllamav3
```
Note that the PyPI package does not contain a prebuilt extension and requires the CUDA toolkit and build prerequisites (i.e. VS Build Tools on Windows, gcc on Linux, `python-dev` headers etc.).

### Build from source

<details>
<summary>Source installation with uv or pip</summary>


`exllamav3` declares a minimum `torch` version (>= 2.6.0) and CUDA version (>= 12.4), but beyond that the user is free to select a version of `torch` that is compatible with their environment.

`torch` can be installed in three ways (from least to most effort):
1. **with `uv`, setting only `--extra cuXXX`** installs `torch` automatically with the specified CUDA version, `torch` version is selected by `uv` from compatible versions in the specific index associated with the chosen CUDA version (options 1 and 2)
2. **with `uv`, creating a thin project that depends on `exllamav3[cuXXX]` and pins a specific `torch` version** — like (1) but `torch` is pinned in the thin project's `pyproject.toml`, see [pinning a specific PyTorch version (optional)](#pinning-a-specific-pytorch-version-optional) for details
3. Manually with `uv pip` or `pip` (options 3 and 4)

The flavor extras (`--extra`) are `cu124`, `cu126`, `cu128`, `cu129`, `cu130`, and `cu132` — pick the one matching your installed CUDA build. Both `uv sync` and `pip install .` build the package in an isolated environment where your `torch` is not visible, so they install the extension sources and compile them at first import (JIT, a few minutes once per torch version). For a precompiled install run `pip install --no-build-isolation .` in an environment that already has `torch`, or use the release wheels. Selecting a flavor installs the matching CUDA build of `torch`.

**Option 1 — Working in the cloned repo directly (`uv sync`):**

```sh
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3
# (Optional) switch to dev branch for latest in-progress features
git checkout dev

uv venv
uv sync --extra cu130
# add --extra examples and/or --extra eval for those extra dependencies
```

**Option 2 — Using `exllamav3` as a dependency from another project (`uv add`):**

```sh
# `uv add` works inside an existing project (a directory with a pyproject.toml).
# `uv init` creates one if you're starting a new project, if integrating into
# an existing project skip `uv init`.
uv init my-project
cd my-project

# local checkout
uv add 'path/to/exllamav3[cu130]'               # non-editable
uv add 'path/to/exllamav3[cu130]' --editable    # editable

# straight from GitHub
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]'                 # default branch
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]' --branch dev    # specific branch
```

**Option 3 — Bring your own `torch` and let `uv` pick the backend automatically:**

```sh
uv venv            # or: uv venv --python-preference only-managed
source .venv/bin/activate
uv pip install torch --torch-backend=auto
uv pip install .
```

`--torch-backend=auto` inspects your system and installs the matching PyTorch CUDA build; see [Automatic backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection).

**Option 4 — With `pip`:**

On Windows, you also need the `triton-windows` package (declared as a dependency in `pyproject.toml`); the attention, cache and recurrent kernels are Triton and ExLlamaV3 does not import without it.

```sh
# install a CUDA-enabled torch first so it matches your setup, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install .
```

</details>

<details>
<summary>Pinning a specific PyTorch version (optional)</summary>

#### Pinning a specific PyTorch version (optional)

The flavor extra picks the *index*, but by default torch resolves to the latest version on that
index that satisfies `>=2.6.0`. To pin a specific torch version while developing on `exllamav3`,
create a **"thin" project** that consumes your local checkout as an editable install and declares
the exact `torch` version itself. This keeps the pin out of the `exllamav3` pyproject, so
you can change the torch version freely without touching the repo.

```
my-exllamav3-dev/          # thin project (uv init)
├── pyproject.toml
└── src/                  # package sources (auto-generated)
```

In `pyproject.toml`:

```toml
[project]
name = "my-exllamav3-dev"
version = "0.1.0"
description = "Dev environment for exllamav3"
requires-python = ">=3.10.11"
dependencies = [
    "exllamav3[cu130]",   # select correct CUDA version
    "torch==2.13.0",      # pin the exact torch version you need
]

[tool.uv.sources]
exllamav3 = { path = "../exllamav3", editable = true }
```

Adjust `../exllamav3` to point at your local checkout, then a plain `uv sync` sets up an
environment with the correct PyTorch index (routed via the `cuXXX` extra),
the pinned version of `torch` from that index (as long as it exists), and an editable install of `exllamav3` so code
changes apply immediately. Switch CUDA flavors by changing the extra (`exllamav3[cu124]`,
`exllamav3[cu128]`, …) and/or the torch pin in the thin project.

Or, if you're installing torch manually with `uv pip install torch` (e.g. as in Option 3 above),
specify the version directly, e.g. `uv pip install "torch==2.11.0" --torch-backend=auto`.

</details>

After installing with one of the options above, you should be able to run the conversion, eval and 
example scripts from the main repo directory, e.g., `uv run python convert.py -i ...` or, for manual
installations once the venv is active, `python convert.py -i ...`

**Build environment variables**

- `MAX_JOBS`: by default ninja may launch too many processes and run out of system memory for 
compilation. Set this to a reasonable value like 4 in that case.
- `EXLLAMA_NOCOMPILE`: set to install the library without compiling the C++/CUDA extension. Torch
will build/load it at runtime instead.
- `EXLLAMA_EXT_LINEINFO`, `EXLLAMA_EXT_COMPRESS`: see [doc/env_vars.md](doc/env_vars.md).

## Examples

A number of example scripts are provided to showcase the features of the backend and generator. 
For instance, a versatile CLI chatbot:

<p align="center">
  <img src="doc/chatpy.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

```sh
python examples/chat.py -m <input_dir> -mode <prompt_mode>

# Wealth of options
python examples/chat.py -h
```

## Architecture support

| Model family                                      | HF architecture                                                               | Multimodal | Notes                 |
|---------------------------------------------------|-------------------------------------------------------------------------------|:----------:|-----------------------|
| **AFM**                                           | `ArceeForCausalLM`                                                            |            |                       |
| **AfMoE**                                         | `AfmoeForCausalLM`                                                            |            |                       |
| **Apertus**                                       | `ApertursForCausalLM`                                                         |            |                       |
| **Command-R** etc.                                | `CohereForCausalLM`                                                           |            |                       |
| **Command-A**, **Command-R+** etc.                | `Cohere2ForCausalLM`                                                          |            |                       |
| **DeciLM**, **Nemotron**                          | `DeciLMForCausalLM`                                                           |            |                       |
| **Deepseek V3**                                   | `DeepseekV3ForCausalLM`                                                       |            |                       |
| **Deepseek V4**                                   | `DeepseekV4ForCausalLM`                                                       |     ✓     |                       |
| **dots.llm1**                                     | `Dots1ForCausalLM`                                                            |            |                       |
| **ERNIE 4.5**                                     | `Ernie4_5_ForCausalLM`<br>`Ernie4_5_MoeForCausalLM`                           |            |                       |
| **EXAONE 4.0**                                    | `Exaone4ForCausalLM`                                                          |            |                       |
| **Gemma 2**                                       | `Gemma2ForCausalLM`                                                           |            |                       |
| **Gemma 3**                                       | `Gemma3ForCausalLM`<br>`Gemma3ForConditionalGeneration`                       |     ✓     |                       |
| **Gemma 4**                                       | `Gemma4ForConditionalGeneration`<br>`Gemma4UnifiedForConditionalGeneration`   |     ✓     | E2B/E4B unsupported   |
| **GLM 4**, **GLM 4.6**, etc.                      | `Glm4ForCausalLM`<br>`Glm4MoeForCausalLM`                                     |            |                       |
| **GLM 4.1V**, **GLM 4.5V**                        | `Glm4vForConditionalGeneration`<br>`Glm4vMoeForConditionalGeneration`         |     ✓     |                       |
| **GLM 4.7 Flash**                                 | `Glm4MoeLiteForCausalLM`                                                      |            |                       |
| **GLM 5.2**                                       | `GlmMoeDsaForCausalLM`                                                        |            |                       |
| **GLM 5.3-Flash**                                 | `Glm5NextForConditionalGeneration`                                            |     ✓     |                       |
| **GPT-OSS**                                       | `GptOssForCausalLM`                                                           |            |                       |
| **HyperCLOVAX**                                   | `HyperCLOVAXForCausalLM`<br>`HCXVisionV2ForCausalLM`                          |     ✓     |                       |
| **Hy3**                                           | `HYV3ForCausalLM`                                                             |            |                       |
| **IQuest-Coder**                                  | `IQuestCoderForCausalLM`                                                      |            |                       |
| **Kimi Linear**                                   | `KimiLinearForCausalLM`                                                       |            |                       |
| **Ling 3.x**                                      | `BailingMoeV3ForCausalLM`                                                     |            |                       |
| **Laguna 2.1**                                    | `LagunaForCausalLM`                                                           |            |                       |
| **LFM 2.5**, **LFM2.5-VL**                        | `Lfm2ForCausalLM`<br>`Lfm2MoeForCausalLM`<br>`Lfm2VlForConditionalGeneration` |     ✓     |                       |
| **LFM 3**, **LFM3-VL**                            | `Lfm3ForCausalLM`<br>`Lfm3VlForConditionalGeneration`                          |     ✓     |                       |
| **Llama 1/2/3**,**3.1-Nemotron** etc.             | `LlamaForCausalLM`                                                            |            |                       |
| **MiMo-RL**                                       | `MiMoForCausalLM`                                                             |            |                       |
| **MiMo-V2.6-Flash**                               | `MiMoV2ForCausalLM`                                                           |     ✓     | no audio              |
| **MiniMax-M2**                                    | `MiniMaxM2ForCausalLM`                                                        |            |                       |
| **Mistral**, **Ministral 3**, **Mistral-4** etc.  | `MistralForCausalLM`<br>`Mistral3ForConditionalGeneration`                    |     ✓     |                       |
| **Mixtral**                                       | `MixtralForCausalLM`                                                          |            |                       |
| **NemotronH, Nemotron-3 Nano/Super**              | `NemotronHForCausalLM`                                                        |            |                       |
| **Nemotron 3.5 Bolt**, **Nemotron 3.5 Super VL**  | `NemotronH_Omni_Reasoning_V3`                                                 |     ✓     | Image and video input |
| **Olmo 3.1**                                      | `Olmo3ForCausalLM`                                                            |            |                       |
| **Olmo-Hybrid**                                   | `OlmoHybridForCausalLM`                                                       |            |                       |
| **Phi3**, **Phi4**                                | `Phi3ForCausalLM`                                                             |            |                       |
| **Qwen 2**, **Qwen 2.5**, **Qwen 2.5 VL**         | `Qwen2ForCausalLM`<br>`Qwen2_5_VLForConditionalGeneration`                    |     ✓     |                       |
| **Qwen 3**                                        | `Qwen3ForCausalLM`<br>`Qwen3MoeForCausalLM`                                   |            |                       |
| **Qwen 3-Next**                                   | `Qwen3NextForCausalLM`                                                        |            |                       |
| **Qwen 3-VL**                                     | `Qwen3VLForConditionalGeneration`                                             |     ✓     |                       |
| **Qwen 3-VL MoE**                                 | `Qwen3VLMoeForConditionalGeneration`                                          |     ✓     |                       |
| **Qwen 3.5**                                      | `Qwen3_5ForConditionalGeneration`                                             |     ✓     |                       |
| **Qwen 3.5 MoE**                                  | `Qwen3_5MoeForConditionalGeneration`                                          |     ✓     |                       |
| **Qwen 3.8-Flash-Next**                           | `Qwen4ExpForConditionalGeneration`                                            |     ✓     |                       |
| **Seed-OSS**                                      | `SeedOssForCausalLM`                                                          |            |                       |
| **SmolLM**                                        | `SmolLM3ForCausalLM`                                                          |            |                       |
| **SolarOpen**                                     | `SolarOpenForCausalLM`                                                        |            |                       |
| **Step 3.5 Flash**                                | `Step3p5ForCausalLM`                                                          |            |                       |
| **Step 3.7 Flash**                                | `Step3p7ForConditionalGeneration`                                             |     ✓     |                       |

Always adding more, stay tuned.

## Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume an interrupted quant job
python convert.py -w <working_dir> -r

# More options
python convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors 
until the converted model can be compiled. It should have enough free space to store an entire copy 
of the output model.

See the [conversion guide](doc/convert.md) for more information, or the 
[self-calibration guide](doc/optimize.md). 

## EXL3 quantization

EXL3 quantization is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. It aims to make
SOTA quantization available to users on consumer hardware. The conversion process is designed to be
simple and efficient and requires only an input model (in HF format) and a target bitrate. By
computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a 
model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger
ones (70B+) on a single high-end consumer GPU (see the [conversion guide](doc/convert.md)).

For more information, see the [**QTIP**](https://arxiv.org/abs/2406.11235) and [**QuIP#**](https://arxiv.org/abs/2402.04396) papers, as well as this 
[excellent writeup](https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip) on **QTIP** from together.ai.


## Community

You are always welcome to join the [ExLlama discord server](https://discord.gg/NSFwVuCjRq) ←🎮


### 🤗 Models on Hugging Face

Browse the [EXL3 model collection](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a) for quantized models. Also shout out to the following lovely
people:

- [ArtusDev](https://huggingface.co/ArtusDev)
- [MikeRoz](https://huggingface.co/MikeRoz)
- [MetaphoricalCode](https://huggingface.co/MetaphoricalCode)
- [Ready.Art](https://huggingface.co/ReadyArt)
- [isogen](https://huggingface.co/isogen/models)


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous
supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (chunked linear-attention prefill kernels, vendored under `exllamav3/vendor/fla`)

<p align="center">
  <img src="doc/cat.png" width="40" alt="">
</p>


