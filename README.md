# U-Lite INT8 Underwater Foreground Segmentation (SUIM)

A lightweight underwater-image segmentation experiment using **U-Lite** and the **SUIM** dataset. This repository compares three training routes that produce quantized 8-bit inference models, with a longer-term goal of developing a hardware-efficient accelerator on the **PYNQ-Z2 FPGA**.

> **Project status:** Software training, quantization, and evaluation completed for seed 42. An FPGA accelerator/bitstream has **not** yet been implemented or benchmarked.

## Project overview

- **Input:** 256 × 256 RGB underwater image.
- **Output:** One binary foreground mask (one logit per pixel; foreground after thresholding).
- **Foreground:** SUIM **HD** (human divers), **WR** (wrecks/ruins), **RO** (robots/instruments), and **FV** (fish/vertebrates).
- **Background:** **BW** (waterbody), **PF** (plants/seagrass), **RI** (reefs/invertebrates), and **SR** (seafloor/rocks).
- **Important:** This is **binary foreground segmentation**, not a multiclass fish-only segmentation model. A predicted foreground pixel is not assigned an individual SUIM class.

The original SUIM RGB annotation codes are decoded into a single binary target in `saliency.py`. Images are resized bilinearly and binary masks with nearest-neighbor interpolation. The fixed experiment uses seed **42** and a saved training/validation split.

## Compared training routes

| Route | Training method | INT8 conversion |
| --- | --- | --- |
| `fp32` | FP32 training (TF32 disabled) | Static post-training quantization (PTQ) |
| `amp_fp16` | CUDA automatic mixed precision (mixed FP16/FP32) | Static PTQ using the same calibration scheme |
| `qat` | Quantization-aware training **from scratch** with simulated quantization | Convert the learned fake-quantized graph to quantized operators |

The training routes use the same U-Lite architecture, input size, split, seed, and experimental configuration. They are compared using **converted INT8 validation foreground IoU**; the held-out TEST set is evaluated after the selection is fixed.

### Reported result

The validation-selected **QAT** route achieved **57.30% pooled foreground IoU** on the **110-image held-out SUIM test set** in the recorded experiment. This is pixel-region IoU for the binary task at the configured resolution; it should **not** be presented as the SUIM paper's boundary-F evaluation or as a multiclass fish-segmentation score.

For full per-route metrics, consult the records under:

```text
runs/main/comparisons/seed_42_20261118_105051_903805_IST/
```

In particular, see `selection.json`, `test_results_20261118_105217_361696_IST.csv`, and the `test_*.json` files. These are experiment records, not a claim that FPGA performance has been measured.

## Quantization and hardware handoff

- Quantized reference inference uses **signed INT8 convolution weights** and **unsigned 8-bit activations** (W8A8).
- Quantized convolution and other supported operators use the host **QNNPACK** reference backend.
- GELU and certain upsampling operations are lowered using 8-bit lookup tables for the exported reference model.
- Export records include `model_int8.ts`, `tensors.npz`, `hardware_manifest.json`, and `fx_graph.txt` in each route's `int8_*` directory.

**`model_int8.ts` is a CPU-side TorchScript reference, not an FPGA bitstream or ready-to-load PYNQ overlay.** FPGA work still requires operator mapping, integer arithmetic verification, feature-buffer/DDR planning, and measurement on the PYNQ-Z2 board.

## Repository contents

| File or folder | Purpose |
| --- | --- |
| `model.py` | U-Lite architecture |
| `saliency.py`, `data.py` | RGB-annotation decoding, dataset preparation, and loaders |
| `workflow.py` | Preparation, training, quantization, comparisons, test evaluation, and prediction |
| `quantization.py` | PTQ, QAT, LUT lowering, and export utilities |
| `metrics.py`, `profiling.py` | Segmentation metrics and computational profiling |
| `test_project.py` | Unit and integration checks |
| `experiment.json` | Experiment configuration |
| `run_three_routes.sh` | Script to train and convert all three routes |
| `reference_profile/` | Architecture profiling summaries |
| `runs/main/` | Selected seed-42 evaluation records and INT8 handoff files |
| `START_HERE.md`, `VALIDATION.md` | Additional usage and validation notes |

Training checkpoints (`*.pt`), complete datasets, virtual environments, and other generated bulk artifacts are intentionally excluded from the public repository.

## Dataset

Download SUIM from the [official SUIM dataset page](https://irvlab.cs.umn.edu/resources/suim-dataset). **The dataset is not bundled in this repository.** Arrange it so the configured root contains:

```text
Dataset/
├── train_val/
│   ├── images/
│   └── masks/
└── TEST/
    ├── images/
    └── masks/
```

Set `dataset_root` in `experiment.json` to the correct path on your own machine. In the GitHub-ready copy, `../Dataset` is intended for a dataset folder beside the repository **when commands are run from the repository root**. Adjust it for a different layout. Dataset source and handling notes are in `dataset_provenance.json`.

The original saved dataset manifest is not publicly included because it may contain machine-specific paths. As a result, a new checkout should run `prepare` to generate its own manifest; this **does not automatically recreate the exact saved split** from the original experiment. Retain the original manifest privately if exact split-level reproduction is required.

## Setup

Use **Python 3.11 or 3.12** and an isolated virtual environment. The recorded experiment uses the PyTorch 2.7.1 quantization interface; changing major framework versions may require code modifications.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install a compatible PyTorch 2.7.1 build for your CPU/CUDA setup first.
python -m pip install -r requirements.txt
python workflow.py doctor
python -m unittest -v test_project.py
```

PyTorch installation instructions: [official PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

## Reproduce the workflow

**Training requires the SUIM dataset and appropriate compute resources.** Check `experiment.json` before running commands.

```bash
# Generate a manifest for the local dataset (review the preview before training).
python workflow.py prepare

# Train and convert each route.
python workflow.py train --route fp32 --seed 42 --device cuda
python workflow.py quantize --route fp32 --seed 42

python workflow.py train --route amp_fp16 --seed 42 --device cuda
python workflow.py quantize --route amp_fp16 --seed 42

python workflow.py train --route qat --seed 42 --device cuda
python workflow.py quantize --route qat --seed 42

# Select by converted INT8 validation score.
python workflow.py compare --seed 42

# Only after selection has been frozen:
python workflow.py evaluate-test --seed 42
```

On Linux, after running `prepare` and checking the preview, you can use `bash run_three_routes.sh experiment.json 42` to execute the training and conversion routes. The shell script assumes CUDA.

For inference using a published TorchScript export:

```bash
python workflow.py predict --model /path/to/model_int8.ts --image /path/to/image.jpg --out prediction.png
```

The exported model must be used with a compatible host PyTorch runtime. See `START_HERE.md` and `VALIDATION.md` for additional details, and note that those documents may contain historical local setup examples.

## Limitations and next steps

This repository reports one fixed seed/split experiment. Generalization to other seeds, scenes, or datasets has not been established by these results. CPU inference timings are **not** FPGA performance figures. The next stage is to implement and verify the quantized computation on **PYNQ-Z2**, optimize memory traffic and buffering, and measure board-level throughput, latency, power, and resource utilization.

## References and attribution

- [U-Lite reference implementation](https://github.com/duong-db/U-Lite)
- [SUIM dataset](https://irvlab.cs.umn.edu/resources/suim-dataset)
- [SUIM reference code](https://github.com/xahidbuffon/SUIM)
- [PyTorch documentation](https://pytorch.org/docs/stable/)

Please check the upstream dataset/code terms before redistributing third-party content. No license is asserted for this repository by this README.
