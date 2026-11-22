# ULite + SUIM: three routes to an INT8 saliency model

This is a semester-one software experiment for your RTX 4070 (12 GB). It trains the ULite architecture you supplied on the **paper's binary semantic saliency task**, produces three comparable W8A8 inference candidates, records timestamped CSVs, and exports information for a later PYNQ-Z2 hardware implementation.

No trained SUIM weights or claimed SUIM accuracy are included. Training runs on your computer. The delivered tests use synthetic inputs; the archive inspection used selected real masks from your official dataset link.

## 1. What the three routes mean

| Route | Training arithmetic | Conversion |
|---|---|---|
| `fp32` | FP32; TF32 disabled | Static post-training quantization (PTQ) |
| `amp_fp16` | CUDA automatic mixed precision: FP16 for suitable operations, FP32 master weights, reductions and optimizer state | Same static PTQ scheme and calibration images |
| `qat` | Quantization-aware training **from random initialization**; FP32 parameters/gradients, with simulated 8-bit weight/activation rounding | Convert the learned fake-quant graph to actual quantized operators |

Route 2 is **mixed FP16 training**, not pure FP16 parameter updates. Route 3 is **QAT from scratch**, not integer-only backpropagation. These distinctions belong in your report. QAT fine-tuning from a trained FP32 model is another useful experiment, but it is not substituted for your third route here.

The three routes share the architecture, saliency conversion, saved split, seed, optimizer, image size, epoch budget and final quantization scheme. Calibration applies only to PTQ; QAT observes activation ranges during training, then freezes them. Do not assume QAT wins; choose using converted INT8 validation results.

## 2. Exact dataset and saliency target

Your download is the `SUIM.zip` linked by the official dataset page:

- https://irvlab.cs.umn.edu/resources/suim-dataset
- https://drive.google.com/file/d/1uEnlqKrlt6lITc_i80NTtb7iHGcO47sU/view?usp=sharing

The ZIP directory contains 1,525 training/validation image-mask pairs and 110 TEST pairs. The configured root is:

```text
/home/robotics/sabana/Project_v2/dataset/
  train_val/images/
  train_val/masks/
  TEST/images/
  TEST/masks/              <- use the combined RGB masks directly here
    Saliency/             <- supplied binary masks; not the RGB loader's input
    FV/, HD/, ...
```

If extraction adds a `SUIM/` directory, change `dataset_root` in `experiment.json` to that inner directory. No archive extraction, copying, renaming or deletion of your dataset happens automatically.

| Original class | RGB bits | Binary saliency |
|---|---|---|
| BW waterbody | 000 | 0 |
| HD human divers | 001 | 1 |
| PF plants/seagrass | 010 | 0 |
| WR wrecks/ruins | 011 | **1.0** |
| RO robots/instruments | 100 | 1 |
| RI reefs/invertebrates | 101 | 0 |
| FV fish/vertebrates | 110 | 1 |
| SR seafloor/rocks | 111 | 0 |

This follows Section V (PDF page 5), Figure 4 and Figure 6 of the supplied `2004.01241v3.pdf`. The repository's `getSaliency()` uses WR=0.8; we use the paper's WR=1.0. The model produces a single union mask and cannot identify which foreground category a predicted pixel belongs to.

**Correction to the earlier strict-color snippet:** real, original SUIM BMP files can contain intermediate RGB channel values. `saliency.py` follows the authors' `mask/255 > 0.5` threshold, equivalently uint8 channel `>127`, before decoding class bits. It then applies the exact binary table above. Two sampled RGB TEST masks produced precisely the same binary labels as their supplied `TEST/masks/Saliency` files. See `dataset_provenance.json` for the limited inspection scope; this is not a full content audit.

Original images are RGB, resized bilinearly to 256x256, and divided by 255. Decoded binary masks use nearest-neighbor resizing. The square resize can distort aspect ratio; it is an explicit common experimental setting. Horizontal flips are the only training augmentation. Validation, calibration and TEST have no augmentation.

## 3. Install on your local Linux computer

Extract this code into a separate folder such as `/home/robotics/sabana/Project_v2/software`. The ZIP contains a top-level `ulite_int8_project` folder; you may use that folder directly or rename it `software`. Commands below assume you renamed it `software`. Keep `dataset` separate. `START_HERE.md` uses commands from whichever extracted folder you choose.

Use Python 3.11 or 3.12. Create an isolated environment:

```bash
cd /home/robotics/sabana/Project_v2/software
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python workflow.py doctor
python -m unittest -v test_project.py
```

The CUDA 12.6 PyTorch wheel supplies its runtime; your `nvidia-smi` CUDA 13.2 label describes driver support, not a requirement to install a PyTorch CUDA 13.2 wheel. Do not replace your working NVIDIA driver for this project. Confirm `cuda_available: true`, your RTX 4070 name, and the FP16 CUDA test before training.

The CPU test environment used `torch==2.7.1+cpu`, `numpy==2.2.6`, `Pillow==11.2.1` and Python 3.12. This project deliberately pins the legacy PyTorch FX quantization/TorchScript interface because these APIs are being superseded. Keep the isolated environment and `pip freeze` record; do not upgrade it mid-comparison. See PyTorch's official installation/previous-version documentation if installation is unavailable for your Python/OS.

## 4. Check the dataset before training

Review `experiment.json`; then run:

```bash
python workflow.py prepare
```

Preparation matches files by stem, validates label format and image-mask dimensions, checks for exact duplicate image files, and saves a fixed saliency-presence-stratified split. It records SHA-256 hashes so later changes are detected. For the standard archive the split is approximately 80/20 of `train_val`; the exact counts are printed. The same manifest is used by every route and model seed.

Open `runs/main/data_preview.png`. Salient objects must be white in the right-hand panel. Also inspect `runs/main/manifest.json`. The 256-image calibration subset is sampled only from training, never validation or TEST. Exact duplicates trigger an error; resolve/group them without mixing TEST into training. Near-duplicate frames or scene identities are not automatically recognized. If scene grouping information is available, use a group split before claiming generalization across scenes.

Preparation does not overwrite an existing manifest. Keep it throughout the comparison. If the dataset root/layout differs, fix the configuration before preparing it. If you intentionally change image size or training settings for a new experiment, use a new `run_root` and rerun all three routes.

## 5. Train and convert each route

Start with batch size 4 at 256x256. It is a conservative starting point for your 12 GB GPU, not a measured memory guarantee. If a route runs out of memory, reduce the common batch size and start a new experiment for all routes. Do not change only one route's effective batch size.

```bash
python workflow.py train --route fp32 --seed 42 --device cuda
python workflow.py quantize --route fp32 --seed 42

python workflow.py train --route amp_fp16 --seed 42 --device cuda
python workflow.py quantize --route amp_fp16 --seed 42

python workflow.py train --route qat --seed 42 --device cuda
python workflow.py quantize --route qat --seed 42
```

Alternatively, after preparation and preview inspection:

```bash
bash run_three_routes.sh experiment.json 42
```

Defaults: 100 epochs, Adam at 1e-4 with default betas, BCEWithLogitsLoss, gradient-norm clipping at 1.0, and no early stopping. The common epoch budget simplifies comparison. QAT batch-normalization statistics freeze after 50% of training and activation observers after 75%; only checkpoints after observer freezing compete for the best QAT checkpoint. QAT validation runs with observers disabled. FP32/AMP select the best floating validation checkpoint, then PTQ converts it; QAT selects the best frozen fake-quant validation checkpoint, then converts it. The final route comparison always uses actual converted inference.

To resume an interrupted run, retain its configuration and use its `last.pt`:

```bash
python workflow.py train --route fp32 --seed 42 --device cuda --resume /absolute/path/to/last.pt
```

Resume restores model, optimizer, scaler and random-generator state at an epoch boundary and creates a new timestamped CSV segment in the same run. It does not promise bit-identical results across GPU models, library versions or operating systems. `--run /absolute/run/directory` selects an older completed training run for the `quantize` command; otherwise the latest completed run for that route/seed is used.

## 6. Logs and profiling

Every training run has its own folder:

```text
runs/main/fp32/seed_42/20261117_HHMMSS_microseconds_IST/
  run_metadata.json
  training_<timestamp>.csv
  tensors_initial_<timestamp>.csv
  tensors_best_<timestamp>.csv
  layers_<timestamp>.csv
  profile_summary_<timestamp>.csv
  profile_summary.json
  best.pt
  last.pt
  status.json
  int8_<export_timestamp>/
    model_int8.ts
    tensors.npz
    hardware_manifest.json
    fx_graph.txt
    validation.json
    validation_<timestamp>.csv
    exported_tensors_<timestamp>.csv
```

All timestamps use **Asia/Kolkata, UTC+05:30**. `runs/main/training_runs.csv` records starts, completions, failures and resumes. Epoch CSVs include train/validation BCE, global foreground IoU/Dice, per-image IoU, learning rate, training/validation time, training images per second, CUDA peak allocated/reserved memory, process peak RSS, QAT observer status and checkpoint selection. Logs flush after each completed epoch. A failure is recorded in the registry/status even if an epoch is incomplete. Run one training process at a time per experiment root.

Tensor CSVs enumerate every named parameter and buffer, including shape, count, dtype, bytes, trainability, minimum, maximum, mean, population standard deviation, zero fraction and finite-value check. They do not dump millions of individual scalar weights into CSV; complete weights are in checkpoints/NPZ.

For your architecture at batch 1 and 256x256:

- Parameters: **878,417**.
- Convolution MACs: **736,509,952**.
- Convolution FLOPs using **1 MAC = 2 FLOPs**: **1,473,019,904**.
- Additional bias, residual-add, BN-affine and GELU arithmetic are reported separately; the resulting estimate is 1,519,165,440 arithmetic operations, **excluding the cost of erf and memory operations**.
- GELU erf evaluations, pooling comparisons and upsampling copies have separate columns. Backpropagation, optimizer operations, concatenation traffic and quantization overhead are not included in convolution FLOPs.
- The five retained skip tensors contain **2,031,616 elements** in total; at 8 bits that is about 1.94 MiB. The first skip alone is 1 MiB. This is a specific buffer-size warning for next semester, not a measured peak-memory or DDR-bandwidth result.

`reference_profile/` contains the architecture counts produced here. Layer counts refer to the original inference graph; native quantization can fuse operations. Compute counts are the same across routes at a fixed shape; training precision changes arithmetic implementation, not nominal network MACs.

## 7. Compare and select without using TEST

```bash
python workflow.py compare --seed 42
```

This writes a timestamped comparison CSV and frozen `selection.json`. The winner has the highest converted INT8 validation foreground IoU, with Dice breaking exact ties. Serialized size, raw weight bytes and host CPU median/p95 latency are reported, but CPU timing is not used as a proxy for FPGA throughput.

All candidates have the same architecture/bit widths. Their nominal FPGA compute and storage needs are therefore similar; the main training-route comparison is accuracy, quantization sensitivity and training cost. A lower-accuracy model cannot be called hardware-superior merely because one noisy CPU timing is shorter.

Repeat seeds 43 and 44 with the **same saved split** for a stronger report:

```bash
bash run_three_routes.sh experiment.json 43
bash run_three_routes.sh experiment.json 44
python workflow.py aggregate --seeds 42 43 44
```

The aggregate reports model-seed variability conditional on one split, not uncertainty over all possible data splits. Decide the training route and selection rule before opening TEST. Then, for each frozen seed comparison:

```bash
python workflow.py evaluate-test --seed 42
```

The test command reports all three already-frozen candidates and marks the validation-selected winner. It does not reselect on TEST. Interrupted test evaluation can resume; existing per-route test outputs are reused. The script refuses to replace a seed's selection after TEST has been opened. Do not use test results to tune hyperparameters, thresholds, calibration data or choose another seed.

Our metrics are pixel-region IoU and Dice at the resized resolution. The public SUIM evaluator uses a boundary F routine and different aggregation choices; do not label our ordinary Dice as an exact reproduction of Table II. The task grouping is reproduced; the complete paper experiment is not.

## 8. What is actually INT8, and what goes to hardware later

The final CPU model uses **signed INT8 convolution weights and unsigned 8-bit activations** with affine zero points (W8A8). QNNPACK provides the reference quantized convolutions, residual additions, BN and pooling. All 45 convolutions are quantized. Native FX leaves GELU and some upsampling in floating point; this project lowers those islands to eleven 256-byte lookup tables, five also performing nearest-neighbor copies. The tables retain the original GELU behavior under the final quantizers. Exhaustive table-code checks and native-versus-lowered inference checks are included. An audit rejects unexpected floating **activation tensors** between the input quantizer and final output dequantizer.

This does **not** assert that every backend instruction or stored constant is integer: native quantized BN state and bias metadata remain FP32. The handoff contains signed weight bytes, qparams, BN parameters, LUT bytes, shapes, groups/dilation/padding, candidate INT32 biases, real requantization multipliers and a sample input/output vector. FP32-bias and integer-bias rounding can differ; the candidate integer biases are not certified bit-exact to every CPU kernel.

`model_int8.ts` is an executable **host CPU reference**, not an FPGA bitstream, PYNQ overlay, CUDA INT8 executable, or guaranteed FINN/Vitis input. Next semester requires choosing an implementation flow, mapping the full operator set, implementing/verifying integer BN and requantization, buffering skip features, handling DDR transfers and measuring actual board performance. Running a model on PYNQ's ARM CPU alone is not FPGA acceleration. FINN/Brevitas or custom HLS/RTL may require model/export changes; no automatic compatibility is promised.

For a fixed 0.5 sigmoid threshold, a dequantized logit >=0 is foreground. In an affine quantized output this becomes code >= output zero point, which can avoid a sigmoid unit in hardware.

To test one exported CPU model on a local image:

```bash
python workflow.py predict --model /path/to/model_int8.ts --image /path/to/image.jpg --out prediction.png
```

Keep `hardware_manifest.json` next to the model. The prediction is at the configured resized resolution.

## 9. Files and validation status

- `model.py`: your ULite architecture; explicit equivalent odd-kernel padding, explicit nearest upsampling, and the ignored `mixer_kernel` argument corrected. GELU/channel widths/residual structure are retained.
- `saliency.py`, `data.py`: paper target, original mask threshold decoding, fixed manifest and loaders.
- `workflow.py`: training/resume, conversion, selection, final evaluation and inference.
- `quantization.py`: PTQ/QAT preparation, integer LUT lowering, audit and handoff.
- `profiling.py`, `metrics.py`: CSV inventories, arithmetic conventions and evaluation.
- `test_project.py`: label, parameter-count, QAT-gradient, quantization-equivalence/export tests and a CUDA AMP test.
- `VALIDATION.md`: actual checks performed in this cloud workspace, with limitations.

First run `doctor`, the unit tests and `prepare`, then share those outputs and the data preview if anything looks wrong. No need to paste every training epoch: share the run's metadata, profile summary, comparison CSV and any error traceback.

Sources: [ULite implementation](https://github.com/duong-db/U-Lite/blob/main/models/ULite.py), [SUIM mask utilities](https://github.com/xahidbuffon/SUIM/blob/master/utils/data_utils.py), [SUIM evaluator](https://github.com/xahidbuffon/SUIM/blob/master/get_f1_iou.py), [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html), [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/), [FINN documentation](https://finn.readthedocs.io/en/latest/).
