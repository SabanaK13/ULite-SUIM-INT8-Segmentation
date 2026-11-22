# Validation performed before delivery

Environment: Linux, Python 3.12, PyTorch 2.7.1+cpu, NumPy 2.2.6, Pillow 11.2.1. No NVIDIA GPU was attached to this cloud workspace. Additional early conversion probes passed on PyTorch 2.15.0+cpu, but the delivered environment is pinned to the tested 2.7.1 interface.

## Passed

- Exact eight-class saliency mapping, including WR=1.0 and intermediate RGB channel values decoded at >127.
- Model forward, QAT backward with finite gradients, PTQ/QAT conversion, graph audit and serialized model reload.
- All 45 convolution modules are quantized. Eleven integer tables replace the GELU/upsampling float islands; five include nearest-neighbor upsampling.
- Every one of the 256 input codes of every table matches the corresponding quantized reference operation. Whole-model outputs match native quantized inference on random, zero and one inputs tested at 64x64.
- The 256x256 parameter/MAC profile is generated. Parameter count is 878,417; convolution MAC count is 736,509,952.
- Synthetic, complete FP32 and QAT training runs: 4 epochs, 64x64, 8 training images, 2 validation images. Both produced timestamped CSVs, parameter inventories, checkpoints, converted models, validation reports, hardware handoff files and CPU timing records.
- FP32 training was interrupted after an epoch, resumed through 8 epochs, and compared with an uninterrupted same-seed CPU run. Final parameter/buffer state matched exactly in that test. Checkpoint saves now use temporary files and atomic replacement.
- Route selection uses converted validation metrics, rejects mismatched manifests, and refuses reselection after TEST is opened. Multi-seed aggregation mechanics were checked with explicitly synthetic metric fixtures; these are not claimed as measured model results.
- Official dataset ZIP directory inspected through HTTP range reads: 1,525 train_val images/masks and 110 TEST images/combined masks. Ten RGB masks were sampled; one contained intermediate RGB values. Two sampled TEST RGB-to-saliency conversions matched the provided binary saliency masks pixel for pixel. See dataset_provenance.json for details.

## Not yet measured or verified

- CUDA AMP training on your RTX 4070. The dedicated CUDA test is included and skipped when CUDA is absent; run it locally.
- Full training, segmentation accuracy, convergence or calibration quality on all of your SUIM data.
- Full-dataset duplicate/label integrity; the local prepare command performs those checks.
- FPGA synthesis, resource utilization, board latency/FPS, board power, DDR traffic, or bit-exact fixed-point HLS/RTL equivalence.
- Compatibility with a future FINN/Vitis/PYNQ toolchain. The exported TorchScript is a CPU reference; the NPZ/manifest are a hardware design handoff.

Synthetic scores and CPU smoke timings are deliberately omitted from the research result tables. They establish execution correctness, not underwater-scene accuracy. The generated comparison and final TEST CSVs must come from your actual local training runs.

Some PyTorch trace warnings report constant quantization scales. These scales are intentionally fixed at inference; the reload and multi-input equivalence tests check the resulting fixed-quantizer artifact.
