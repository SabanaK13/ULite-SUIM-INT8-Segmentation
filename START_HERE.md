# Start here on your RTX 4070 computer

1. Extract this ZIP. Open a terminal **inside the extracted `ulite_int8_project` folder**. It can stay beside your `dataset` directory; do not put code inside the image/mask folders.

2. Install the isolated software environment (Python 3.11 or 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python workflow.py doctor
python -m unittest -v test_project.py
```

3. Confirm the root in `experiment.json` is the folder that directly contains `train_val` and `TEST`. It currently points to `/home/robotics/sabana/Project_v2/dataset`. Then:

```bash
python workflow.py prepare
```

Open `runs/main/data_preview.png` and check the image/white-saliency alignment. Keep the saved manifest unchanged across all runs. Share the doctor/test/prepare output first if a check fails. Full training has **not** been run for you in the cloud.

4. Once these checks pass, run the first route:

```bash
python workflow.py train --route fp32 --seed 42 --device cuda
python workflow.py quantize --route fp32 --seed 42
```

Then follow README sections 5–7 for AMP FP16, QAT, comparisons, repeated seeds and final held-out evaluation. Use `run_three_routes.sh` only when you are ready to run all three complete training jobs. Every run gets its own timestamped directory and CSV files.

The foreground is exactly **HD + RO + FV + WR**; background is **BW + PF + RI + SR**. Original SUIM mask channels are thresholded at >127 before grouping. The updated loader handles intermediate values found in the official original BMP masks. Wrecks receive **1.0**, as specified in the paper.

The eventual output is three W8A8 CPU reference models plus hardware handoff files. FPGA implementation and board measurements are next-semester work; neither a CUDA timing nor a CPU timing is a PYNQ-Z2 FPGA FPS result.
