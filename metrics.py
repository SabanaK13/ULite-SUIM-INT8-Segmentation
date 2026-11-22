"""Pixel metrics and host CPU latency; not the paper's boundary F evaluator."""
import time
import numpy as np
import torch


@torch.inference_mode()
def evaluate(model, batches, device='cpu'):
    model.eval()
    tp = fp = fn = tn = 0
    image_ious, image_dices, nonempty_ious = [], [], []
    total_loss = samples = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        z = model(x)
        total_loss += torch.nn.functional.binary_cross_entropy_with_logits(z.float(), y).item() * len(x)
        samples += len(x)
        pred, real = z >= 0, y >= 0.5
        dims = (1, 2, 3)
        batch_tp = (pred & real).sum(dims).cpu().numpy()
        batch_fp = (pred & ~real).sum(dims).cpu().numpy()
        batch_fn = (~pred & real).sum(dims).cpu().numpy()
        batch_tn = (~pred & ~real).sum(dims).cpu().numpy()
        tp += int(batch_tp.sum()); fp += int(batch_fp.sum())
        fn += int(batch_fn.sum()); tn += int(batch_tn.sum())
        for a, b, c in zip(batch_tp, batch_fp, batch_fn):
            iou = float(a/(a+b+c)) if a+b+c else 1.0
            dice = float(2*a/(2*a+b+c)) if 2*a+b+c else 1.0
            image_ious.append(iou); image_dices.append(dice)
            if a+c > 0:
                nonempty_ious.append(iou)
    union = tp+fp+fn
    return {'images': samples, 'bce_loss': total_loss/samples,
            'foreground_iou_global': tp/union if union else None,
            'foreground_dice_global': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'foreground_iou_per_image_mean': float(np.mean(image_ious)),
            'foreground_dice_per_image_mean': float(np.mean(image_dices)),
            'foreground_iou_nonempty_gt_mean': float(np.mean(nonempty_ious)) if nonempty_ious else None,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'threshold_probability': 0.5,
            'metric_note': 'Pixel-region metrics at resized resolution. Per-image empty prediction + empty GT scores 1. Not boundary F or a certified reproduction of Table II.'}


@torch.inference_mode()
def benchmark_cpu(model, example, threads=4, warmup=10, repeats=50):
    if example.shape[0] != 1 or example.device.type != 'cpu':
        raise ValueError('Host latency comparison requires batch-one CPU input.')
    torch.set_num_threads(threads)
    model.eval()
    for _ in range(warmup):
        model(example)
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(example)
        values.append((time.perf_counter()-start)*1000)
    median = float(np.median(values))
    return {'cpu_median_ms': median, 'cpu_p95_ms': float(np.percentile(values, 95)),
            'cpu_model_only_fps_from_median': 1000/median, 'cpu_threads': threads,
            'latency_scope': 'host CPU batch-one model call; excludes image loading/preprocessing/transfers; not FPGA FPS',
            'warmup_calls': warmup, 'timed_calls': repeats}
