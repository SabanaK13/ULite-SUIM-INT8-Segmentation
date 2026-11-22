"""Three-route SUIM saliency experiment. See README.md before running."""
import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
import platform
import random
import resource
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image
from torch.ao.quantization import disable_observer, enable_observer
from torch.ao.nn.intrinsic.qat import freeze_bn_stats

from data import loader, prepare_data, read_manifest, file_hash, SUIM
from metrics import evaluate, benchmark_cpu
from model import ULite
from profiling import arithmetic_profile, tensor_inventory, write_csv, exported_tensor_inventory
from quantization import prepare, convert, export_handoff, set_engine, ENGINE

ROUTES = ['fp32', 'amp_fp16', 'qat']
TIMEZONE = ZoneInfo('Asia/Kolkata')


def stamp():
    return datetime.now(TIMEZONE).strftime('%Y%m%d_%H%M%S_%f_IST')


def now():
    return datetime.now(TIMEZONE).isoformat(timespec='seconds')


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def config_from(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    cfg['dataset_root'] = str(Path(cfg['dataset_root']).expanduser().resolve())
    cfg['run_root'] = str((path.parent / cfg['run_root']).resolve())
    if cfg['image_size'] < 64 or cfg['image_size'] % 32:
        raise ValueError('image_size must be >=64 and divisible by 32.')
    for key in ['batch_size', 'epochs', 'calibration_images', 'cpu_threads', 'benchmark_repeats']:
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive.')
    if not 0 < cfg['validation_fraction'] < 1 or cfg['workers'] < 0 or cfg['learning_rate'] <= 0:
        raise ValueError('Invalid data/training configuration.')
    if not 0 < cfg['qat_freeze_bn_fraction'] < cfg['qat_freeze_observers_fraction'] < 1:
        raise ValueError('QAT freeze fractions must satisfy 0 < BN < observers < 1.')
    return cfg


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def environment():
    return {'python': sys.version, 'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'platform': platform.platform(), 'processor': platform.processor(),
            'cuda_available': torch.cuda.is_available(), 'torch_cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'timezone': 'Asia/Kolkata', 'quantized_engine': ENGINE}


def registry(cfg, event, run, route, seed, detail=''):
    path = Path(cfg['run_root']) / 'training_runs.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['timestamp_ist', 'event', 'route', 'seed', 'run_directory', 'detail'])
        if not exists:
            writer.writeheader()
        writer.writerow({'timestamp_ist': now(), 'event': event, 'route': route,
                         'seed': seed, 'run_directory': str(run), 'detail': detail})


def route_root(cfg, route, seed):
    return Path(cfg['run_root']) / route / f'seed_{seed}'


def resolve_run(cfg, args):
    if args.run:
        run = Path(args.run).resolve()
    else:
        pointer = route_root(cfg, args.route, args.seed) / 'latest_completed.json'
        if not pointer.exists():
            raise ValueError(f'No completed {args.route} training run for seed {args.seed}.')
        run = Path(json.loads(pointer.read_text())['run_directory'])
    meta = json.loads((run / 'run_metadata.json').read_text())
    if meta['route'] != args.route or meta['seed'] != args.seed:
        raise ValueError('Requested route/seed does not match this run.')
    return run, meta


def freeze_qat(model, epoch, cfg):
    model.train()
    bn_epoch = max(2, math.floor(cfg['epochs']*cfg['qat_freeze_bn_fraction'])+1)
    obs_epoch = max(3, math.floor(cfg['epochs']*cfg['qat_freeze_observers_fraction'])+1)
    if epoch >= bn_epoch:
        model.apply(freeze_bn_stats)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
    model.apply(disable_observer if epoch >= obs_epoch else enable_observer)
    return epoch >= obs_epoch


def save_rng(batches):
    numpy_state = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate(),
            'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'loader': batches.generator.get_state()}


def restore_rng(state, batches):
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    batches.generator.set_state(state['loader'])


def train(cfg, args):
    if args.route == 'amp_fp16' and not args.device.startswith('cuda'):
        raise ValueError('The AMP FP16 route requires CUDA. CPU is permitted only for FP32/QAT smoke tests.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Run doctor and install the CUDA-enabled PyTorch wheel.')
    if args.route == 'qat' and cfg['epochs'] < 4:
        raise ValueError('QAT needs at least 4 epochs for observer/BN-freezing stages; use 100 for the real experiment.')
    manifest, manifest_hash = read_manifest(cfg)
    torch.set_num_threads(cfg['cpu_threads'])
    seed_all(args.seed)
    run_id = stamp()
    resumed = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    run = Path(args.resume).resolve().parent if resumed else route_root(cfg, args.route, args.seed) / run_id
    run.mkdir(parents=True, exist_ok=True)
    if resumed:
        if resumed['route'] != args.route or resumed['seed'] != args.seed or resumed['manifest_hash'] != manifest_hash or resumed['config'] != cfg:
            raise ValueError('Resume requires the same route, seed, config and unchanged dataset manifest.')
        if resumed['epoch'] >= cfg['epochs']:
            raise ValueError('This run is already complete.')
    base = ULite()
    if not resumed:
        meta = {'run_id': run_id, 'started_at_ist': now(), 'route': args.route,
                'seed': args.seed, 'config': cfg, 'manifest_hash': manifest_hash,
                'environment': environment(),
                'training_precision': {'fp32': 'FP32; TF32 disabled',
                                       'amp_fp16': 'CUDA AMP FP16 compute with FP32 master weights/optimizer',
                                       'qat': 'QAT from random initialization; FP32 training with W8A8 fake quantization'}[args.route],
                'saliency_mapping': manifest['saliency_mapping']}
        save_json(run / 'run_metadata.json', meta)
    else:
        meta = json.loads((run / 'run_metadata.json').read_text())
    registry(cfg, 'resumed' if resumed else 'started', run, args.route, args.seed)
    try:
        if not resumed:
            tensor_inventory(base, run, run_id, 'initial')
            arithmetic_profile(copy.deepcopy(base), cfg['image_size'], run, run_id)
        model = prepare(base, cfg['image_size'], qat=True) if args.route == 'qat' else base
        if resumed:
            model.load_state_dict(resumed['model'])
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        amp = args.route == 'amp_fp16'
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        seed_all(args.seed)  # Reset after FX/profiling; same data RNG in all routes.
        training = loader(manifest['train'], cfg, args.seed, train=True)
        validation = loader(manifest['validation'], cfg, args.seed)
        start_epoch, best = 1, -1.0
        patience = int(cfg.get('early_stopping_patience', 20))
        min_delta = float(cfg.get('early_stopping_min_delta', 0.001))
        if patience < 1 or min_delta < 0:
            raise ValueError('Invalid early-stopping configuration.')
        early_stop_best = -1.0
        early_stop_bad_epochs = 0
        stopped_early = False
        if resumed:
            optimizer.load_state_dict(resumed['optimizer'])
            scaler.load_state_dict(resumed['scaler'])
            restore_rng(resumed['rng'], training)
            start_epoch, best = resumed['epoch']+1, resumed['best_score']
            early_stop_best = resumed.get('early_stop_best', -1.0)
            early_stop_bad_epochs = resumed.get('early_stop_bad_epochs', 0)
            if early_stop_bad_epochs >= patience:
                raise ValueError(
                    'This checkpoint has already reached early stopping. '
                    'Use its best.pt for quantization.'
                )
            if (run / 'best.pt').exists():
                best = max(best, torch.load(run / 'best.pt', map_location='cpu', weights_only=True)['best_score'])
        log_path = run / f'training_{run_id}.csv'
        fields = ['timestamp_ist', 'route', 'seed', 'epoch', 'learning_rate', 'train_bce', 'val_bce',
                  'val_iou_global', 'val_dice_global', 'val_iou_per_image_mean',
                  'train_seconds', 'validation_seconds', 'train_images_per_second',
                  'gpu_peak_allocated_mib', 'gpu_peak_reserved_mib', 'process_peak_rss_mib',
                  'qat_observers_frozen', 'selected_as_best',
                  'early_stop_monitoring', 'early_stop_bad_epochs',
                  'early_stop_patience', 'early_stop_min_delta',
                  'early_stop_triggered']
        with log_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); handle.flush()
            for epoch in range(start_epoch, cfg['epochs']+1):
                model.train()
                frozen = freeze_qat(model, epoch, cfg) if args.route == 'qat' else False
                if args.device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                begin, total, seen = time.perf_counter(), 0.0, 0
                for x, y in training:
                    x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda' if amp else 'cpu', dtype=torch.float16 if amp else torch.bfloat16, enabled=amp):
                        logits = model(x)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss; inspect data, qparams and learning rate.')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer); scaler.update()
                    total += loss.item()*len(x); seen += len(x)
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                train_seconds = time.perf_counter()-begin
                if args.route == 'qat':
                    model.apply(disable_observer)  # Validation never calibrates observers.
                begin = time.perf_counter()
                metrics = evaluate(model, validation, args.device)
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                validation_seconds = time.perf_counter()-begin
                score = metrics['foreground_iou_global']
                if score is None:
                    raise ValueError('Undefined validation saliency IoU; inspect resized ground truths.')
                eligible = args.route != 'qat' or frozen

                # Always retain the highest eligible validation IoU.
                improved = eligible and score > best
                if improved:
                    best = score

                # Patience uses its own reference so several small gains
                # can cumulatively exceed min_delta.
                if eligible:
                    if (
                        early_stop_best < 0
                        or score > early_stop_best + min_delta
                    ):
                        early_stop_best = score
                        early_stop_bad_epochs = 0
                    else:
                        early_stop_bad_epochs += 1

                stopped_early = (
                    eligible
                    and early_stop_bad_epochs >= patience
                )
                checkpoint = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                              'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                              'rng': save_rng(training), 'epoch': epoch, 'best_score': best,
                              'validation': metrics, 'route': args.route, 'seed': args.seed,
                              'config': cfg, 'manifest_hash': manifest_hash,
                              'early_stop_best': early_stop_best,
                              'early_stop_bad_epochs': early_stop_bad_epochs,
                              'early_stop_triggered': stopped_early}
                if improved:
                    save_checkpoint(run / 'best.pt', checkpoint)
                save_checkpoint(run / 'last.pt', checkpoint)
                row = {'timestamp_ist': now(), 'route': args.route, 'seed': args.seed, 'epoch': epoch,
                       'learning_rate': optimizer.param_groups[0]['lr'], 'train_bce': total/seen,
                       'val_bce': metrics['bce_loss'], 'val_iou_global': score,
                       'val_dice_global': metrics['foreground_dice_global'],
                       'val_iou_per_image_mean': metrics['foreground_iou_per_image_mean'],
                       'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                       'train_images_per_second': seen/train_seconds,
                       'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else 0,
                       'gpu_peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20 if args.device.startswith('cuda') else 0,
                       'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       'qat_observers_frozen': frozen,
                       'selected_as_best': improved,
                       'early_stop_monitoring': eligible,
                       'early_stop_bad_epochs': early_stop_bad_epochs,
                       'early_stop_patience': patience,
                       'early_stop_min_delta': min_delta,
                       'early_stop_triggered': stopped_early}
                writer.writerow(row); handle.flush()
                print(json.dumps(row), flush=True)
                if stopped_early:
                    print(
                        f'Early stopping at epoch {epoch}: '
                        f'no validation IoU improvement greater than '
                        f'{min_delta} for {patience} eligible epochs. '
                        f'Using best.pt with IoU={best:.6f}.',
                        flush=True,
                    )
                    break
        best_checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        model.cpu().load_state_dict(best_checkpoint['model'])
        tensor_inventory(model.cpu(), run, run_id, 'best')
        save_json(run / 'status.json', {
            'status': 'completed',
            'finished_at_ist': now(),
            'best_score': best,
            'stopped_early': stopped_early,
            'last_epoch': epoch,
            'best_epoch': best_checkpoint['epoch'],
            'early_stop_bad_epochs': early_stop_bad_epochs,
            'early_stop_patience': patience,
            'early_stop_min_delta': min_delta,
        })
        save_json(route_root(cfg, args.route, args.seed) / 'latest_completed.json', {'run_directory': str(run)})
        registry(cfg, 'completed', run, args.route, args.seed, f'best validation IoU={best}')
        print(f'Completed training: {run}', flush=True)
    except BaseException as error:
        save_json(run / 'status.json', {'status': 'interrupted_or_failed', 'timestamp_ist': now(), 'error': str(error)})
        registry(cfg, 'interrupted_or_failed', run, args.route, args.seed, str(error))
        raise


def quantize(cfg, args):
    manifest, manifest_hash = read_manifest(cfg)
    run, meta = resolve_run(cfg, args)
    if meta['manifest_hash'] != manifest_hash or meta['config'] != cfg:
        raise ValueError('Configuration or manifest changed since training.')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    torch.set_num_threads(cfg['cpu_threads'])
    if args.route == 'qat':
        prepared = prepare(ULite(), cfg['image_size'], qat=True)
        prepared.load_state_dict(checkpoint['model'])
        prepared.eval().apply(disable_observer)
    else:
        model = ULite().eval()
        model.load_state_dict(checkpoint['model'])
        prepared = prepare(model, cfg['image_size'])
        with torch.inference_mode():
            for x, _ in loader(manifest['calibration'], cfg, args.seed, workers=0):
                prepared(x)
    example = SUIM(manifest['calibration'][:1], cfg['image_size'])[0][0][None]
    model_int8, replacements = convert(prepared, example)
    export_id = stamp()
    output = run / ('int8_'+export_id)
    output.mkdir()
    handoff = export_handoff(model_int8, example, output, replacements)
    exported_tensor_inventory(output / 'tensors.npz', output, export_id)
    validation = evaluate(model_int8, loader(manifest['validation'], cfg, args.seed, workers=0))
    # Benchmark the reloaded serialized artifact, as later consumers will run it.
    serialized = torch.jit.load(str(output / 'model_int8.ts')).eval()
    timing = benchmark_cpu(serialized, example, cfg['cpu_threads'], cfg['benchmark_warmup'], cfg['benchmark_repeats'])
    report = {'timestamp_ist': now(), 'route': args.route, 'seed': args.seed,
              'run_directory': str(run), 'export_directory': str(output),
              'manifest_hash': manifest_hash, 'config': cfg, 'environment': environment(),
              'parent_validation_iou': checkpoint['validation']['foreground_iou_global'],
              'parent_metric_kind': 'frozen fake-quant QAT' if args.route == 'qat' else 'FP32 inference of trained master weights',
              'validation': validation, 'latency': timing,
              'quantized_model_file_bytes': (output / 'model_int8.ts').stat().st_size,
              'quantized_model_sha256': file_hash(output / 'model_int8.ts'),
              'raw_int8_weight_bytes': handoff['raw_weight_bytes'],
              'lookup_table_bytes': handoff['lookup_table_bytes'],
              'unexpected_float_activation_nodes': handoff['unexpected_float_activation_nodes']}
    save_json(output / 'validation.json', report)
    write_csv(output / f'validation_{export_id}.csv', [{
        'timestamp_ist': report['timestamp_ist'], 'route': args.route, 'seed': args.seed,
        **validation, **timing, 'model_file_bytes': report['quantized_model_file_bytes'],
        'raw_weight_bytes': handoff['raw_weight_bytes']}])
    save_json(run / 'latest_int8.json', {'export_directory': str(output)})
    print(f'Exported verified W8A8 CPU reference and hardware handoff: {output}', flush=True)


def candidate_reports(cfg, seed):
    reports = []
    for route in ROUTES:
        args = argparse.Namespace(route=route, seed=seed, run=None)
        run, _ = resolve_run(cfg, args)
        pointer = json.loads((run / 'latest_int8.json').read_text())
        reports.append(json.loads((Path(pointer['export_directory']) / 'validation.json').read_text()))
    return reports


def compare(cfg, args):
    _, manifest_hash = read_manifest(cfg)
    previous = Path(cfg['run_root']) / f'selection_seed_{args.seed}.json'
    if previous.exists():
        previous_selection = json.loads(Path(json.loads(previous.read_text())['selection_file']).read_text())
        if previous_selection['test_evaluation_started']:
            raise ValueError('This seed has already used TEST. Keep its frozen selection; do not reselect using test feedback.')
    reports = candidate_reports(cfg, args.seed)
    if any(r['manifest_hash'] != manifest_hash or r['config'] != cfg for r in reports):
        raise ValueError('Candidates must share the same unchanged dataset manifest and configuration.')
    if any(r['unexpected_float_activation_nodes'] for r in reports):
        raise ValueError('A candidate contains unexpected float activation nodes.')
    cpus = [r['environment']['platform']+r['environment']['processor'] for r in reports]
    if len(set(cpus)) != 1:
        raise ValueError('CPU benchmarks were produced on different platforms; rerun them on one host.')
    rows = []
    for r in reports:
        rows.append({'timestamp_ist': r['timestamp_ist'], 'route': r['route'], 'seed': args.seed,
                     'validation_iou': r['validation']['foreground_iou_global'],
                     'validation_dice': r['validation']['foreground_dice_global'],
                     'parent_validation_iou': r['parent_validation_iou'],
                     'iou_drop_from_route_parent': r['parent_validation_iou']-r['validation']['foreground_iou_global'],
                     'model_file_bytes': r['quantized_model_file_bytes'],
                     'raw_int8_weight_bytes': r['raw_int8_weight_bytes'], **r['latency'],
                     'export_directory': r['export_directory']})
    winner = max(reports, key=lambda r: (r['validation']['foreground_iou_global'], r['validation']['foreground_dice_global']))
    comparison_id = stamp()
    output = Path(cfg['run_root']) / 'comparisons' / f'seed_{args.seed}_{comparison_id}'
    output.mkdir(parents=True)
    write_csv(output / f'comparison_{comparison_id}.csv', rows)
    selection = {'timestamp_ist': now(), 'seed': args.seed, 'manifest_hash': manifest_hash,
                 'config': cfg, 'selected_route': winner['route'],
                 'selected_export': winner['export_directory'], 'candidates': reports,
                 'rule': 'Highest converted INT8 validation foreground IoU; Dice breaks exact ties. TEST is not consulted. Same architecture/bit-width in all routes; host CPU latency is descriptive, not a prediction of FPGA speed.',
                 'test_evaluation_started': False}
    save_json(output / 'selection.json', selection)
    save_json(Path(cfg['run_root']) / f'selection_seed_{args.seed}.json', {'selection_file': str(output / 'selection.json')})
    print(json.dumps({'selected_route': winner['route'], 'validation_iou': winner['validation']['foreground_iou_global'], 'comparison_directory': str(output)}, indent=2))


def evaluate_test(cfg, args):
    manifest, manifest_hash = read_manifest(cfg)
    pointer = json.loads((Path(cfg['run_root']) / f'selection_seed_{args.seed}.json').read_text())
    selection_path = Path(pointer['selection_file'])
    selection = json.loads(selection_path.read_text())
    if selection['manifest_hash'] != manifest_hash or selection['config'] != cfg:
        raise ValueError('Dataset/configuration changed after model selection.')
    for candidate in selection['candidates']:
        artifact = Path(candidate['export_directory']) / 'model_int8.ts'
        if file_hash(artifact) != candidate['quantized_model_sha256']:
            raise ValueError(f'Frozen candidate artifact changed: {artifact}')
    selection['test_evaluation_started'] = True
    selection['test_opened_at_ist'] = now()
    save_json(selection_path, selection)
    set_engine()
    torch.set_num_threads(cfg['cpu_threads'])
    output = selection_path.parent
    rows = []
    for candidate in selection['candidates']:
        existing = output / f'test_{candidate["route"]}.json'
        if existing.exists():
            rows.append(json.loads(existing.read_text()))
            continue  # Resume interrupted evaluation without changing frozen candidates.
        model = torch.jit.load(str(Path(candidate['export_directory']) / 'model_int8.ts')).eval()
        metrics = evaluate(model, loader(manifest['test'], cfg, args.seed, workers=0))
        row = {'timestamp_ist': now(), 'route': candidate['route'], 'seed': args.seed,
               'selected_on_validation': candidate['route'] == selection['selected_route'], **metrics}
        rows.append(row)
        save_json(output / f'test_{candidate["route"]}.json', row)
    write_csv(output / f'test_results_{stamp()}.csv', rows)
    print(json.dumps(rows, indent=2))


def aggregate(cfg, args):
    _, manifest_hash = read_manifest(cfg)
    seeds = args.seeds or [42, 43, 44]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Specify distinct seeds.')
    reports = [r for seed in seeds for r in candidate_reports(cfg, seed)]
    if any(r['manifest_hash'] != manifest_hash or r['config'] != cfg for r in reports):
        raise ValueError('All seeds/routes must share the experiment configuration and manifest.')
    rows = []
    for route in ROUTES:
        group = [r for r in reports if r['route'] == route]
        ious = np.array([r['validation']['foreground_iou_global'] for r in group])
        dices = np.array([r['validation']['foreground_dice_global'] for r in group])
        rows.append({'timestamp_ist': now(), 'route': route, 'seeds': str(seeds),
                     'n_seeds': len(group), 'validation_iou_mean': float(ious.mean()),
                     'validation_iou_std_sample': float(ious.std(ddof=1)) if len(group)>1 else None,
                     'validation_dice_mean': float(dices.mean()),
                     'validation_dice_std_sample': float(dices.std(ddof=1)) if len(group)>1 else None,
                     'note': 'Model-seed variability on one fixed split; TEST not consulted.'})
    output = Path(cfg['run_root']) / 'comparisons'
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'aggregate_{stamp()}.csv'
    write_csv(path, rows)
    print(json.dumps({'best_route_by_mean_validation_iou': max(rows,key=lambda r:r['validation_iou_mean'])['route'], 'csv': str(path)},indent=2))


def predict(cfg, args):
    if not args.model or not args.image or not args.out:
        raise ValueError('predict requires --model, --image, and --out.')
    set_engine()
    model = torch.jit.load(args.model).eval()
    manifest_path = Path(args.model).parent / 'hardware_manifest.json'
    handoff = json.loads(manifest_path.read_text())
    size = next(r['output']['shape'][-1] for r in handoff['graph'] if r['kind'] == 'placeholder')
    with Image.open(args.image) as original:
        image = original.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()/255)[None]
    with torch.inference_mode():
        mask = (model(x)[0, 0] >= 0).numpy().astype(np.uint8)*255
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.out)
    print(f'Saved {size}x{size} saliency mask: {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['doctor', 'prepare', 'train', 'quantize', 'compare', 'aggregate', 'evaluate-test', 'predict'])
    parser.add_argument('--config', default=str(Path(__file__).with_name('experiment.json')))
    parser.add_argument('--route', choices=ROUTES, default='fp32')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', help='Model seeds to aggregate; defaults to 42 43 44')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--run', help='Use this exact completed training run for quantize')
    parser.add_argument('--resume', help='Resume train from its last.pt; appends a separate timestamped CSV segment')
    parser.add_argument('--model'); parser.add_argument('--image'); parser.add_argument('--out')
    args = parser.parse_args()
    cfg = config_from(args.config)
    if args.command == 'doctor':
        print(json.dumps(environment(), indent=2)); set_engine()
        if torch.cuda.is_available():
            x = torch.rand(2, 3, 64, 64, device='cuda')
            model = ULite().cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                z = model(x)
            print('CUDA FP16 forward passed; output dtype:', z.dtype)
    elif args.command == 'prepare':
        prepare_data(cfg, cfg['run_root'])
    else:
        {'train': train, 'quantize': quantize, 'compare': compare, 'aggregate': aggregate,
         'evaluate-test': evaluate_test, 'predict': predict}[args.command](cfg, args)


if __name__ == '__main__':
    main()
