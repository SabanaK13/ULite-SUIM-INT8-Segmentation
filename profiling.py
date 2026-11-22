"""Transparent, batch-one arithmetic counts and complete tensor inventories."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def tensor_inventory(model, directory, timestamp, stage):
    rows = []
    for kind, tensors in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]:
        for name, value in tensors:
            v = value.detach().cpu()
            if v.is_quantized:
                v = v.dequantize()
            f = v.double()
            rows.append({'timestamp_ist': timestamp, 'stage': stage, 'kind': kind,
                         'name': name, 'shape': str(list(value.shape)), 'dtype': str(value.dtype),
                         'elements': value.numel(), 'storage_bytes': value.numel()*value.element_size(),
                         'requires_grad': bool(value.requires_grad),
                         'min': f.min().item() if f.numel() else None,
                         'max': f.max().item() if f.numel() else None,
                         'mean': f.mean().item() if f.numel() else None,
                         'std_population': f.std(unbiased=False).item() if f.numel() else None,
                         'zero_fraction': (f == 0).double().mean().item() if f.numel() else None,
                         'finite': bool(torch.isfinite(f).all())})
    write_csv(Path(directory) / f'tensors_{stage}_{timestamp}.csv', rows)


@torch.inference_mode()
def arithmetic_profile(model, size, directory, timestamp):
    """Profile the unmodified floating ULite graph, batch 1.

    MAC = one multiply-accumulate = 2 FLOPs (convention).
    Bias adds are extra. Eval BN is counted as precomputed affine 2 FLOPs/elem.
    GELU: 4 scalar arithmetic ops + 1 erf evaluation/elem; erf cost unassigned.
    Pool comparisons and memory copies are reported separately, not FLOPs.
    """
    model = model.cpu().eval()
    rows, hooks = [], []
    classes = (nn.Conv2d, nn.BatchNorm2d, nn.GELU, nn.MaxPool2d, nn.Upsample)
    def make_hook(name):
        def hook(module, inputs, output):
            elements = output.numel()
            row = {'timestamp_ist': timestamp, 'layer': name, 'type': type(module).__name__,
                   'input_shape': str(list(inputs[0].shape)), 'output_shape': str(list(output.shape)),
                   'output_elements': elements, 'output_bytes_fp32': elements*4,
                   'output_bytes_at_8bit': elements, 'conv_macs': 0,
                   'conv_flops_2mac': 0, 'bias_adds': 0, 'other_arithmetic_flops': 0,
                   'erf_evaluations': 0, 'pool_comparisons': 0, 'upsample_output_copies': 0,
                   'direct_parameters': sum(p.numel() for p in module.parameters(recurse=False))}
            if isinstance(module, nn.Conv2d):
                macs = elements*(module.in_channels//module.groups)*np.prod(module.kernel_size)
                row.update(conv_macs=int(macs), conv_flops_2mac=int(2*macs),
                           bias_adds=elements if module.bias is not None else 0)
            elif isinstance(module, nn.BatchNorm2d):
                row['other_arithmetic_flops'] = 2*elements
            elif isinstance(module, nn.GELU):
                row.update(other_arithmetic_flops=4*elements, erf_evaluations=elements)
            elif isinstance(module, nn.MaxPool2d):
                kernel = module.kernel_size
                area = kernel*kernel if isinstance(kernel, int) else int(np.prod(kernel))
                row['pool_comparisons'] = elements*(area-1)
            elif isinstance(module, nn.Upsample):
                row['upsample_output_copies'] = elements
            elif type(module).__name__ == 'AxialDW':
                # Two residual additions; convolutions already have their own rows.
                row['other_arithmetic_flops'] = 2*elements
            rows.append(row)
        return hook
    for name, module in model.named_modules():
        if isinstance(module, classes) or type(module).__name__ == 'AxialDW':
            hooks.append(module.register_forward_hook(make_hook(name)))
    try:
        output = model(torch.zeros(1, 3, size, size))
        if output.shape != (1, 1, size, size):
            raise RuntimeError('Unexpected model output shape.')
    finally:
        for hook in hooks:
            hook.remove()
    totals = {key: sum(row[key] for row in rows) for key in
              ['conv_macs', 'conv_flops_2mac', 'bias_adds', 'other_arithmetic_flops',
               'erf_evaluations', 'pool_comparisons', 'upsample_output_copies']}
    skip_elements = sum(channels*(size//(2**i))**2 for i, channels in enumerate([16, 32, 64, 128, 256]))
    totals.update(parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  parameter_bytes_fp32=sum(p.numel()*p.element_size() for p in model.parameters()),
                  input_shape=[1, 3, size, size], skip_tensor_elements_total=skip_elements,
                  skip_tensor_bytes_at_8bit=skip_elements,
                  skip_tensor_bytes_fp32=skip_elements*4,
                  first_skip_bytes_at_8bit=16*size*size,
                  estimated_arithmetic_flops_excluding_erf_and_memory=totals['conv_flops_2mac']+totals['bias_adds']+totals['other_arithmetic_flops'],
                  convention='1 MAC = 2 FLOPs. Conv bias adds extra. BN eval affine=2/element; GELU=4 arithmetic ops plus one separately counted erf. Pool comparisons and memory copies excluded from FLOPs. Concatenation traffic, allocation, quantize/requantize, optimizer/backward cost and hardware overhead are not counted.',
                  note='Architecture workload per image, not measured FPGA throughput. Sum of output bytes is not peak live memory. Parameter tensor inventories are separate CSVs.')
    write_csv(Path(directory) / f'layers_{timestamp}.csv', rows)
    write_csv(Path(directory) / f'profile_summary_{timestamp}.csv', [{'timestamp_ist': timestamp, **totals}])
    (Path(directory) / 'profile_summary.json').write_text(json.dumps(totals, indent=2))
    return totals


def exported_tensor_inventory(npz_path, directory, timestamp):
    rows = []
    with np.load(npz_path) as tensors:
        for name in tensors.files:
            v = tensors[name]
            rows.append({'timestamp_ist': timestamp, 'name': name, 'shape': str(list(v.shape)),
                         'dtype': str(v.dtype), 'elements': v.size, 'bytes': v.nbytes,
                         'min': float(v.min()) if v.size else None,
                         'max': float(v.max()) if v.size else None,
                         'mean': float(v.mean()) if v.size else None,
                         'std_population': float(v.std()) if v.size else None})
    write_csv(Path(directory) / f'exported_tensors_{timestamp}.csv', rows)
