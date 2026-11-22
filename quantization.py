"""W8A8 CPU reference and explicit hardware handoff for ULite.

PyTorch FX is deliberately version-pinned: it is a legacy API, not a promise of
support in future torch releases. See README for the toolchain boundary.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.fx import Interpreter
from torch.ao.quantization import get_default_qconfig_mapping, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx

ENGINE = 'qnnpack'  # Full uint8 activation range, per-tensor signed int8 weights.


def set_engine():
    if ENGINE not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f'This PyTorch build lacks {ENGINE}. Use the tested CPU/CUDA wheel.')
    torch.backends.quantized.engine = ENGINE


def prepare(model, size, qat=False):
    set_engine()
    example = (torch.zeros(1, 3, size, size),)
    if qat:
        return prepare_qat_fx(model.cpu().train(), get_default_qat_qconfig_mapping(ENGINE), example)
    return prepare_fx(model.cpu().eval(), get_default_qconfig_mapping(ENGINE), example)


class IntegerLookup(nn.Module):
    """256-entry uint8 table, optionally followed by nearest-neighbor copies.

    Floating point is used OFFLINE to construct the table. The forward path
    indexes integer bytes. Quantized tensor scales remain Python metadata.
    """
    def __init__(self, table, scale, zero_point, upsample=False):
        super().__init__()
        self.register_buffer('table', table.to(torch.uint8))
        self.scale, self.zero_point = float(scale), int(zero_point)
        self.upsample = bool(upsample)

    def forward(self, x):
        values = self.table[x.int_repr().long()]
        if self.upsample:
            values = values.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return torch._make_per_tensor_quantized_tensor(values, self.scale, self.zero_point)


def tensor_meta(value):
    if not isinstance(value, torch.Tensor):
        return None
    result = {'shape': list(value.shape), 'dtype': str(value.dtype),
              'quantized': value.is_quantized, 'elements': value.numel()}
    if value.is_quantized:
        result.update(scale=value.q_scale(), zero_point=value.q_zero_point())
    return result


class Capture(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.metadata = {}

    def run_node(self, node):
        result = super().run_node(node)
        self.metadata[node.name] = tensor_meta(result)
        return result


def lower_float_islands(model, example):
    """Replace FX's float GELU/Upsample islands with equivalent byte tables.

    Refuse unsupported graph shapes. Check the complete output before returning;
    the test suite additionally checks every input code for each table.
    """
    model.eval()
    with torch.inference_mode():
        capture = Capture(model)
        expected = capture.run(example)
    graph = model.graph
    replacements = []
    removed = set()
    for node in list(graph.nodes):
        if node in removed or node.op != 'call_module':
            continue
        operation = model.get_submodule(node.target)
        if not isinstance(operation, (nn.GELU, nn.Upsample)):
            continue
        source = node.args[0]
        if source.op != 'call_method' or source.target != 'dequantize':
            continue  # A later operation in an island is handled with its first.
        chain, cursor = [], node
        while cursor.op == 'call_module' and isinstance(model.get_submodule(cursor.target), (nn.GELU, nn.Upsample)):
            chain.append(cursor)
            users = list(cursor.users)
            if len(users) != 1:
                raise RuntimeError(f'Unsupported branching float island: {cursor.name}.')
            cursor = users[0]
        quantize = cursor
        if quantize.op != 'call_function' or quantize.target != torch.quantize_per_tensor:
            raise RuntimeError(f'No static output quantizer after {node.name}.')
        qsource = source.args[0]
        metadata = capture.metadata[qsource.name]
        if not metadata or metadata['dtype'] != 'torch.quint8':
            raise RuntimeError(f'Expected unsigned 8-bit input at {node.name}.')
        output_scale = float(getattr(model, quantize.args[1].target))
        output_zp = int(getattr(model, quantize.args[2].target))
        codes = torch.arange(256, dtype=torch.uint8)
        levels = torch._make_per_tensor_quantized_tensor(codes, metadata['scale'], metadata['zero_point']).dequantize()
        upsample, operation_names = False, []
        for member in chain:
            operation = model.get_submodule(member.target)
            if isinstance(operation, nn.Upsample):
                if upsample or operation.mode != 'nearest' or operation.scale_factor != 2:
                    raise RuntimeError('Only a single nearest 2x upsample per island is supported.')
                upsample = True
                operation_names.append('nearest2x')
            else:
                levels = operation(levels)
                operation_names.append('GELU')
        table = torch.quantize_per_tensor(levels, output_scale, output_zp, torch.quint8).int_repr()
        name = 'integer_' + node.name
        model.add_submodule(name, IntegerLookup(table, output_scale, output_zp, upsample))
        with graph.inserting_before(node):
            new_node = graph.call_module(name, args=(qsource,))
        quantize.replace_all_uses_with(new_node)
        graph.erase_node(quantize)
        for member in reversed(chain):
            graph.erase_node(member)
            removed.add(member)
        if not source.users:
            graph.erase_node(source)
        replacements.append({'node': name, 'operations': operation_names,
                             'input_scale': metadata['scale'],
                             'input_zero_point': metadata['zero_point'],
                             'output_scale': output_scale, 'output_zero_point': output_zp})
    graph.eliminate_dead_code()
    graph.lint()
    model.recompile()
    model.delete_all_unused_submodules()
    with torch.inference_mode():
        actual = model(example)
    if not torch.equal(expected, actual):
        error = float((expected - actual).abs().max())
        raise RuntimeError(f'Integer lookup lowering changed the output: max error {error}.')
    return model, replacements


def convert(prepared, example):
    set_engine()
    converted = convert_fx(copy.deepcopy(prepared).cpu().eval())
    return lower_float_islands(converted, example.cpu())


def audit(model, example):
    with torch.inference_mode():
        capture = Capture(model)
        capture.run(example)
    records, unexpected = [], []
    for node in model.graph.nodes:
        meta = capture.metadata[node.name]
        module = model.get_submodule(node.target) if node.op == 'call_module' else None
        row = {'name': node.name, 'kind': node.op, 'target': str(node.target),
               'inputs': [n.name for n in node.all_input_nodes], 'output': meta,
               'args_repr': str(node.args), 'kwargs_repr': str(node.kwargs)}
        if module is not None:
            row['module_type'] = type(module).__module__ + '.' + type(module).__name__
        records.append(row)
        if meta and not meta['quantized'] and node.op not in ('placeholder', 'get_attr', 'output'):
            is_final_dequant = node.op == 'call_method' and node.target == 'dequantize' and all(u.op == 'output' for u in node.users)
            if not is_final_dequant:
                unexpected.append(node.name)
    if unexpected:
        raise RuntimeError(f'Unexpected floating activation nodes: {unexpected}')
    return records


def export_handoff(model, example, directory, replacements):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    graph = audit(model, example)
    arrays, operators = {}, []
    qconv_count, weight_bytes, lut_bytes = 0, 0, 0
    by_name = {row['name']: row for row in graph}
    for row in graph:
        if row['kind'] != 'call_module':
            continue
        name = row['target']
        module = model.get_submodule(name)
        key = name.replace('.', '__')
        item = {'module': name, 'type': row['module_type'], 'output': row['output']}
        if isinstance(module, torch.ao.nn.quantized.Conv2d):
            weight = module.weight()
            if weight.qscheme() not in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                raise RuntimeError('Handoff expects the configured per-tensor weight scheme.')
            arrays[key + '__weight_int8'] = weight.int_repr().numpy()
            bias = module.bias()
            arrays[key + '__bias_fp32'] = bias.detach().numpy() if bias is not None else np.zeros(module.out_channels, np.float32)
            item.update(weight_key=key + '__weight_int8', bias_key=key + '__bias_fp32',
                        weight_scale=weight.q_scale(), weight_zero_point=weight.q_zero_point(),
                        kernel=list(module.kernel_size), stride=list(module.stride),
                        padding=list(module.padding), dilation=list(module.dilation),
                        groups=module.groups, weight_layout='OIHW')
            input_meta = by_name[row['inputs'][0]]['output']
            item['input'] = input_meta
            # Export the desired integer-convolution bias representation as a
            # design candidate; backend FP32-bias rounding can differ by one LSB.
            divisor = input_meta['scale'] * weight.q_scale()
            integer_bias = np.rint(arrays[key + '__bias_fp32'].astype(np.float64) / divisor)
            if np.any(np.abs(integer_bias) > np.iinfo(np.int32).max):
                raise RuntimeError(f'INT32 bias overflow at {name}.')
            arrays[key + '__bias_int32_candidate'] = integer_bias.astype(np.int32)
            item['bias_int32_candidate_key'] = key + '__bias_int32_candidate'
            item['requantization_real_multiplier'] = divisor / row['output']['scale']
            qconv_count += 1
            weight_bytes += weight.numel()
        elif isinstance(module, IntegerLookup):
            arrays[key + '__table_uint8'] = module.table.cpu().numpy()
            item.update(table_key=key + '__table_uint8', upsample_factor=2 if module.upsample else 1)
            lut_bytes += module.table.numel()
        elif isinstance(module, torch.ao.nn.quantized.BatchNorm2d):
            for field in ('weight', 'bias', 'running_mean', 'running_var'):
                arrays[key + '__' + field] = getattr(module, field).detach().numpy()
            item.update(eps=module.eps, array_prefix=key)
        elif isinstance(module, nn.MaxPool2d):
            item.update(kernel=module.kernel_size, stride=module.stride, padding=module.padding)
        operators.append(item)
    with torch.inference_mode():
        logits = model(example)
    arrays['example_input_fp32_nchw'] = example.cpu().numpy()
    arrays['example_logits_fp32_nchw'] = logits.cpu().numpy()
    arrays['example_binary_mask_uint8'] = (logits >= 0).to(torch.uint8).cpu().numpy()
    final_dequant = next(row for row in reversed(graph) if row['kind'] == 'call_method' and row['target'] == 'dequantize')
    final_meta = by_name[final_dequant['inputs'][0]]['output']
    arrays['example_logits_uint8_nchw'] = torch.quantize_per_tensor(logits, final_meta['scale'], final_meta['zero_point'], torch.quint8).int_repr().numpy()
    first_quant = next(row for row in graph if row['target'] == str(torch.quantize_per_tensor))
    input_scale, input_zp = first_quant['output']['scale'], first_quant['output']['zero_point']
    arrays['example_input_uint8_nchw'] = torch.quantize_per_tensor(example, input_scale, input_zp, torch.quint8).int_repr().numpy()
    np.savez(directory / 'tensors.npz', **arrays)
    report = {'format_version': 1, 'backend_reference': ENGINE,
              'weights': 'signed int8; per-tensor symmetric calibration',
              'activations': 'unsigned uint8, affine, range 0..255',
              'convolution_accumulator_target': 'int32; check overflow in hardware implementation',
              'input_preprocessing': 'RGB bilinear resize; NCHW float32 image / 255',
              'binary_decision': 'dequantized output logit >= 0; quantized output code >= output zero point',
              'quantized_convolutions': qconv_count, 'raw_weight_bytes': weight_bytes,
              'lookup_table_bytes': lut_bytes, 'unexpected_float_activation_nodes': [],
              'lookup_lowering': replacements, 'operators': operators, 'graph': graph,
              'limitations': [
                  'CPU reference artifact, not a PYNQ overlay or FINN/HLS input guaranteed to compile.',
                  'Quantized BN and convolution bias retain FP32 parameter metadata; implement and verify integer affine/requantization arithmetic separately.',
                  'bias_int32_candidate uses round-to-nearest-even; exact CPU backend bias/requantization rounding is not certified equivalent.',
                  'Graph tensor shapes and byte counts are not a measured peak live memory or DDR traffic estimate.',
                  'Residual additions and concatenations require matching the recorded scales and zero points.',
              ]}
    (directory / 'hardware_manifest.json').write_text(json.dumps(report, indent=2))
    (directory / 'fx_graph.txt').write_text(str(model.graph))
    traced = torch.jit.trace(model, example, check_trace=True)
    torch.jit.save(traced, str(directory / 'model_int8.ts'))
    loaded = torch.jit.load(str(directory / 'model_int8.ts')).eval()
    with torch.inference_mode():
        if not torch.equal(loaded(example), logits):
            raise RuntimeError('Reloaded INT8 model differs from exported model.')
    return report
