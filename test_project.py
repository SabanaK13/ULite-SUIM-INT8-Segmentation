"""Unit and quantization checks; real CUDA AMP test runs only when CUDA exists."""
import copy
import tempfile
import unittest
from unittest.mock import patch
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.ao.quantization.quantize_fx import convert_fx

from model import ULite
from saliency import rgb_to_saliency
from quantization import prepare, convert, audit, IntegerLookup, export_handoff
from profiling import arithmetic_profile, tensor_inventory
from workflow import compare, aggregate


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_saliency_grouping(self):
        colors = np.array([[[0,0,0], [0,0,255], [0,255,0], [0,255,255],
                            [255,0,0], [255,0,255], [255,255,0], [255,255,255]]], np.uint8)
        np.testing.assert_array_equal(rgb_to_saliency(colors), [[0,1,0,1,1,0,1,0]])
        # Original SUIM masks can contain intermediate channel values.
        near_colors = np.array([[[10,20,230], [0,190,220], [200,15,25],
                                 [200,240,3], [200,5,220], [127,128,127]]], np.uint8)
        np.testing.assert_array_equal(rgb_to_saliency(near_colors), [[1,1,1,1,0,0]])
        for bad in [np.zeros((2,2),np.uint8), np.zeros((2,2,3),np.float32)]:
            with self.assertRaises(ValueError): rgb_to_saliency(bad)

    def test_profile_and_parameter_inventory(self):
        model = ULite()
        with tempfile.TemporaryDirectory() as tmp:
            report = arithmetic_profile(model, 256, tmp, 'test')
            self.assertEqual(report['parameters'], 878417)
            self.assertEqual(report['conv_flops_2mac'], report['conv_macs']*2)
            self.assertEqual(report['first_skip_bytes_at_8bit'], 1048576)
            tensor_inventory(model, tmp, 'test', 'initial')
            self.assertIn('conv_out.weight', (Path(tmp)/'tensors_initial_test.csv').read_text())

    def test_ptq_integer_islands_and_export(self):
        torch.manual_seed(7)
        x = torch.rand(1,3,64,64)
        prepared = prepare(ULite(),64)
        with torch.no_grad():
            for _ in range(3): prepared(torch.rand_like(x))
        native = convert_fx(copy.deepcopy(prepared).eval())
        quantized, changes = convert(prepared,x)
        self.assertEqual(len(changes),11)
        self.assertEqual(sum('nearest2x' in c['operations'] for c in changes),5)
        audit(quantized,x)
        with torch.inference_mode():
            for sample in [x,torch.zeros_like(x),torch.ones_like(x),torch.rand_like(x)]:
                torch.testing.assert_close(quantized(sample),native(sample),rtol=0,atol=0)
            for change in changes:
                codes = torch.arange(256,dtype=torch.uint8).reshape(1,1,1,256)
                values = torch._make_per_tensor_quantized_tensor(codes,change['input_scale'],change['input_zero_point'])
                expected = values.dequantize()
                for operation in change['operations']:
                    if operation=='GELU': expected=torch.nn.functional.gelu(expected)
                    else: expected=torch.nn.functional.interpolate(expected,scale_factor=2,mode='nearest')
                expected = torch.quantize_per_tensor(expected,change['output_scale'],change['output_zero_point'],torch.quint8)
                actual=quantized.get_submodule(change['node'])(values)
                self.assertTrue(torch.equal(actual.int_repr(),expected.int_repr()))
        with tempfile.TemporaryDirectory() as tmp:
            report=export_handoff(quantized,x,tmp,changes)
            self.assertEqual(report['quantized_convolutions'],45)
            self.assertEqual(report['raw_weight_bytes'],870016)
            self.assertEqual(report['unexpected_float_activation_nodes'],[])

    def test_qat_backward_conversion(self):
        model=prepare(ULite(),64,qat=True)
        x=torch.rand(2,3,64,64)
        y=torch.randint(0,2,(2,1,64,64)).float()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        q,_=convert(model,x[:1])
        self.assertEqual(tuple(q(x[:1]).shape),(1,1,64,64))

    def test_comparison_uses_converted_validation_and_checks_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg={'run_root':tmp}
            reports=[]
            for route,score in [('fp32',0.6),('amp_fp16',0.65),('qat',0.7)]:
                reports.append({'timestamp_ist':'synthetic test fixture', 'route':route,
                                'manifest_hash':'fixed','config':cfg,'unexpected_float_activation_nodes':[],
                                'environment':{'platform':'fixture','processor':'fixture'},
                                'validation':{'foreground_iou_global':score,'foreground_dice_global':score},
                                'parent_validation_iou':0.9 if route=='fp32' else score,
                                'quantized_model_file_bytes':100,'raw_int8_weight_bytes':50,
                                'latency':{'cpu_median_ms':1},'export_directory':'fixture-only'})
            args=argparse.Namespace(seed=42,seeds=[42])
            with patch('workflow.read_manifest',return_value=({},'fixed')), patch('workflow.candidate_reports',return_value=reports):
                compare(cfg,args)
                pointer=json.loads((Path(tmp)/'selection_seed_42.json').read_text())
                selection_path=Path(pointer['selection_file'])
                selection=json.loads(selection_path.read_text())
                self.assertEqual(selection['selected_route'],'qat')
                aggregate(cfg,args)
                reports[0]['manifest_hash']='changed'
                with self.assertRaises(ValueError): compare(cfg,args)
                reports[0]['manifest_hash']='fixed'
                selection['test_evaluation_started']=True
                selection_path.write_text(json.dumps(selection))
                with self.assertRaises(ValueError): compare(cfg,args)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required: CPU tests cannot validate FP16 GPU training')
    def test_cuda_amp_backward(self):
        model=ULite().cuda()
        optimizer=torch.optim.Adam(model.parameters())
        scaler=torch.amp.GradScaler('cuda')
        x=torch.rand(2,3,64,64,device='cuda')
        y=torch.randint(0,2,(2,1,64,64),device='cuda').float()
        with torch.autocast('cuda',dtype=torch.float16):
            z=model(x)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),y)
        self.assertEqual(z.dtype,torch.float16)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        self.assertTrue(torch.isfinite(loss))


if __name__=='__main__':
    unittest.main()
