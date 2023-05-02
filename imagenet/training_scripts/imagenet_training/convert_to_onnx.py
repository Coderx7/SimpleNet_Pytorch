#in the name of God the most compassionate the most merciful 
# conver pytorch model to onnx models
import os
import argparse
import numpy as np

import torch
import onnx
import onnxruntime

from timm.models import create_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='simpnet', help='model architecture (default: simpnet)')
parser.add_argument('--num-classes', type=int, default=1000, help='Number classes in dataset')
parser.add_argument('--weights', default='', type=str, metavar='PATH', help='path to model weights (default: none)')
parser.add_argument('--output', default='simpnet.onnx', type=str, metavar='FILENAME', help='Output model file (.onnx model)')
# parser.add_argument('--opset', default=0, type=int, help='opset version (default:0) valid values, 0 to 10')
parser.add_argument('--use_input_dir', action='store_true', default=False, help='save in the same directory as input')
parser.add_argument('--jit', action='store_true', default=False, help='convert the model to jit before conversion to onnx')
parser.add_argument('--netscale', type=float, default=1.0, help='scale of the net (default 1.0)')
parser.add_argument('--netidx', type=int, default=0, help='which network to use (5mil or 8mil)')
parser.add_argument('--netmode', type=int, default=2, help='which stride mode to use(1 to 5)')
# parser.add_argument('--drop-rates', type=int, default=2, help='which stride mode to use(1 to 5)')

args = parser.parse_args()
# create model
model = create_model(args.model,
                     num_classes=args.num_classes,
                     checkpoint_path=args.weights,
                     scale=args.netscale,
                     network_idx = args.netidx, 
                     mode = args.netmode,)

print('Restoring model weights...')
model_weights = torch.load(args.weights, map_location='cpu')
model.load_state_dict(model_weights)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
   
new_output_name = args.output 
if args.use_input_dir:
    base_name = os.path.basename(args.weights)
    dir = args.weights.replace(base_name,'')
    new_output_name = os.path.join(dir,base_name.replace('.pth','.onnx'))

if args.jit:
    model = torch.jit.trace(model, dummy_input)
    model.save(f"{new_output_name.replace('.onnx','-jit')}.pt")

input_names = ["data"]
output_names = ["pred"]
# for caffe conversion its must be 9.
#! train mode crashes for some reason, need to report the bug.
torch.onnx.export(model, dummy_input, new_output_name, opset_version=9, verbose=True, input_names=input_names, output_names=output_names)

print(f'Converted successfully to onnx.')
print('Testing the new onnx model...')
# Load the ONNX model
model_onnx = onnx.load(new_output_name)
# Check that the model is well formed
onnx.checker.check_model(model_onnx)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model_onnx.graph))

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# pytorch model output
torch_out = model(dummy_input)
# onnx model output
ort_session = onnxruntime.InferenceSession(new_output_name)
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
