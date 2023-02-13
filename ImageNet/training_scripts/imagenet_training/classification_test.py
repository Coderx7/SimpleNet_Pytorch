#in the name of God the most compassionate the most merciful 
# classification test for trained models
import argparse
from PIL import Image

import torch
from timm.models import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='simpnet', help='model architecture (default: simpnet)')
parser.add_argument('--num-classes', type=int, default=1000, help='Number classes in dataset')
parser.add_argument('--weights', default='', type=str, metavar='PATH', help='path to model weights (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--jit', action='store_true', default=False, help='convert the model to jit before doing classification!')
parser.add_argument('--netscale', type=float, default=1.0, help='scale of the net (default 1.0)')
parser.add_argument('--netidx', type=int, default=0, help='which network to use (5mil or 8mil)')
parser.add_argument('--netmode', type=int, default=2, help='which stride mode to use(1 to 5)')


args = parser.parse_args()

# create model
model = create_model(
    args.model,
    num_classes=args.num_classes,
    pretrained=args.pretrained,
    checkpoint_path=args.weights,
    scale=args.netscale,
    network_idx = args.netidx,
    mode = args.netmode,
    )
model.eval()

if not args.pretrained and not args.weights:
    print(f'WARNING: No pretrained weights specified! (pretrained is False and there is no checkpoint specified!)') 

if args.jit:
    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    model = torch.jit.trace(model, dummy_input)

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

filename = "./misc_files/dog.jpg"
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0)
# save the transformed image for visualization or testing the ported models
torchvision.utils.save_image(tensor.squeeze(0),'img_test_transformed.jpg')

with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(f'{probabilities.shape}') # prints: torch.Size([1000])

filename="./misc_files/imagenet_classes.txt"
with open(filename, "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
print(f'Top categories:')
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

# prints class names and probabilities like:
# Samoyed 0.6719008088111877
# Pomeranian 0.0326448492705822
# Arctic fox 0.032615162432193756
# white wolf 0.030538644641637802
# Great Pyrenees 0.02852558344602585


