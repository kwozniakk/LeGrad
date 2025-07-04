import os
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from PIL import Image

from legrad import LeWrapper, visualize
from legrad.utils import load_model_by_repo_id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_repo = 'minchul/cvlface_adaface_vit_base_kprpe_webface12m'
dfa_repo = 'minchul/cvlface_DFA_mobilenet'
cache = os.path.expanduser('~/.cvlface_cache')
img_path = 'face.jpg'  # replace with your image path

# 1. Load CVLFace backbone + aligner
backbone = load_model_by_repo_id(vit_repo, f'{cache}/{vit_repo}', None).to(device).eval()
aligner = load_model_by_repo_id(dfa_repo, f'{cache}/{dfa_repo}', None).to(device).eval()
print("\u2713 Models loaded")

# 2. Preprocess and align image
preprocess = Compose([
    Resize((112, 112), interpolation=InterpolationMode.BILINEAR),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
aligned, *_ = aligner(image)

# 3. Equip backbone with LeGrad and compute explanation
model = LeWrapper(backbone)
heatmap = model.compute_legrad(image=aligned, text_embedding=None)

# 4. Visualize result
visualize(image=image, heatmaps=heatmap)
