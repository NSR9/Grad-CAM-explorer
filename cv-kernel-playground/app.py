import io
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models

st.set_page_config(page_title="Grad-CAM Explorer", layout="wide")
st.title("🔍 Grad-CAM Explorer")
st.caption("Upload an image → pick a layer → see what the CNN focuses on. Uses a pretrained ResNet-18 (ImageNet).")

# ----------------- Model setup -----------------
@st.cache_resource
def load_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.eval()
    return m

model = load_model()

# pick a target layer (last conv by default)
LAYER_OPTIONS = {
    "layer4 (last conv block)": "layer4",
    "layer3": "layer3",
    "layer2": "layer2",
    "layer1": "layer1",
}

layer_name = st.sidebar.selectbox("Target layer for Grad-CAM", list(LAYER_OPTIONS.keys()), index=0)
alpha = st.sidebar.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)

# ----------------- Hooks for Grad-CAM -----------------
activations = {}
gradients = {}

def hook_forward(module, inp, out):
    # Store activations without detaching to preserve gradient computation
    activations["value"] = out

def hook_backward(module, grad_in, grad_out):
    # Store gradients without detaching to preserve gradient computation
    gradients["value"] = grad_out[0]

def get_target_layer(model, name: str):
    return getattr(model, name)

target_layer = get_target_layer(model, LAYER_OPTIONS[layer_name])
# (Re)register hooks each run (Streamlit reruns script on interaction)
for h in getattr(model, "_gc_hooks", []):  # cleanup
    h.remove()
model._gc_hooks = [
    target_layer.register_forward_hook(hook_forward),
    target_layer.register_full_backward_hook(hook_backward),
]

# ----------------- Pre/Post -----------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
preprocess = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

@st.cache_data
def load_imagenet_labels():
    # minimal, stable list baked into torchvision weights metadata
    return models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]

classes = load_imagenet_labels()

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    if t.ndim == 3:
        t = t.permute(1,2,0).numpy()
        t = (t * 255).astype(np.uint8)
        return Image.fromarray(t)
    raise ValueError("Expected 3D CHW tensor")

def overlay_heatmap(img: Image.Image, heat: np.ndarray, alpha=0.45) -> Image.Image:
    # heat in [0,1], upscale to image size, apply COLORMAP_JET-like palette manually
    heat_rgb = np.uint8(255 * heat)
    # simple JET-ish palette
    import matplotlib.cm as cm
    cmap = cm.get_cmap("jet")
    colored = (cmap(heat_rgb)[:,:,:3] * 255).astype(np.uint8)
    colored = Image.fromarray(colored).resize(img.size, Image.BICUBIC)
    blend = Image.blend(img.convert("RGB"), colored, alpha=alpha)
    return blend

# ----------------- UI -----------------
up = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if not up:
    st.info("Tip: try a photo with a clear object (dog, bike, guitar, etc.).")
    st.stop()

orig = Image.open(up).convert("RGB")
inp = preprocess(orig).unsqueeze(0)  # 1x3x224x224

# Enable gradient computation for the input tensor
inp.requires_grad_(True)

# Forward pass to get logits
logits = model(inp)
probs = F.softmax(logits, dim=1)[0]
topk = torch.topk(probs, k=5)
top5_idx = topk.indices.tolist()
top5_val = topk.values.tolist()

# choose target class
top_labels = [f"{classes[i]} ({probs[i].item():.2f})" for i in top5_idx]
target_choice = st.selectbox("Target class for explanation", top_labels, index=0)
target_idx = top5_idx[top_labels.index(target_choice)]

# Grad-CAM: backprop on chosen class
model.zero_grad()
score = logits[0, target_idx]
score.backward(retain_graph=True)

A = activations["value"]           # [N, C, H, W]
dA = gradients["value"]            # [N, C, H, W]
weights = dA.mean(dim=(2,3), keepdim=True)  # GAP on gradients → [N,C,1,1]
cam = (weights * A).sum(dim=1, keepdim=True)  # [N,1,H,W]
cam = F.relu(cam)
cam = cam[0,0].detach().cpu().numpy()
# normalize to [0,1]
if cam.max() > cam.min():
    cam = (cam - cam.min()) / (cam.max() - cam.min())
else:
    cam = np.zeros_like(cam)

# upscale CAM to input display size
disp_transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
disp_img = disp_transform(orig)
cam_img = Image.fromarray(np.uint8(cam * 255)).resize(disp_img.size, Image.BICUBIC)
cam_np = np.asarray(cam_img, dtype=np.uint8) / 255.0

# ----------------- Layout -----------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Original")
    st.image(disp_img, use_container_width=True)
with c2:
    st.subheader("Grad-CAM Overlay")
    st.image(overlay_heatmap(disp_img, cam_np, alpha=alpha), use_container_width=True)

st.markdown("### Top-5 predictions")
for i, (idx, val) in enumerate(zip(top5_idx, top5_val), start=1):
    st.write(f"{i}. **{classes[idx]}** — {val:.3f}")

st.caption(
    "Grad-CAM = global-average-pooled gradients × activations at a chosen conv layer, "
    "ReLUed and upsampled onto the image. It highlights the regions most influential for the target class."
)
