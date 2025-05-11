import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from unet import UNet


def preprocess_pil(img: Image.Image, scale: float) -> torch.Tensor:
    """Resize & convert a PIL image to a 4‑D float tensor [1,C,H,W] (0‑1)."""
    if scale != 1.0:
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, resample=Image.BILINEAR)
    arr = np.array(img).transpose(2, 0, 1)  # C,H,W
    tensor = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
    return tensor


def predict_img(
    net: UNet,
    img: Image.Image,
    device: torch.device,
    scale: float,
    threshold: float,
):
    """Run a forward pass; return a PIL Image mask.

    * If ``net.n_classes == 1`` → Binary segmentation → output 0/255 mask.
    * Else                → Multi‑class segmentation → output trainId (0‑C) mask.
    """
    net.eval()
    x = preprocess_pil(img, scale).to(device)

    with torch.no_grad():
        logits = net(x)

    if net.n_classes > 1:
        # Multi‑class: softmax→argmax
        probs = F.softmax(logits, dim=1)[0]        # [C,H,W]
        mask  = probs.argmax(0).cpu().byte().numpy()
        return Image.fromarray(mask)
    else:
        # Binary: sigmoid→threshold
        probs = torch.sigmoid(logits)[0, 0]        # [H,W]
        mask  = (probs > threshold).cpu().numpy().astype(np.uint8) * 255
        return Image.fromarray(mask)


def get_args():
    parser = argparse.ArgumentParser(description="UNet inference (binary & multi‑class)")
    parser.add_argument('-m', '--model', required=True, help='Path to .pth model')
    parser.add_argument('-i', '--input', required=True, help='Input image or folder')
    parser.add_argument('-o', '--output', default='output', help='Output file or folder')
    parser.add_argument('-c', '--classes', type=int, default=19, help='Number of classes (1=binary)')
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Downscale factor')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for binary mask')
    parser.add_argument('--no-cuda', action='store_true', help='Force CPU')
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')

    net = UNet(n_channels=3, n_classes=args.classes).to(device)
    ckpt = torch.load(args.model, map_location=device)
    # 支援純 state_dict 或包含其他欄位的 ckpt
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    net.load_state_dict(state_dict, strict=False)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        img_files = list(in_path.glob('*.png')) + list(in_path.glob('*.jpg')) + list(in_path.glob('*.jpeg'))
    else:
        img_files = [in_path]
        if out_path.is_dir():
            out_path = out_path / (in_path.stem + '_mask.png')

    for img_f in img_files:
        img = Image.open(img_f).convert('RGB')
        mask = predict_img(net, img, device, args.scale, args.threshold)

        save_path = (
            out_path / (img_f.stem + '_mask.png') if out_path.is_dir() else out_path
        )
        mask.save(save_path)
        print(f'Saved {save_path}')


if __name__ == '__main__':
    main()
