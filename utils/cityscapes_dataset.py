from pathlib import Path
from PIL import Image
import numpy as np, torch
from torch.utils.data import Dataset

# 30→19 類對應表（255 代表 ignore，不計入 loss）
ID2TRAIN = {
     0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255,
     7: 0,  8: 1,  9:255,10:255,
    11: 2, 12: 3, 13: 4,
    14:255,15:255,16:255,
    17: 5, 18: 6, 19: 7, 20: 8,
    21: 9, 22:10, 23:11, 24:12, 25:13, 26:14, 27:15, 28:16, 29:17, 30:18,
    31:255,32:255,33:255
}

def _map_ids(arr):
    return np.vectorize(ID2TRAIN.get)(arr).astype('uint8')

class Cityscapes(Dataset):
    def __init__(self, root, split='train', scale=0.5):
        img_dir  = Path(root) / 'leftImg8bit' / split
        mask_dir = Path(root) / 'gtFine'      / split
        self.imgs  = sorted(img_dir.rglob('*_leftImg8bit.png'))
        self.masks = [mask_dir / p.parent.name / (p.stem.replace('leftImg8bit','gtFine_labelIds') + '.png')
                      for p in self.imgs]
        self.scale = scale

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img  = Image.open(self.imgs[idx]).convert('RGB')
        mask = Image.fromarray(_map_ids(np.array(Image.open(self.masks[idx]))))
        if self.scale != 1:
            new = (int(img.width*self.scale), int(img.height*self.scale))
            img  = img.resize(new, Image.BILINEAR)
            mask = mask.resize(new, Image.NEAREST)
        img  = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
