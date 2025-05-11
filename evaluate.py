import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff

def _batch_iou(pred: torch.Tensor, target: torch.Tensor,
               num_classes: int = 19, ignore_index: int = 255) -> float:
    # pred: [B, C, H, W] logits；target: [B, H, W] trainId
    pred = pred.argmax(dim=1).view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()

    mask = target != ignore_index
    pred, target = pred[mask], target[mask]

    ious = []
    for cls in range(num_classes):
        inter = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or (pred == cls, target == cls).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return np.mean(ious) if ious else float('nan')

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_sum = 0.0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            if isinstance(batch, dict):
                images, true_masks = batch['image'], batch['mask']
            else:
                images, true_masks = batch
            #image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                iou_sum  += _batch_iou(mask_pred, mask_true, net.n_classes)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                iou_sum  += _batch_iou(mask_pred, mask_true, net.n_classes)

    net.train()
    mean_dice = dice_score / max(num_val_batches, 1)
    mean_iou  = iou_sum  / max(num_val_batches, 1)
    return mean_dice, mean_iou
