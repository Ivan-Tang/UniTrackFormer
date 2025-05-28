import torch
import torch.nn.functional as F


class LossModule(torch.nn.Module):
    def __init__(self, alpha=1000, beta=200, gamma=1, use_bce_mask=True):
        super().__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta  # 掩码损失权重
        self.gamma = gamma  # 参数损失权重
        self.use_bce_mask = use_bce_mask

    def dice_loss(self, pred, target, eps=1e-6):
        num = 2 * (pred * target).sum(dim=1)
        denom = (pred + target).sum(dim=1) + eps
        dice = 1 - num / denom
        return dice.mean()

    def forward(
        self,
        track_logits,
        hit_masks,
        track_props,
        target_cls,
        target_masks,
        target_props,
    ):
        cls_loss = F.binary_cross_entropy_with_logits(track_logits, target_cls)

        pred_probs = torch.sigmoid(hit_masks)
        dice = self.dice_loss(pred_probs, target_masks)
        if self.use_bce_mask:
            bce = F.binary_cross_entropy(pred_probs, target_masks)
            mask_loss = 0.5 * dice + 0.5 * bce
        else:
            mask_loss = dice

        prop_loss = F.mse_loss(track_props, target_props)

        total_loss = (
            self.alpha * cls_loss + self.beta * mask_loss + self.gamma * prop_loss
        )

        return {
            "total": total_loss,
            "cls": self.alpha * cls_loss,
            "mask": self.beta * mask_loss,
            "param": self.gamma * prop_loss,
        }
