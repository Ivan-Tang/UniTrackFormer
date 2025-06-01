import torch
import torch.nn.functional as F


class LossModule(torch.nn.Module):
    def __init__(self, alpha=1000, beta=200, gamma=1, use_bce_mask=True, aux_loss_weight=0.1):
        super().__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta  # 掩码损失权重
        self.gamma = gamma  # 参数损失权重
        self.aux_loss_weight = aux_loss_weight  # 辅助损失权重
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
        intermediate_outputs=None,
        hit_scores=None,
        hit_targets=None,
    ):
        # Main task losses
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

        # Auxiliary losses from intermediate decoder outputs
        aux_loss = 0.0
        if intermediate_outputs is not None:
            for aux_output in intermediate_outputs:
                aux_cls_loss = F.binary_cross_entropy_with_logits(
                    aux_output['track_logits'], target_cls
                )
                
                aux_pred_probs = torch.sigmoid(aux_output['hit_assignment'])
                aux_dice = self.dice_loss(aux_pred_probs, target_masks)
                if self.use_bce_mask:
                    aux_bce = F.binary_cross_entropy(aux_pred_probs, target_masks)
                    aux_mask_loss = 0.5 * aux_dice + 0.5 * aux_bce
                else:
                    aux_mask_loss = aux_dice
                
                aux_prop_loss = F.mse_loss(aux_output['track_properties'], target_props)
                
                aux_loss += (
                    self.alpha * aux_cls_loss + 
                    self.beta * aux_mask_loss + 
                    self.gamma * aux_prop_loss
                )
            
            aux_loss = aux_loss * self.aux_loss_weight

        # Hit filtering loss (if provided)
        hit_filter_loss = 0.0
        if hit_scores is not None and hit_targets is not None:
            hit_filter_loss = F.binary_cross_entropy_with_logits(hit_scores, hit_targets)

        total_loss = total_loss + aux_loss + hit_filter_loss

        return {
            "total": total_loss,
            "cls": self.alpha * cls_loss,
            "mask": self.beta * mask_loss,
            "param": self.gamma * prop_loss,
            "aux": torch.tensor(aux_loss) if isinstance(aux_loss, float) else aux_loss,
            "hit_filter": torch.tensor(hit_filter_loss) if isinstance(hit_filter_loss, float) else hit_filter_loss,
        }
