import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

class HungarianMatcher:
    def __init__(self, cost_class=1.0, cost_mask=1.0, cost_params=1.0):
        """
        匈牙利匹配器，用于将模型预测与真实目标进行最优分配
        
        Args:
            cost_class: 分类损失的权重
            cost_mask: 掩码损失的权重  
            cost_params: 参数损失的权重
        """
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_params = cost_params
    
    def forward(self, outputs, targets):
        """
        执行匈牙利匹配
        
        Args:
            outputs: dict包含 'track_logits', 'hit_assignment', 'track_properties'
            targets: dict包含 'target_cls', 'target_masks', 'target_props'
        
        Returns:
            list of tuples (query_idx, target_idx) 表示匹配关系
        """
        with torch.no_grad():
            # 获取预测和目标
            pred_logits = outputs['track_logits']  # [n_queries]
            pred_masks = outputs['hit_assignment']   # [n_queries, n_hits]
            pred_params = outputs['track_properties'] # [n_queries, 4]
            
            target_cls = targets['target_cls']      # [n_tracks]
            target_masks = targets['target_masks']  # [n_tracks, n_hits]
            target_params = targets['target_props'] # [n_tracks, 4]
            
            n_queries = pred_logits.size(0)
            n_targets = target_cls.size(0)
            
            # 处理空预测的情况
            if n_queries == 0 or n_targets == 0:
                return []
            
            # 计算成本矩阵
            cost_matrix = self.compute_cost_matrix(
                pred_logits, pred_masks, pred_params,
                target_cls, target_masks, target_params
            )
            
            # 处理空成本矩阵
            if cost_matrix.numel() == 0:
                return []
            
            # 使用匈牙利算法求解
            query_indices, target_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            # 只保留有效的匹配（目标类别为1的）
            valid_matches = []
            for q_idx, t_idx in zip(query_indices, target_indices):
                if t_idx < n_targets and target_cls[t_idx] > 0.5:  # 有轨道的目标
                    valid_matches.append((q_idx, t_idx))
            
            return valid_matches
    
    def compute_cost_matrix(self, pred_logits, pred_masks, pred_params, 
                           target_cls, target_masks, target_params):
        """
        计算匹配的成本矩阵
        """
        n_queries = pred_logits.size(0)
        n_targets = target_cls.size(0)
        
        # 处理空预测的情况
        if n_queries == 0:
            return torch.empty(0, n_targets)
        if n_targets == 0:
            return torch.empty(n_queries, 0)
        
        # 处理可能的维度不匹配
        pred_n_hits = pred_masks.size(1) if len(pred_masks.shape) > 1 else 0
        target_n_hits = target_masks.size(1) if len(target_masks.shape) > 1 else 0
        
        # 确保hits维度匹配
        if pred_n_hits != target_n_hits and pred_n_hits > 0 and target_n_hits > 0:
            print(f"警告：匈牙利匹配中hits维度不匹配 - pred {pred_n_hits} vs target {target_n_hits}")
            
            # 取最小维度以避免索引错误
            min_hits = min(pred_n_hits, target_n_hits)
            pred_masks = pred_masks[:, :min_hits]
            target_masks = target_masks[:, :min_hits]
            print(f"调整后维度: pred_masks {pred_masks.shape}, target_masks {target_masks.shape}")
        
        # 扩展目标以匹配查询数量（添加"无轨道"类别）
        extended_target_cls = torch.zeros(n_queries, device=pred_logits.device)
        if n_targets > 0:
            extended_target_cls[:n_targets] = target_cls
        
        # 使用调整后的hits维度
        current_n_hits = pred_masks.size(1) if len(pred_masks.shape) > 1 else 0
        extended_target_masks = torch.zeros(n_queries, current_n_hits, device=pred_logits.device)
        if n_targets > 0 and current_n_hits > 0:
            extended_target_masks[:n_targets] = target_masks[:n_targets, :current_n_hits]
        
        n_params = pred_params.size(1)
        extended_target_params = torch.zeros(n_queries, n_params, device=pred_logits.device)
        if n_targets > 0:
            extended_target_params[:n_targets] = target_params
        
        # 计算各种成本
        # 1. 分类成本 - 使用概率差异
        pred_probs = torch.sigmoid(pred_logits)
        cost_class = torch.abs(pred_probs.unsqueeze(1) - extended_target_cls.unsqueeze(0))
        
        # 2. 掩码成本 - 使用Dice距离
        pred_mask_probs = torch.sigmoid(pred_masks)  # [n_queries, n_hits]
        
        # 计算每对query-target的Dice距离
        cost_mask = torch.zeros(n_queries, n_queries, device=pred_logits.device)
        for i in range(n_queries):
            for j in range(n_queries):
                if j < n_targets:
                    # 计算Dice距离
                    intersection = (pred_mask_probs[i] * extended_target_masks[j]).sum()
                    union = pred_mask_probs[i].sum() + extended_target_masks[j].sum()
                    dice = 1 - (2 * intersection) / (union + 1e-6)
                    cost_mask[i, j] = dice
                else:
                    # 对于"无轨道"目标，成本是预测掩码的均值（希望预测为全0）
                    cost_mask[i, j] = pred_mask_probs[i].mean()
        
        # 3. 参数成本 - 使用L2距离
        cost_params = torch.cdist(pred_params, extended_target_params, p=2)
        
        # 组合所有成本
        total_cost = (self.cost_class * cost_class + 
                     self.cost_mask * cost_mask + 
                     self.cost_params * cost_params)
        
        return total_cost
    


class LossModule(torch.nn.Module):
    def __init__(self, alpha=1000, beta=200, gamma=1, use_bce_mask=True, aux_loss_weight=0.1):
        super().__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta  # 掩码损失权重
        self.gamma = gamma  # 参数损失权重
        self.aux_loss_weight = aux_loss_weight  # 辅助损失权重
        self.use_bce_mask = use_bce_mask
        
        # 匈牙利匹配器
        self.matcher = HungarianMatcher(
            cost_class=1.0,
            cost_mask=5.0, 
            cost_params=1.0
        )

    def dice_loss(self, pred, target, eps=1e-6):
        num = 2 * (pred * target).sum(dim=1)
        denom = (pred + target).sum(dim=1) + eps
        dice = 1 - num / denom
        return dice.mean()
    
    def prepare_targets_with_matching(self, outputs, targets):
        """
        使用匈牙利匹配准备目标标签
        """
        n_queries = outputs['track_logits'].size(0)
        n_targets = targets['target_cls'].size(0)
        device = outputs['track_logits'].device
        
        # 执行匈牙利匹配
        matches = self.matcher.forward(outputs, targets)
        
        # 创建匹配后的目标
        matched_target_cls = torch.zeros(n_queries, device=device)
        matched_target_masks = torch.zeros_like(outputs['hit_assignment'])
        matched_target_params = torch.zeros_like(outputs['track_properties'])
        
        # 处理维度不匹配的情况
        pred_n_hits = outputs['hit_assignment'].size(1) if len(outputs['hit_assignment'].shape) > 1 else 0
        target_n_hits = targets['target_masks'].size(1) if len(targets['target_masks'].shape) > 1 else 0
        
        # 根据匹配结果分配目标
        for query_idx, target_idx in matches:
            matched_target_cls[query_idx] = 1.0  # 有轨道
            
            # 处理mask维度匹配
            if pred_n_hits > 0 and target_n_hits > 0:
                if pred_n_hits <= target_n_hits:
                    # 截断目标mask以匹配预测mask
                    matched_target_masks[query_idx] = targets['target_masks'][target_idx, :pred_n_hits]
                else:
                    # 填充目标mask以匹配预测mask
                    padding = torch.zeros(pred_n_hits - target_n_hits, device=device)
                    matched_target_masks[query_idx] = torch.cat([
                        targets['target_masks'][target_idx],
                        padding
                    ])
            
            matched_target_params[query_idx] = targets['target_props'][target_idx]
        
        # 未匹配的查询保持为0（无轨道）
        
        return matched_target_cls, matched_target_masks, matched_target_params

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
        # 准备输出和目标字典
        outputs = {
            'track_logits': track_logits,
            'hit_assignment': hit_masks,
            'track_properties': track_props
        }
        
        targets = {
            'target_cls': target_cls,
            'target_masks': target_masks,
            'target_props': target_props
        }
        
        # 使用匈牙利匹配重新排列目标
        matched_cls, matched_masks, matched_params = self.prepare_targets_with_matching(outputs, targets)
        
        # 计算主要损失
        cls_loss = F.binary_cross_entropy_with_logits(track_logits, matched_cls)

        pred_probs = torch.sigmoid(hit_masks)
        dice = self.dice_loss(pred_probs, matched_masks)
        if self.use_bce_mask:
            bce = F.binary_cross_entropy(pred_probs, matched_masks)
            mask_loss = 0.5 * dice + 0.5 * bce
        else:
            mask_loss = dice

        prop_loss = F.mse_loss(track_props, matched_params)

        total_loss = (
            self.alpha * cls_loss + self.beta * mask_loss + self.gamma * prop_loss
        )

        # 辅助损失（中间层输出）
        aux_loss = 0.0
        if intermediate_outputs is not None:
            for aux_output in intermediate_outputs:
                aux_outputs = {
                    'track_logits': aux_output['track_logits'],
                    'hit_assignment': aux_output['hit_assignment'],
                    'track_properties': aux_output['track_properties']
                }
                
                # 对每个辅助输出也进行匈牙利匹配
                aux_matched_cls, aux_matched_masks, aux_matched_params = \
                    self.prepare_targets_with_matching(aux_outputs, targets)
                
                aux_cls_loss = F.binary_cross_entropy_with_logits(
                    aux_output['track_logits'], aux_matched_cls
                )
                
                aux_pred_probs = torch.sigmoid(aux_output['hit_assignment'])
                aux_dice = self.dice_loss(aux_pred_probs, aux_matched_masks)
                if self.use_bce_mask:
                    aux_bce = F.binary_cross_entropy(aux_pred_probs, aux_matched_masks)
                    aux_mask_loss = 0.5 * aux_dice + 0.5 * aux_bce
                else:
                    aux_mask_loss = aux_dice
                
                aux_prop_loss = F.mse_loss(aux_output['track_properties'], aux_matched_params)
                
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
