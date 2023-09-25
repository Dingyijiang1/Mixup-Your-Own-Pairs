from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils import compute_distance, compute_batch_distance

def scatter_(tensor, dim, index, value):
    tensor = tensor.to(index.device)
    if dim == 0:
        tensor[index] = value
    elif dim == 1:
        for i in range(tensor.size(0)):
            tensor[i][int(index[i].item())] = value
    else:
        raise ValueError("This function only supports dim 0 or 1.")
    return tensor


def ones_like(tensor):
    return torch.ones(tensor.size()).to(tensor.device)


class EmpiricalCDF:
    def __init__(self, tensor_1d):
        tensor_1d = tensor_1d.cuda()
        sorted_tensor, indices = torch.sort(tensor_1d)
        cdf_values = torch.arange(1, len(tensor_1d)+1, dtype=torch.float).cuda() / len(tensor_1d)
        self.sorted_tensor = sorted_tensor
        self.cdf_values = cdf_values

    def __call__(self, x):
        # Initialize tensor for CDF values
        cdf_values_x = torch.zeros_like(x)
        
        # Calculate CDF for each entry in tensor x
        for i in range(x.shape[0]):
            # Get the first index where sorted_tensor >= x[i]
            index = (self.sorted_tensor >= x[i]).nonzero(as_tuple=True)[0][0]
            
            # If x[i] is smaller than the smallest value in the tensor,
            # set its CDF value to 0.0 directly because it's out of the tensor's range.
            if index == 0:
                cdf_values_x[i] = torch.tensor(0.0).cuda()
            
            # Otherwise, set the CDF value at that index.
            else:
                cdf_values_x[i] = self.cdf_values[index - 1]

        return cdf_values_x

class SupConLoss_v2(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, feature_sim='cosine'):
        super(SupConLoss_v2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.feature_sim = feature_sim

    @staticmethod
    def check_nan(tensor, tensor_name):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{tensor_name} must be a PyTorch tensor.")
        
        if torch.isnan(tensor).any().item():
            print(f"NaN value encountered in {tensor_name}!")

    @staticmethod
    def calculate_weight(anchors, labels, myop_labels=None, distance='L1'):
        '''calcuate the weight that control the push strength for negative pairs through L1 distance
        Args:
            anchors: 1D tensor [N, ]
            labels : 1D tensor [N, ]
            myop_labels: 2D tensor [N, M]
        Returns:
            weights: 2D tensor [N, N], 
        '''

        if distance == 'L1':
            weights = torch.reciprocal(torch.abs(anchors.view(-1, 1) - labels.view(1, -1)) + 1) # [1, 1e-2]
        elif distance == 'L2':
            weights = torch.reciprocal(torch.pow((anchors.view(-1, 1) - labels.view(1, -1)), 2) + 1) # [1, 1e-4]
        elif distance == 'L1_inv':
            weights = torch.abs(anchors.view(-1, 1) - labels.view(1, -1)) + 1  # [1, 1e2]
        elif distance == 'L2_inv':
            weights = torch.pow((anchors.view(-1, 1) - labels.view(1, -1)), 2) + 1 # [1, 1e4]
        
        if myop_labels is not None:
            if distance == 'L1':
                weights = torch.cat([weights, torch.reciprocal(torch.abs(anchors.unsqueeze(1).expand_as(myop_labels) - myop_labels) + 1)], dim=1)
            elif distance == 'L2':
                weights = torch.cat([weights, torch.reciprocal(torch.pow(anchors.unsqueeze(1).expand_as(myop_labels) - myop_labels, 2) + 1)], dim=1)
            elif distance == 'L1_inv':
                weights = torch.cat([weights, torch.abs(anchors.unsqueeze(1).expand_as(myop_labels) - myop_labels) + 1], dim=1)
            elif distance == 'L2_inv':
                weights = torch.cat([weights, torch.pow((anchors.unsqueeze(1).expand_as(myop_labels) - myop_labels + 1), 2) + 1], dim=1)
        
        return weights

    def forward(self, features, labels=None, mask=None, mixtures=None, myop_mask=None, myop_label=None, use_weight=False, distance='L1', feature_sim='cosine'):
        """Compute loss for model. If both `labels` and `mask` are None,
        Args:
            features: hidden vector of shape [bsz, n_views, dim].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
            mixtures: mixup for hard-neg and hard-pos [2*bsz, 4*bsz^2, dim] 
            myop_mask: mask for mixtures [2*bsz, 4*bsz^2] (1: positive, 0: negative, -1: execlude)
            myop_label: labels for the mixtures [2*bsz, 4*bsz^2] where each entry filled with (calcualted) targets for the mixtures
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.transpose(0, 1)).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if self.feature_sim == 'L1':
            anchor_dot_contrast = torch.div(compute_distance(anchor_feature, contrast_feature, dist_type='L1') , self.temperature)
        elif self.feature_sim == 'L2':
            anchor_dot_contrast = torch.div(compute_distance(anchor_feature, contrast_feature, dist_type='L2'), self.temperature)
        else:
            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.transpose(0, 1)), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        
        # Original tensor full of ones
        tensor_ones = torch.ones(batch_size * anchor_count, dtype=torch.float32).to(device)

        # Indices
        indices = torch.arange(batch_size * anchor_count).long().view(-1, 1).to(device)

        # Using the custom scatter
        tensor_ones = ones_like(mask)
        indices = torch.arange(batch_size * anchor_count).view(-1, 1).to(device)
        logits_mask = scatter_(tensor_ones, 1, indices, 0)
        
        mask = mask * logits_mask

        # myop start here
        if mixtures is not  None:
            # anchor_feature = features[:, 1] # mixtures for the second branch only
            if self.feature_sim == 'L1':
                anchor_dot_mixture = torch.div(compute_batch_distance(mixtures, anchor_feature, dist_type='L1'), self.temperature)
            if self.feature_sim == 'L2':
                anchor_dot_mixture = torch.div(compute_batch_distance(mixtures, anchor_feature, dist_type='L2'), self.temperature)
            else:
                anchor_dot_mixture = torch.div(torch.bmm(mixtures, anchor_feature.unsqueeze(2)).squeeze(2), self.temperature)
            # self.check_nan(anchor_dot_mixture, 'anchor_dot_mixture')
            
            # this is problemetic ?
            # max_val, _ = torch.max(anchor_dot_mixture, dim=1, keepdim=True) # for numerical stability
            # myop_logits = torch.exp((anchor_dot_mixture - max_val.detach()) / self.temperature) # for numerical stability

            max_val, _ = torch.max(anchor_dot_mixture, dim=1, keepdim=True) # for numerical stability
            myop_logits = anchor_dot_mixture - max_val.detach() # for numerical stability

            # print("myop_logits shape:", myop_logits.shape) #[64, 58]
            # print("logits shape:", logits.shape) #[64, 64]
            # print("myop_mask shape:", myop_mask.shape) #[64, 58]
            # print("mask shape:", mask.shape) #[64, 64]

            logits = torch.cat([logits, myop_logits], dim=1)
            mask = torch.cat([mask, myop_mask], dim=1)

            temp_logits_mask = torch.where(myop_mask != -1, torch.ones_like(myop_mask), torch.zeros_like(myop_mask))
            # print("temp_logits_mask shape: ", temp_logits_mask.shape)
            # print("logits_mask", logits_mask.shape)
            logits_mask = torch.cat([logits_mask, temp_logits_mask], dim=1)
    
            # print("logits_mask shape:", logits_mask.shape) #[64, 59]
            # print("myop_mask shape:", mask.shape) # [64, 122]
            # print("logits shape:", logits.shape)# [64, 122]
        # myop end here

        # compute log_prob
        if use_weight and mixtures is not None:
            neg_weight = self.calculate_weight(labels.squeeze(1), labels.squeeze(1), myop_label , distance=distance)
            # self.check_nan(neg_weight, 'weight for negative pairs')
            exp_logits = torch.exp(logits) * logits_mask * neg_weight # add the weight 
            # self.check_nan(exp_logits, 'weight for exp logits')
            log_prob = (logits - torch.log(exp_logits.sum(1, keepdim=True))) 
            # self.check_nan(log_prob, 'weight for log prob')
        elif use_weight:
            neg_weight = self.calculate_weight(labels.squeeze(1), labels.squeeze(1), distance=distance) 
            exp_logits = torch.exp(logits) * logits_mask * neg_weight # add the weight 
            log_prob = (logits - torch.log(exp_logits.sum(1, keepdim=True))) 
        else:
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = (logits - torch.log(exp_logits.sum(1, keepdim=True))) 
        
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = ((mask == 1).float() * log_prob).sum(1) / ((mask == 1).float().sum(1) + 1.0)
        # self.check_nan(mean_log_prob_pos, 'mean_log_prob_pos')
        # mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0
        
        # Calculate average similarity for positive and negative pairs
        positive_similarity = ((mask == 1).float() * torch.exp(logits)).sum(1) / ((mask == 1).float().sum(1) + 1.0)
        negative_similarity = ((mask == 0).float() * torch.exp(logits)).sum(1) / ((mask == 0).float().sum(1) + 1.0)
        avg_positive_similarity = positive_similarity.mean().item()
        avg_negative_similarity = negative_similarity.mean().item()

        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, avg_positive_similarity, avg_negative_similarity

'''
supervised SimCSE
'''
class SupSimCSE(nn.Module):
    def __init__(self, temperature=0.05, base_temperature=0.05, feature_sim='cosine'):
        super(SupSimCSE, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.feature_sim = feature_sim

    def forward(self, f1, f2, label, cutoff=2):
        sim_matrix = torch.matmul(f1, f2.transpose(0, 1))
        sim_matrix = torch.diag(sim_matrix)
      
        # compute log_prob
        exp_logits = torch.exp(sim_matrix / self.temperature)
        log_prob = sim_matrix - torch.log(exp_logits.sum(0, keepdim=True))
        
        loss = F.cross_entropy(log_prob.view(1, -1), label > cutoff)
        return loss, 0, 0