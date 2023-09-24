import torch
from torch.distributions.beta import Beta
from utils import compress_mixtures, calculate_alpha_vectorized
from einops import rearrange


def mixing_func(target, feature, mix_neg, mix_pos, n=0.5, b=0):
    
    target = target.squeeze(1).cpu()
    # print(target)
    # mixing-construct mixture tensor
    mixtures = torch.zeros((target.shape[0], target.shape[0], target.shape[0], feature.shape[-1])) 
    myop_label = torch.zeros((target.shape[0], target.shape[0], target.shape[0]))
    myop_mask = torch.zeros((target.shape[0], target.shape[0], target.shape[0])) - 1

    # mixing-condition for negative pairs
    i = torch.arange(target.shape[0]).reshape(-1, 1, 1)
    j = torch.arange(target.shape[0]).reshape(1, -1, 1)
    k = torch.arange(target.shape[0]).reshape(1, 1, -1)

    t_i = target.reshape(-1, 1, 1)
    t_j = target.reshape(1, -1, 1)
    t_k = target.reshape(1, 1, -1)

    neg_condition1 = torch.eq(i, j) & ~torch.eq(i, k)
    neg_condition2 = ~torch.eq(t_j, t_k)
    neg_condition = neg_condition1 & neg_condition2
    # print(sum(neg_condition.reshape(-1)))
    myop_mask[neg_condition] = 0

    # pos_condition = ((torch.lt(t_i, t_k) & (torch.abs(t_i - t_k) < n).int()) & (torch.lt(t_j, t_i) & (torch.abs(t_j - t_i) < n).int())).bool()
    pos_condition1 = torch.lt(t_i, t_k) & (torch.abs(t_i - t_k) < n)
    pos_condition2 = torch.lt(t_j, t_i) & (torch.abs(t_j - t_i) < n)
    pos_condition = pos_condition1 & pos_condition2
    myop_mask[pos_condition] = 1

    # --mixup--negatives--#
    if mix_neg:
        indices = torch.nonzero((myop_mask == 0))
        # print(target.shape, feature.shape, indices.shape)
        indices = indices.cpu().transpose(0, 1)
        
        if b == 0:
            beta = Beta(2, 8)
        elif b ==1:
            beta = Beta(8, 2)
        else:
            beta = Beta(5, 5)
        alpha = beta.sample((indices.shape[1],)).reshape(-1, 1)
        # print(alpha.shape, feature.shape, indices.shape)
        new_mixtures = alpha * feature.clone().cpu()[indices[1]] + (1 - alpha) * feature.clone().cpu()[indices[2]]
        new_labels = alpha.squeeze(1) * target.clone().cpu()[indices[1]] + (1-alpha.squeeze(1)) * target.clone().cpu()[indices[2]]

        mixtures[indices[0], indices[1], indices[2]] = new_mixtures
        myop_label[indices[0], indices[1], indices[2]] = new_labels.detach()
    # --mixup--negatives--#

    # --mixup--positives--#
    if mix_pos:
        indices = torch.nonzero((myop_mask == 1)).cpu()
        chunk_size = 1000
        num_indices = indices.shape[0]
        for start_idx in range(0, num_indices, chunk_size):
            end_idx = min(start_idx + chunk_size, num_indices)
            chunk_indices = indices[start_idx:end_idx]
            i, j, k = chunk_indices.transpose(0, 1)

            # Calculate alpha for all i, j, k combinations in the current chunk using the vectorized function
            alpha = calculate_alpha_vectorized(i, j, k, target).cpu().view(-1, 1)
            # Update mixtures using calculated alpha for the current chunk
            mixtures[i, j, k] = (alpha * feature.clone().cpu()[j]) + ((1 - alpha) * feature.clone().cpu()[k])
            myop_label[i, j, k] = target[i]
    # --mixup--positives--#

    mixtures_rearranged = rearrange(mixtures, 'b n m d -> b (n m) d')
    myop_mask_rearranged = rearrange(myop_mask, 'b n m -> b (n m)')
    myop_label_rearranged = rearrange(myop_label, 'b n m -> b (n m)')
    
    # print('before compress')
    # print(mixtures_rearranged.shape)
    # print(myop_mask_rearranged.shape)
    # print(myop_label_rearranged.shape)
    myop_mask_rearranged, mixtures_rearranged, myop_label_rearranged = compress_mixtures(myop_mask_rearranged, mixtures_rearranged, myop_label_rearranged) # compress matrix
    # print('after compress') # the compress is not working 
    # print(mixtures_rearranged.shape)
    # print(myop_mask_rearranged.shape)
    # print(myop_label_rearranged.shape)
    mixtures_rearranged = mixtures_rearranged.detach()
    myop_mask_rearranged = myop_mask_rearranged.detach()
    myop_label_rearranged = myop_label_rearranged.detach()

    return mixtures_rearranged, myop_mask_rearranged, myop_label_rearranged


