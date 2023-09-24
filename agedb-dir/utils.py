import os
import shutil
import torch
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch.nn.functional as F

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args):
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    if os.path.exists(folders_util[-1]) and not args.resume and not args.pretrained and not args.evaluate:
        if query_yes_no('overwrite previous folder: {} ?'.format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + ' removed.')
        else:
            raise RuntimeError('Output folder {} already exists'.format(folders_util[-1]))
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(args, state, is_best, prefix=''):
    filename = f"{args.store_root}/{args.store_name}/{prefix}ckpt.pth.tar"
    torch.save(state, filename)
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def calculate_alpha(anchor, left, right):
    # calcuate the alpha for mixup positive samples

    if anchor == left:
        return 1
    elif anchor == right:
        return 0
    elif left == right:
        return 0
    else:
        return (anchor - right)/(left - right)

def remove_zero_rows(tensor):
    # Identify which rows have all their elements equal to 0
    zero_rows = (tensor == 0).all(dim=1)

    # Create a mask representing the rows to keep
    mask = ~zero_rows

    # Extract the non-zero rows from the tensor
    result = tensor.masked_select(mask.unsqueeze(1)).view(-1, tensor.size(1))

    # Get the indices of the removed rows
    removed_indices = torch.nonzero(zero_rows).flatten()

    return result, removed_indices

def remove_elements_by_indices(tensor, indices):
    # Create a mask representing the elements to keep
    mask = torch.ones_like(tensor, dtype=torch.bool)
    mask[indices] = False

    # Extract the elements from the tensor using the mask
    result = tensor.masked_select(mask)

    return result

# Calculate the alpha tensor for all i, j, k combinations
# def calculate_alpha_vectorized(i, j, k, sorted_targets):
#     # Assuming the original calculate_alpha function takes three scalars as input
#     alpha = torch.where(sorted_targets[j] < sorted_targets[i], sorted_targets[i] - sorted_targets[j], torch.zeros_like(sorted_targets[i])) / torch.where(sorted_targets[k] > sorted_targets[i], sorted_targets[k] - sorted_targets[i], torch.ones_like(sorted_targets[i]))
#     return alpha


def calculate_alpha_vectorized(i, j, k, sorted_targets):
    # Check if the inputs are valid
    assert torch.all(i < len(sorted_targets)), "All values in i should be less than the length of sorted_targets."
    assert torch.all(j < len(sorted_targets)), "All values in j should be less than the length of sorted_targets."
    assert torch.all(k < len(sorted_targets)), "All values in k should be less than the length of sorted_targets."
    
    # Use advanced indexing to gather the elements
    sorted_targets_i = sorted_targets[i]
    sorted_targets_j = sorted_targets[j]
    sorted_targets_k = sorted_targets[k]
    
    # Compute alpha
    alpha = (sorted_targets_k - sorted_targets_i) / (torch.clamp(sorted_targets_k - sorted_targets_j, min=1e-9))

    # Clip alpha to be in the range [0, 1]
    alpha = torch.clamp(alpha, min=0, max=1)

    return alpha

# # compress the mixtures matix 
# def compress_mixtures(mask, mixture):

#     # Convert mask to float type for operations
#     mask = mask.float()
    
#     # Create mask where elements are -1
#     minus_one_mask = mask == -1
#     # Create masked array
#     masked_array = mask.masked_fill(minus_one_mask, float('-inf'))
    
#     # Sort each row and track the indices
#     sorted_mask, index_matrix = torch.sort(masked_array, dim=1)
    
#     # Create mask where elements are -inf for the sorted mask
#     mask_sorted = sorted_mask == float('-inf')
#     # Fill in the mask
#     new_mask = sorted_mask.masked_fill(mask_sorted, -1)
#     index_matrix = index_matrix.masked_fill(mask_sorted, 0)  # Keep index_matrix masked with 0
    
#     # Create masks for columns with all -1 or -inf
#     new_mask_mask = ~(new_mask==-1).all(axis=0)
#     index_matrix_mask = ~(index_matrix==0).all(axis=0)  # Here, index_matrix will not have -1, so we keep it 0
    
#     # Remove all columns with all -1 or -inf
#     new_mask = new_mask[:, new_mask_mask]
#     index_matrix = index_matrix[:, index_matrix_mask]
    
#     # Apply the index to mixture 
#     valid_indices_mask = index_matrix.unsqueeze(-1) != 0  # Create a mask of valid indices
#     new_mixture = mixture.gather(1, index_matrix.unsqueeze(-1).expand(-1, -1, mixture.size(-1)))
#     new_mixture = new_mixture.masked_fill(~valid_indices_mask, -1)  # Fill invalid positions with -1

#     return new_mask, new_mixture

def compress_mixtures(mask, mixture, myop_label=None):

    # Convert mask to float type for operations
    mask = mask.float()
    
    # Create mask where elements are -1
    minus_one_mask = mask == -1
    
    # Create masked array
    masked_array = mask.masked_fill(minus_one_mask, float('-inf'))
    
    # Sort each row and track the indices
    sorted_mask, index_matrix = torch.sort(masked_array, dim=1)
    
    # Create mask where elements are -inf for the sorted mask
    mask_sorted = sorted_mask == float('-inf')
    
    # Fill in the mask
    new_mask = sorted_mask.masked_fill(mask_sorted, -1)
    
    index_matrix = index_matrix.masked_fill(mask_sorted, 0)  # Keep index_matrix masked with 0
    
    # Create masks for columns with all -1 or -inf
    new_mask_mask = ~(new_mask==-1).all(axis=0)
    index_matrix_mask = ~(index_matrix==0).all(axis=0)  # Here, index_matrix will not have -1, so we keep it 0
    
    # Remove all columns with all -1 or -inf
    new_mask = new_mask[:, new_mask_mask]
    index_matrix = index_matrix[:, index_matrix_mask]
    
    # Apply the index to mixture 
    valid_indices_mask = index_matrix.unsqueeze(-1) != 0  # Create a mask of valid indices
    new_mixture = mixture.gather(1, index_matrix.unsqueeze(-1).expand(-1, -1, mixture.size(-1)))
    new_mixture = new_mixture.masked_fill(~valid_indices_mask, -1)  # Fill invalid positions with -1

    # if myop_label is provided, apply the same operations as mask
    new_myop_label = None
    if myop_label is not None:
        myop_label = myop_label.float()
        
        minus_one_mask_myop = myop_label == -1
        masked_myop_label = myop_label.masked_fill(minus_one_mask_myop, float('-inf'))
        
        sorted_myop, _ = torch.sort(masked_myop_label, dim=1)
        
        myop_sorted = sorted_myop == float('-inf')
        
        new_myop_label = sorted_myop.masked_fill(myop_sorted, -1)
        
        new_myop_label = new_myop_label[:, new_mask_mask]  # Use the same mask as for new_mask

    return new_mask, new_mixture, new_myop_label


def compute_distance(A, B, dist_type='L1'):
    # Expand dims to [a, 1, c] and [1, b, c]
    A = A.unsqueeze(1)
    B = B.unsqueeze(0)

    if dist_type == 'L1':
        # Compute L1 distance
        dist = -1 * torch.abs(A - B).sum(-1)
    elif dist_type == 'L2':
        # Compute L2 distance
        dist = -1 * torch.sqrt((torch.abs(A - B) ** 2).sum(-1))
    elif dist_type == 'cosine':
        # Compute cosine distance
        cosine_sim = F.cosine_similarity(A, B, dim=-1)
        dist = 1 - cosine_sim
    else:
        raise ValueError("Invalid distance type. Use 'L1', 'L2' or 'cosine'")

    return dist

def compute_batch_distance(A, B, dist_type='L1'):
    # A = A.unsqueeze(2)
    B = B.unsqueeze(1)
    
    if dist_type == 'L1':
        dist = torch.abs(A - B).sum(-1)
    elif dist_type == 'L2':
        epsilon = 1
        dist = torch.sqrt(((A - B) ** 2).sum(-1) + epsilon)
    elif dist_type == 'cosine':
        cosine_sim = F.cosine_similarity(A, B, dim=-1)
        dist = 1 - cosine_sim
    else:
        raise ValueError("Invalid distance type. Use 'L1', 'L2' or 'cosine'")
    
    return dist