from typing import Dict
import numpy as np
import torch

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.stats import pearsonr, spearmanr, gmean


def get_text_field_mask(text_field_tensors: Dict[str, torch.Tensor]) -> torch.LongTensor:
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask of shape
    ``(batch_size, num_tokens)``.  This mask will be 0 where the tokens are padding, and 1
    otherwise.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we assume that the tensor in
    the dictionary with the lowest number of dimensions has plain token ids.  This allows us to
    also handle cases where the input is actually a ``ListField[TextField]``.

    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.byteTensors inside Variables makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    >>> mask = torch.ones([260]).byte()
    >>> mask.sum() # equals 260.
    >>> var_mask = torch.autograd.Variable(mask)
    >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])
    token_tensor = tensor_dims[0][1]

    return (token_tensor != 0).long()

def device_mapping(cuda_device: int):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """
    def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage
    return inner_device_mapping

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

def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.5, clip_max=2.):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 <= 0.).any() or (v2 < 0.).any():
        valid_pos = (((v1 > 0.) + (v2 >= 0.)) == 2)
        factor = torch.clamp(v2[valid_pos] / v1[valid_pos], clip_min, clip_max)
        matrix[:, valid_pos] = (matrix[:, valid_pos] - m1[valid_pos]) * torch.sqrt(factor) + m2[valid_pos]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2

def resume_checkpoint(model, model_state, backbone_only=False):
    model.pair_encoder.load_state_dict(
        {k.split('.', 1)[1]: v for k, v in model_state.items() if 'pair_encoder' in k}
    )
    if not backbone_only:
        getattr(model, 'sts-b_pred_layer').load_state_dict(
            {k.split('.', 1)[1]: v for k, v in model_state.items() if 'sts-b_pred_layer' in k}
        )

    return model

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        # kernel = gaussian(ks)
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

class STSShotAverage:
    def __init__(self, metric):
        self._pred = []
        self._label = []
        self._count = 0
        self._metric = metric
        self._num_bins = 50
        # under np.float32 division
        self._shot_idx = {
            'many': [0, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 49],
            'medium': [2, 4, 6, 8, 27, 35, 37],
            'few': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 29, 31, 33, 39, 41, 43, 45, 47]
        }

        def get_bin_idx(label):
            _, bins_edges = np.histogram(a=np.array([], dtype=np.float32), bins=self._num_bins, range=(0., 5.))
            if label == 5.:
                return self._num_bins - 1
            else:
                return np.where(bins_edges > label)[0][0] - 1
        self._get_bin_idx = get_bin_idx

    def __call__(self, pred, label):
        self._pred += pred.tolist()
        self._label += label.tolist()
        self._count += len(pred)

    def get_metric(self, reset=False, type=None):
        label_bin_idx = list(map(self._get_bin_idx, self._label))
        def bin2shot(idx):
            if idx in self._shot_idx['many']:
                return 'many'
            elif idx in self._shot_idx['medium']:
                return 'medium'
            else:
                return 'few'
        label_category = np.array(list(map(bin2shot, label_bin_idx)))

        pred_shot = {'many': [], 'medium': [], 'few': [], 'overall': []}
        label_shot = {'many': [], 'medium': [], 'few': [], 'overall': []}
        metric = {'many': {}, 'medium': {}, 'few': {}, 'overall': {}}
        for shot in ['overall', 'many', 'medium', 'few']:
            pred_shot[shot] = np.array(self._pred)[label_category == shot] * 5. if shot != 'overall' else np.array(self._pred) * 5.
            label_shot[shot] = np.array(self._label)[label_category == shot] if shot != 'overall' else np.array(self._label)
            if 'mse' in self._metric:
                metric[shot]['mse'] = np.mean((pred_shot[shot] - label_shot[shot]) ** 2) if pred_shot[shot].size > 0 else 0.
            if 'l1' in self._metric:
                metric[shot]['l1'] = np.mean(np.abs(pred_shot[shot] - label_shot[shot])) if pred_shot[shot].size > 0 else 0.
            if 'gmean' in self._metric:
                if pred_shot[shot].size <= 0:
                    metric[shot]['gmean'] = 0.
                else:
                    diff = np.abs(pred_shot[shot] - label_shot[shot])
                    if diff[diff == 0.].size:
                        diff[diff == 0.] += 1e-10
                        metric[shot]['gmean'] = gmean(diff) if pred_shot[shot].size > 0 else 0.
                    else:
                        metric[shot]['gmean'] = gmean(np.abs(pred_shot[shot] - label_shot[shot])) if pred_shot[shot].size > 0 else 0.
            if 'pearsonr' in self._metric:
                metric[shot]['pearsonr'] = pearsonr(pred_shot[shot], label_shot[shot])[0] if pred_shot[shot].size > 1 else 0.
            if 'spearmanr' in self._metric:
                metric[shot]['spearmanr'] = spearmanr(pred_shot[shot], label_shot[shot])[0] if pred_shot[shot].size > 1 else 0.
            metric[shot]['num_samples'] = pred_shot[shot].size
        if reset:
            self.reset()
        return metric['overall'] if type == 'overall' else metric


    def reset(self):
        self._pred = []
        self._label = []
        self._count = 0

def calculate_alpha_vectorized(i, j, k, sorted_targets):
    # Check if the inputs are valid
    assert (i < len(sorted_targets)).all(), "All values in i should be less than the length of sorted_targets."
    assert (j < len(sorted_targets)).all(), "All values in j should be less than the length of sorted_targets."
    assert (k < len(sorted_targets)).all(), "All values in k should be less than the length of sorted_targets."
    
    # Use advanced indexing to gather the elements
    sorted_targets_i = sorted_targets[i]
    sorted_targets_j = sorted_targets[j]
    sorted_targets_k = sorted_targets[k]
    
    # Compute alpha
    alpha = (sorted_targets_k - sorted_targets_i) / (torch.clamp(sorted_targets_k - sorted_targets_j, min=1e-9))

    # Clip alpha to be in the range [0, 1]
    alpha = torch.clamp(alpha, min=0, max=1)

    return alpha


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
    new_mask_mask = (new_mask != -1).sum(dim=0) == new_mask.size(0)
    index_matrix_mask = (index_matrix != 0).sum(dim=0) == new_mask.size(0)  # Here, index_matrix will not have -1, so we keep it 0
    
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