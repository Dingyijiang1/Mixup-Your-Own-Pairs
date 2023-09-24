import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean
from einops import rearrange

import torch.multiprocessing as mp
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboard_logger import Logger
from torch.distributions.beta import Beta
from torch.cuda.amp import GradScaler, autocast
import torch.optim.lr_scheduler as lr_scheduler

from resnet import resnet50
from loss import *
from datasets import IMDBWIKI
from utils import *
from supremix_loss import SupConLoss_v2

import os
os.environ["KMP_WARNINGS"] = "FALSE"


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# contrastive learning related
parser.add_argument('--contrastive_method', default='supcon', choices=['simclr', 'supcon',  'supremix', 'mix_neg', 'mix_pos'], help='which contrastive method to use')
parser.add_argument('--use_weight', type=bool, default=True, help='whether to use weight for negative pairs')
parser.add_argument('--distance', default='L1_inv', choices=['L1', 'L2', 'L1_inv', 'L2_inv'])
parser.add_argument('--temperature', type=float, default=0.3, help='temperature hyperparameter')
parser.add_argument('--pool', type=int, default=1)
parser.add_argument('--reduction', type=str, default='mean', choices=['mean', 'sum'])
parser.add_argument('--dim', default=128, type=int)
parser.add_argument('--save_freq', default=None, type=int)
parser.add_argument('--scheduler', default=False, action='store_true')
parser.add_argument('--beta', default=0, type=int, choices=[0, 1, 2], help='0 : beta(5, 5), 1: beta(2, 8), 2: beta(8, 2)')
parser.add_argument('--n', default=5, type=int)

# path related
parser.add_argument('--dataset', type=str, default='imdb_wiki', help='dataset name')
parser.add_argument('--data_dir', type=str, default='/data', help='data directory')
parser.add_argument('--store_root', type=str, default='/checkpoint', help='path to the dataset')
# training/optimization related
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--print_freq', type=int, default=100, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--workers', type=int, default=10, help='number of workers used in data loading')

# checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--pretrained', type=str, default='', help='checkpoint file path to load backbone weights')


parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

args.start_epoch, args.best_loss = 0, 1e5
args.evaluate = False

if len(args.store_name):
    args.store_name = f'_{args.store_name}'
args.store_name += f'_{args.contrastive_method}_{args.temperature}'
if args.use_weight:
    args.store_name += f'_weighted_neg_{args.distance}'
args.store_name = f"{args.dataset}_{args.model}{args.store_name}_{args.batch_size}_{args.pool}_{args.lr}_{args.beta}"
prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")

tb_logger = Logger(logdir=os.path.join(args.store_root, args.store_name), flush_secs=2)


def main():
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # Data
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))

    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']

    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train_cl')
    val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_val, img_size=args.img_size, split='val')
    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model
    print('=====> Building model...')
    model = resnet50()
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    


    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
        torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)




    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume) if args.gpu is None else \
                torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss, pos_sim, neg_sim = train(train_loader, model, optimizer, epoch, scheduler=None, n=args.n, b=args.beta)

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False)

        tb_logger.log_value('train_loss', train_loss, epoch)
        tb_logger.log_value('positive similarities', pos_sim, epoch)
        tb_logger.log_value('negative similarities', neg_sim, epoch)

    # test with best checkpoint
    print("=" * 120)

def mixing_func(target, feature, mix_neg, mix_pos, n, b):
    target = target.repeat(2, 1).squeeze(1).cpu()
    # mixing-construct mixture tensor
    mixtures = torch.zeros((target.shape[0], target.shape[0], target.shape[0], args.dim)) 
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
    myop_mask[neg_condition] = 0
    pos_condition = ((torch.lt(t_i, t_k) & (torch.abs(t_i - t_k) < n).int()) & (torch.lt(t_j, t_i) & (torch.abs(t_j - t_i) < n).int())).bool()
    myop_mask[pos_condition] = 1

    # --mixup--negatives--#
    if mix_neg:
        indices = torch.nonzero((myop_mask == 0)).cpu().t() 
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
            i, j, k = chunk_indices.T

            # Calculate alpha for all i, j, k combinations in the current chunk using the vectorized function
            alpha = calculate_alpha_vectorized(i, j, k, target).cpu().view(-1, 1)
            # Update mixtures using calculated alpha for the current chunk
            mixtures[i, j, k] = (alpha * feature.clone().cpu()[j]) + ((1 - alpha) * feature.clone().cpu()[k])
            myop_label[i, j, k] = target[i]
    # --mixup--positives--#

    mixtures_rearranged = rearrange(mixtures, 'b n m d -> b (n m) d')
    myop_mask_rearranged = rearrange(myop_mask, 'b n m -> b (n m)')
    myop_label_rearranged = rearrange(myop_label, 'b n m -> b (n m)')

    myop_mask_rearranged, mixtures_rearranged, myop_label_rearranged = compress_mixtures(myop_mask_rearranged, mixtures_rearranged, myop_label_rearranged) # compress matrix
    mixtures_rearranged = mixtures_rearranged.detach()
    myop_mask_rearranged = myop_mask_rearranged.detach()
    myop_label_rearranged = myop_label_rearranged.detach()

    return mixtures_rearranged, myop_mask_rearranged, myop_label_rearranged


def train(train_loader, model, optimizer, epoch, scheduler, n, b):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter(f'Loss ({args.contrastive_method.upper()})', ':.3f')
    pos_sims = AverageMeter('Positive similarities', ':3f')
    neg_sims = AverageMeter('Negative similarities', ':3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, pos_sims, neg_sims],
        prefix="Epoch: [{}]".format(epoch)
    )
    scaler = GradScaler()
    model.train()
    end = time.time()
    for idx, (inputs, targets, weights) in enumerate(train_loader):
        targets, weights = targets.cuda(non_blocking=True), weights.cuda(non_blocking=True)
        bsz = targets.shape[0]
        inputs = torch.cat([inputs[0], inputs[1]], dim=0).cuda(non_blocking=True)
        with autocast():
            # Forward pass
            features = model(inputs, targets, epoch, output_embedding=True)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            if args.contrastive_method in ['supcon_uni', 'simclr_uni']:
                f2 = f2.detach()
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # [b, view, feature_dim]
        
        if args.contrastive_method in ['simclr', 'simclr_uni']:
            loss, p_sim, n_sim = SupConLoss_v2(temperature=args.temperature, base_temperature=args.temperature)(features=features)
        if args.contrastive_method in ['supcon', 'supcon_uni']:
            loss, p_sim, n_sim = SupConLoss_v2(temperature=args.temperature, base_temperature=args.temperature)(features=features, labels=targets, use_weight=args.use_weight, distance=args.distance)
        if args.contrastive_method in ['mix_neg', 'mix_pos', 'supremix']:
            # loss = SupConLoss_v2(temperature=args.temperature, base_temperature=args.temperature)(features=features, labels=targets, use_weight=args.use_weight, distance=args.distance)
            loss = None
            p_sim = None
            n_sim = None
            sub_batch_size=int(args.batch_size / args.pool)
            for proc_id in range(args.pool):
                start_n = proc_id * sub_batch_size
                end_n = (proc_id + 1) * sub_batch_size
                sub_target = targets[start_n:end_n,:].cpu().clone()
                sub_feature = features[start_n:end_n,:,:].cpu().clone()
                sub_feature_mixing = torch.cat([sub_feature[:,0,:], sub_feature[:,1,:]], dim=0)
                
                mixtures_rearranged, myop_mask_rearranged, myop_label_rearranged = mixing_func(sub_target, sub_feature_mixing, mix_neg=(args.contrastive_method in ['mix_neg', 'supremix']), mix_pos=(args.contrastive_method in ['mix_pos', 'supremix']), n=n, b=b)
                sub_loss, pos_sim, neg_sim = SupConLoss_v2(temperature=args.temperature, base_temperature=args.temperature)(features=sub_feature.cuda(), labels=sub_target.cuda(), mixtures=mixtures_rearranged.cuda(), myop_mask=myop_mask_rearranged.cuda(), use_weight=args.use_weight, myop_label=myop_label_rearranged.cuda(), distance=args.distance)

                if args.reduction == 'mean':
                    sub_loss = sub_loss / args.pool
                
                if loss == None:
                    loss = sub_loss
                    p_sim = pos_sim
                    n_sim = neg_sim
                else:
                    loss += sub_loss 
                    p_sim += pos_sim
                    n_sim += neg_sim

        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        losses.update(loss.item(), inputs.size(0))
        pos_sims.update(p_sim, inputs.size(0))
        neg_sims.update(n_sim, inputs.size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)


    return losses.avg, pos_sims.avg, neg_sims.avg

if __name__ == '__main__':
    main()