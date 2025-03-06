import argparse
import os
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import csv
import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
print('local_rank: ',local_rank)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def calc_iou(pred, target):
    """
    Calculate Intersection over Union (IoU) for binary masks.
    
    Args:
    - pred (numpy.ndarray): Predicted binary mask (0 or 1).
    - target (numpy.ndarray): Target binary mask (0 or 1).
    
    Returns:
    - float: IoU score.
    """

    # print('Inside IOU function:')
    # print(pred.shape, target.shape)
    target = (target.cpu().numpy()).astype(np.uint8)
    pred = (pred.cpu().numpy()).astype(np.uint8)
    pred = pred.squeeze()
    target = target.squeeze()

    # print('pred : ',pred.shape, np.unique(pred))
    # print('target : ',target.shape, np.unique(target))

    # pred = convert_to_binary(pred)
    # target = convert_to_binary(target)

    # print('pred : ',pred.shape, np.unique(pred))
    # print('target : ',target.shape, np.unique(target))


    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)  # Adding epsilon to avoid division by zero
    return iou

def convert_to_binary(mask):
    """
    Convert a mask with values 0 and 255 to binary values 0 and 1.
    
    Args:
    - mask (numpy.ndarray): Input mask with values 0 and 255.
    
    Returns:
    - numpy.ndarray: Binary mask with values 0 and 1.
    """
    binary_mask = np.where(mask == 255, 1, 0)
    return binary_mask


def calc_dice(pred, target):
    # print('hi')
    """
    Calculate Dice coefficient for binary masks.
    
    Args:
    - pred (numpy.ndarray): Predicted binary mask (0 or 1).
    - target (numpy.ndarray): Target binary mask (0 or 1).
    
    Returns:
    - float: Dice coefficient.
    """
    # print('Inside Dice function:')
    # print(pred.shape, target.shape)
    target = (target.cpu().numpy()).astype(np.uint8)
    pred = (pred.cpu().numpy()).astype(np.uint8)
    pred = pred.squeeze()
    target = target.squeeze()

    # print('pred : ',pred.shape, np.unique(pred))
    # print('target : ',target.shape, np.unique(target))

    # pred = convert_to_binary(pred)
    # target = convert_to_binary(target)

    # print('pred : ',pred.shape, np.unique(pred))
    # print('target : ',target.shape, np.unique(target))

    intersection = np.logical_and(pred, target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)  # Adding epsilon to avoid division by zero
    return dice



def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

# def calc_iou(pred, target):
#     # print('Pred unique values:', torch.unique(pred))
#     # print('target unique values:', torch.unique(target))
#     intersection = (pred & target).float().sum((1, 2))
#     union = (pred | target).float().sum((1, 2))
#     iou = (intersection + 1e-6) / (union + 1e-6)
#     return iou.mean().item()

# def calc_dice(pred, target):
#     intersection = (pred & target).float().sum((1, 2))
#     dice = (2. * intersection + 1e-6) / (pred.float().sum((1, 2)) + target.float().sum((1, 2)) + 1e-6)
#     return dice.mean().item()

def eval_psnr(loader, model, eval_type=None):
    print('Evaluate the model')
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    iou_scores = []
    dice_scores = []
    n=0

    for batch in loader:
        print(n)
        # if n==10:
        #     break
        n+=1
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        # print('Before sigmoid, Pred unique values:', torch.max(inp), torch.min(inp))
        pred = torch.sigmoid(model.infer(inp))
        # print('After sigmoid, Pred unique values:', torch.max(pred), torch.min(pred))
        # print('Target unique values:', torch.unique(batch['gt']))

        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)

        for p, g in zip(batch_pred, batch_gt):
            iou = calc_iou(p > 0.5, g > 0.5)
            dice = calc_dice(p > 0.5, g > 0.5)
            # iou = calc_iou(p, g)
            # dice = calc_dice(p, g)
            iou_scores.append(iou)
            dice_scores.append(dice)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)
    mean_iou = sum(iou_scores) / len(iou_scores)
    mean_dice = sum(dice_scores) / len(dice_scores)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4,  mean_iou, mean_dice

def prepare_training():
    epoch_start = 0  # Default value
    print('Prepare start')
    if config.get('resume') is not None:
        print('Prepare start A1')
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        checkpoint_path = config['model_latest_checkpoint']
        print('checkpoint_path: ',checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        print('checkpoint: ',checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Latest model loaded')
        print('Last updated epoch:', checkpoint['epoch'])
        epoch_start = checkpoint['epoch'] + 1  # Corrected variable name
        print('Start epoch:', epoch_start)
    else:
        print('Prepare start B1')
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 0  # Corrected variable name
    
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, epoch):
    print('Epoch No: ',epoch)
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    i = 0
    print('Train the model')
    for batch in train_loader:
        # print(f"Epoch: {epoch}, Iter: {i}")
        # if i==3:
        #     break
        i+=1
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        # print('Image shape: ', inp.shape)
        # print(f"Image: max{inp.max()}, min{inp.min()}")
        # print('Mask shape: ', gt.shape)
        # print(f"Mask: max{gt.max()}, min{gt.min()}")
        # break
        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)

def plot_graph(data, y_label, label_name, filename):
    plt.plot(data, label=label_name)
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(filename)

def main(config_, save_path, args):
    
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    # sam_checkpoint = torch.load(config['sam_checkpoint'])
    # model.load_state_dict(sam_checkpoint, strict=False)
    
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()

    train_loss = []
    mean_iou_scores = []
    mean_dice_scores = []

    for epoch in range(epoch_start, epoch_max):
        print('epoch: ',epoch)
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model, epoch)
        train_loss.append(train_loss_G)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')
            # save(config, model, optimizer, save_path, epoch_max, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4, mean_iou, mean_dice = eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))
            mean_iou_scores.append(mean_iou)
            mean_dice_scores.append(mean_dice)

            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)
                log_info.append('val: {}={:.4f}'.format('mean_iou', mean_iou))
                writer.add_scalars('mean_iou', {'val': mean_iou}, epoch)
                log_info.append('val: {}={:.4f}'.format('mean_dice', mean_dice))
                writer.add_scalars('mean_dice', {'val': mean_dice}, epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')
                        # save(config, model, optimizer, save_path, epoch_max, 'best')
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save(config, model, save_path, 'best')
                        # save(config, model, optimizer, save_path, epoch_max, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()
        # chekpoint_path = os.path.join(save_path, f"checkpoint.pth")
        checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }
        
        torch.save(checkpoint, os.path.join(save_path, f"checkpoint.pth"))
        print('Checkpoint saved for epoch: ',epoch)
        if epoch==epoch_max-1 or epoch%10==0:
            save(config, model, save_path, f'final_model_{epoch}')

        # train_loss_rows = [[loss] for loss in train_loss]
        # with open('train_loss.csv', 'w', newline='') as file:
        #     csv_writer = csv.writer(file)
        #     csv_writer.writerows(train_loss_rows)
    # plot_graph(train_loss, 'Loss', 'Train_Loss', 'train_loss_plot.png')
    # plot_graph(mean_iou_scores, 'Mean_IOU', 'Train_Mean_IOU', 'train_mean_iou.png')
    # plot_graph(mean_dice_scores, 'Mean_Dice', 'Train_Mean_Dice', 'train_mean_dice.png')


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


# def save(config, model, optimizer, save_path, epoch_max, name):
#     if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
#         if config['model']['args']['encoder_mode']['name'] == 'evp':
#             prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
#             decode_head = model.encoder.decode_head.state_dict()
#             torch.save({"prompt": prompt_generator, "decode_head": decode_head},
#                        os.path.join(save_path, f"prompt_epoch_{name}.pth"))
#         else:
#             torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
#     else:
#         checkpoint = {
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'epoch': epoch_max
#                      }
#     torch.save(checkpoint, os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/cod-sam-vit-b.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    # local_rank = int(os.environ['LOCAL_RANK'])
    # local_rank = int(os.environ.get('LOCAL_RANK', -1))  # Use environment variable

    args = parser.parse_args()
    # config = yaml.load(f, Loader=yaml.FullLoader)
    # print('config loaded.')

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
