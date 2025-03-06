import argparse
import os
import numpy as np
from PIL import Image
import cv2
import csv
import json
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
# from mmcv.runner import load_checkpoint




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

    # print('pred : ',pred.shape, np.unique(pred))
    # print('target : ',target.shape, np.unique(target))

    pred = convert_to_binary(pred)
    target = convert_to_binary(target)

    # print('pred : ',pred.shape, np.unique(pred))
    # print('target : ',target.shape, np.unique(target))

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)  # Adding epsilon to avoid division by zero
    iou = round(iou, 3)
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
    """
    Calculate Dice coefficient for binary masks.
    
    Args:
    - pred (numpy.ndarray): Predicted binary mask (0 or 1).
    - target (numpy.ndarray): Target binary mask (0 or 1).
    
    Returns:
    - float: Dice coefficient.
    """
    # print('Inside Dice function:')

    # print('pred : ',pred.shape, np.unique(pred))
    # print('target : ',target.shape, np.unique(target))

    pred = convert_to_binary(pred)
    target = convert_to_binary(target)

    # print('pred : ',pred.shape, np.unique(pred))
    # print('target : ',target.shape, np.unique(target))

    intersection = np.logical_and(pred, target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)  # Adding epsilon to avoid division by zero
    dice = round(dice, 3)
    return dice


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_scores_to_csv(image_names, iou_scores, dice_scores, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['Image', 'IoU', 'Dice'])
        # Write scores row by row
        for image_name, iou, dice in zip(image_names, iou_scores, dice_scores):
            writer.writerow([image_name, iou, dice])

# Function to resize the image
def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

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

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    iou_scores = []
    dice_scores = []
    image_names =[]

    # i = 0
    # save_folder = 'outputs'
    # os.makedirs(save_folder, exist_ok=True)

    i=0
    for batch in pbar:
        # print()
        # print('## Batch items: ', batch)
        # print('$$ Batch no: ',i)
        image_name = batch[1][0]  # This is a tuple of image names
        # print('Image: ', image_name)
        batch=batch[0]
        # i += 1
        # if i==10:
        #     break

        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        output = model.infer(inp)
        # print('output unique values:', torch.unique(output))
        pred = torch.sigmoid(output)
        # print('Pred unique values:', torch.unique(pred))
        binary_image = torch.where(pred < 0.5, torch.tensor(0, dtype=torch.bool, device=pred.device), torch.tensor(1, dtype=torch.bool, device=pred.device))
        # binary_image = (pred > 0.5).float()
        # print('Binary unique values:', torch.max(binary_image))
        gt_binary = (batch['gt'] > 0.5)

        gt_binary = (gt_binary.cpu().numpy() * 255).astype(np.uint8)
        binary_np = (binary_image.cpu().numpy() * 255).astype(np.uint8)
        binary_np = binary_np.squeeze()
        gt_binary = gt_binary.squeeze()

        # Save the numpy array as PNG using cv2
        folder = '/home/scai/mtech/aib232071/scratch/project/SAM-Adapter-PyTorch/test_output_ext_new'
        os.makedirs(folder, exist_ok=True)

        # images_path = os.path.join(folder, 'images')
        true_mask_path = os.path.join(folder, 'true_mask')
        pred_mask_path = os.path.join(folder, 'pred_mask')

        # os.makedirs(images_path, exist_ok=True)
        os.makedirs(true_mask_path, exist_ok=True)
        os.makedirs(pred_mask_path, exist_ok=True)
        # print('file path: ',true_mask_path)

        # image_name = os.path.join(images_path, f'image_{i}.jpg')
        # true_mask_name = os.path.join(true_mask_path, f'image_{i}.jpg')
        # pred_mask_name = os.path.join(pred_mask_path, f'image_{i}.jpg')
        

        # true_mask_name = os.path.join(true_mask_path, f'image_{i:05d}.jpg')
        # pred_mask_name = os.path.join(pred_mask_path, f'image_{i:05d}.jpg')

        true_mask_name = os.path.join(true_mask_path, image_name)
        pred_mask_name = os.path.join(pred_mask_path, image_name)


        # cv2.imwrite(image_name, inp_np)
        cv2.imwrite(true_mask_name, gt_binary)
        cv2.imwrite(pred_mask_name, binary_np)
        # break

        # Calculate IoU and Dice
        iou = calc_iou(binary_np, gt_binary)
        dice = calc_dice(binary_np, gt_binary)

        # Debug: Print IoU and Dice for current batch
        print(f'Batch {i}:image_name: {image_name},  IoU: {iou}, Dice: {dice}')

        iou_scores.append(iou)
        dice_scores.append(dice)
        image_names.append(image_name)

        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))
        

    
    # print('iou_scores: ', iou_scores)
    # print('dice_scores: ', dice_scores)

    save_scores_to_csv(image_names, iou_scores, dice_scores, 'test_scores_ext.csv')

    # file_iou_scores = 'iou_scores.csv'
    # with open(file_iou_scores, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(iou_scores)

    # file_dice_scores = 'dice_scores.csv'
    # with open(file_dice_scores, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(dice_scores)

    mean_iou = sum(iou_scores) / len(iou_scores)
    mean_dice = sum(dice_scores) / len(dice_scores)

    # print('Mean IoU:', mean_iou)
    # print('Mean Dice:', mean_dice)
    # for batch in pbar:
    #     print(i)
    #     i+=1
    #     for k, v in batch.items():
    #         batch[k] = v.cuda()

    #     inp = batch['inp']
        

    #     pred = torch.sigmoid(model.infer(inp))
    #     # print(pred)
    #     # print('Pred: ',torch.unique(pred), torch.max(pred),torch.min(pred))
    #     # binary_image = (pred > 0.5).int()
    #     # print('Binary: ',torch.unique(binary_image), torch.max(binary_image),torch.min(binary_image))
    #     # image_path = os.path.join(save_folder, f'binary_image_{i}.jpg')
    #     # image = (binary_image * 255).astype(np.uint8)  # Scale to 255 for proper visualization
    #     # # Save the image
    #     # cv2.imwrite(image_path, image)

    #     iou = calc_iou(pred > 0.5, batch['gt'] > 0.5)
    #     dice = calc_dice(pred > 0.5, batch['gt'] > 0.5)
    #     print(iou, dice)
    #     iou_scores.append(iou)
    #     dice_scores.append(dice)

    #     result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
    #     val_metric1.add(result1.item(), inp.shape[0])
    #     val_metric2.add(result2.item(), inp.shape[0])
    #     val_metric3.add(result3.item(), inp.shape[0])
    #     val_metric4.add(result4.item(), inp.shape[0])

    #     if verbose:
    #         pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
    #         pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
    #         pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
    #         pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    # print('iou_scores: ',iou_scores)
    # print('dice_scores: ',dice_scores)
    # mean_iou = sum(iou_scores) / len(iou_scores)
    # mean_dice = sum(dice_scores) / len(dice_scores)

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), mean_iou, mean_dice


class CustomDataset():
    def __init__(self, dataset, image_names):
        self.dataset = dataset
        self.image_names = image_names  # List of image names loaded from the JSON file

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]
        image_name = self.image_names[idx]
        return image, image_name



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()
    image_names_json='/home/scai/mtech/aib232071/scratch/project/test_data/ext_test_gbc_file_names.json'
    with open(image_names_json, 'r') as f:
        image_names = json.load(f)


    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    print('dataset size: ',len(dataset))

    custom_dataset = CustomDataset(dataset, image_names)

    # loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8)
    # print(f"Number of batches in the DataLoader: {len(loader)}")

    loader = DataLoader(custom_dataset, batch_size=spec['batch_size'], num_workers=8)

    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    
    metric1, metric2, metric3, metric4,  mean_iou, mean_dice = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True)
    # print('metric1: {:.4f}'.format(metric1))
    # print('metric2: {:.4f}'.format(metric2))
    # print('metric3: {:.4f}'.format(metric3))
    # print('metric4: {:.4f}'.format(metric4))
    print('Mean iou: {:.4f}'.format(mean_iou))
    print('Mean dice: {:.4f}'.format(mean_dice))
