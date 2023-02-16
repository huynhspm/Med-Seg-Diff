import numpy as np
import pandas as pd
import cv2
import torch
import shutil
import blobfile as bf
from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchmetrics import JaccardIndex, Dice


def visualize(img):
    '''
    Normalize for visualization.
    '''
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def metric(path, jaccard_threshold):
    gt_path = "/home/vinhpt/MedSegDiff/test/gt/" + path + '.jpg'
    pred_path = "/home/vinhpt/MedSegDiff/test/pred/" + path + "_output_ens.jpg"
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)
    gt = visualize(gt)
    # pred = visualize(pred)
    # jaccard = JaccardIndex(task="binary", ignore_index=0, threshold=jaccard_threshold)
    dice = Dice(average='micro')
    return dice(torch.tensor(gt, dtype=torch.int8), torch.tensor(pred, dtype=torch.int8))
    # print(jaccard(torch.tensor([[1, 0], [0, 1]], dtype=torch.int8), torch.tensor([[1, 1], [1, 0]], dtype=torch.int8)))
    # print(torch.tensor(mask, dtype=torch.int8))
    # print(torch.tensor(difference, dtype=torch.int8))

if __name__ == "__main__":
    jaccard_threshold = 0.15

    # files = pd.read_csv("/home/vinhpt/MedSegDiff/scripts/test.csv", header=None)
    files = pd.read_csv("/home/vinhpt/MedSegDiff/scripts/test3.csv", header=None)
    output = []
    # gts = []
    # preds = []
    for i in range(files.__len__()):
        file = files.loc[i][0].split('/')[-1].replace('.npy', '')
        # print(file)
        tmp = metric(file, jaccard_threshold)
        output.append(tmp)
        # gt, pred = get_mask_and_difference(files.loc[i][0].split('/')[-1])
        # gts.append(gt)
        # preds.append(pred)
    # output = iou2(torch.tensor(gts, dtype=torch.int8), torch.tensor(preds, dtype=torch.int8), n_classes=2)
    print(len(output))
    
    sum = 0
    
    cnt = 0
    tmp = 0
    for o in output:
        if not torch.isnan(o):
            sum = sum + o
            cnt = cnt + 1
        else: 
            tmp = tmp + 1 # full black
    print(float(sum) / float(cnt))
    print(tmp)