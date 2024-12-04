# evaluation metrics for actionvos, modified from GRES (https://github.com/henghuiding/ReLA)
# cal ious and acc between positive/negative objs in 2 folders
# every obj in each frame is treated as an obj

# NOTE: our annotators found some errors in original 330 videos val_human split
# which was used for the papers' results.
# These videos are marked as 
# 'missing == True' or 'redundant == True' or 'other == True' (see val_human.json).
# in this file, we filter these videos
# and use 294 videos as val_human for evaluation. 
# This only change metrics within 1% 
# and would not influence any conclusions in our paper.

# NOTE: due to different random seeds in training
# ~1% difference comparing to our papers' tables is normal.
# if you found your reproduced results are far from ours,
# please contact me by email oyly(at)iis.u-tokyo.ac.jp.

import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, required=True)
parser.add_argument('--gt_path', type=str, required=True)
parser.add_argument('--split_json', type=str, required=True)

def cal_gres_all(gt_path,pred_path,split_json,filter=True):
    meta = json.load(open(split_json))
    mious_pos = []
    mious_neg = []
    cis_pos = []
    cus_pos = []
    cis_neg = []
    cus_neg = []
    gious = []
    TN,TP,FN,FP = 0,0,0,0
    N_Video = 0
    for seq in tqdm(meta):
        if filter: # skip some errored seq
            if seq['redundant'] or seq['other'] or seq['missing']:
                continue
        N_Video += 1
        folder_name = '{:08d}_{}_{}_{}'.format(seq['seq_id'],seq['video'],seq['verb'],seq['noun'])
        for frame in seq['sparse_frames']:
            mask_gt_path = os.path.join(gt_path,folder_name,frame.replace('jpg','png'))
            mask_pred_path = os.path.join(pred_path,folder_name,frame.replace('jpg','png'))
            gt = np.array(Image.open(mask_gt_path))
            pred = np.array(Image.open(mask_pred_path))
            for k in seq['object_classes'].keys():
                # for positive obj
                if seq['object_classes'][k]['positive']:                 
                    color = int(k)
                    p = np.where(pred==color,1,0).astype(float)
                    g = np.where(gt==color,1,0).astype(float)
                    intersection = (p*g).sum()
                    union = (p+g).sum()-intersection
                    if union <= 0:
                        # no-target and no detected
                        # NOTE: for union = 0 samples, we do not count for mIoUs
                        # but count them for gIoU and accs.
                        gious.append(1)
                        TN += 1 
                    else:
                        mious_pos.append(intersection/union)
                        cis_pos.append(intersection)
                        cus_pos.append(union)
                        #some cases, pos obj may not visible in this frame
                        if g.sum() <= 0:
                            # no-target but detected
                            FP += 1
                            gious.append(0)
                        else:
                            gious.append(intersection/union)
                            if p.sum() <= 0:
                                # has target but not detected
                                FN += 1
                            else:
                                # has target and detected
                                TP += 1
                else:
                    color = int(k)
                    p = np.where(pred==color,1,0).astype(float)
                    g = np.where(gt==color,1,0).astype(float)
                    intersection = (p*g).sum()
                    union = (p+g).sum()-intersection
                    if union <= 0:
                        # no-target and no detected
                        gious.append(1)
                        TN += 1
                    else:
                        mious_neg.append(intersection/union)
                        cis_neg.append(intersection)
                        cus_neg.append(union)
                        #some cases, neg obj may not visible in this frame
                        if g.sum() <= 0:
                            # no-target but detected
                            FP += 1
                            gious.append(0)
                        else:
                            #gious.append(intersection/union)
                            if p.sum() <= 0:
                                # neg target and no detected
                                gious.append(1)
                                TN += 1
                            else:
                                # neg target and detected
                                FP += 1
                                gious.append(0)
    print(f'----- evaluation on {N_Video} videos -----')
    print('pos-mIoU:  ', sum(mious_pos)/len(mious_pos))
    print('neg-mIoU:  ', sum(mious_neg)/len(mious_neg))
    print('pos-cIoU:  ', sum(cis_pos)/sum(cus_pos))
    print('neg-cIoU:  ', sum(cis_neg)/sum(cus_neg))
    print('gIoU:      ', sum(gious)/len(gious))
    print('acc:       ',(TN+TP)/(TN+TP+FN+FP))
    print('TN,TP,FN,FP',TN,TP,FN,FP)
    print(f'------------------------------------')

def main():
    args = parser.parse_args()
    cal_gres_all(args.gt_path,args.pred_path,args.split_json,True)

if __name__ == '__main__':
    main()