"""
actionvos data loader
Note that we adjust the transform file (for data augmentation)
# TODO check possible bug when box [0,0,0,0] goes to augmentation.
"""
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
import datasets.transforms_video_actionvos as T

import os
from PIL import Image
import json
import numpy as np
import random

#from datasets.categories import ytvos_category_dict as category_dict


class ActionVOSDataset(Dataset):
    """
    In this version, sampling every <video, caption, mask> triplet
    if the object is negative, the mask would be all zero
    """
    def __init__(self, actionvos_folder: Path, ann_file: Path, transforms, return_masks: bool, 
                 num_frames: int, max_skip: int, use_weights: bool, image_set: str):
        self.actionvos_folder = actionvos_folder     
        self.ann_file = ann_file         
        self._transforms = transforms    
        self.return_masks = return_masks # not used
        self.num_frames = num_frames     
        self.max_skip = max_skip
        self.use_weights = use_weights
        self.image_set = image_set
        # create video meta data
        self.prepare_metas()       

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.actionvos_folder),'ImageSets', f'{self.image_set}_objects_category.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(os.path.join(str(self.actionvos_folder),'ImageSets', self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = int(exp_dict['obj_id'])
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    # get object category and pos
                    obj_id = exp_dict['obj_id']
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    meta['class_id'] = exp_dict['class_id']
                    meta['positive'] = exp_dict['positive']
                    self.metas.append(meta)
        #self.metas = self.metas[:10]# for debug

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, obj_id, category, positive, frames, frame_id = \
                        meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['positive'], meta['frames'], meta['frame_id']
            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = meta['class_id']
            vid_len = len(frames)

            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            if self.num_frames != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)
    
                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >=global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            # read frames and masks and weights
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            positives = []
            weights = []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.actionvos_folder), 'JPEGImages_Sparse', self.image_set, video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.actionvos_folder), 'Annotations_Sparse', self.image_set, video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('P')
                if self.use_weights:
                    weight_path = os.path.join(str(self.actionvos_folder), 'Weights_Sparse', self.image_set, video, frame_name + '.png')
                    weight = Image.open(weight_path).convert('P')
                    weight = np.array(weight)

                # create the target
                label =  torch.tensor(category_id) 
                mask = np.array(mask)
                
                if self.use_weights:
                    # for where weight == 3, mask should be zero
                    # for where mask == 0, final weight should be 1
                    mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                    weight = np.where(mask == 0, 1, weight)
                    mask = np.where(weight == 3, 0, mask)
                    # we could map weight to other numbers
                    # by statistic of the training set, we set 5 to 3 and 4, 2 to hand obj
                    weight = np.where(weight>=3, 5, weight)
                    if positive:
                        if (mask > 0).any():
                            y1, y2, x1, x2 = self.bounding_box(mask)
                            box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                        else: # some frame didn't contain the instance
                            box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    else:
                        box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                        mask = np.zeros_like(mask)  
                else:
                    # check positive
                    if positive:
                        mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                        if (mask > 0).any():
                            y1, y2, x1, x2 = self.bounding_box(mask)
                            box = torch.tensor([x1, y1, x2, y2]).to(torch.float)                        
                        else: # some frame didn't contain the instance
                            box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    else:
                        box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                        mask = np.zeros_like(mask)
                    weight = np.ones_like(mask)
                mask = torch.from_numpy(mask)
                weight = torch.from_numpy(weight)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)
                valid.append(1)
                positives.append(positive)
                weights.append(weight)

            # transform
            w, h = img.size
            labels = torch.stack(labels, dim=0) 
            boxes = torch.stack(boxes, dim=0) 
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0)
            weights = torch.stack(weights, dim=0) 
            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'labels': labels,                        # [T,]
                'boxes': boxes,                          # [T, 4], xyxy
                'masks': masks,                          # [T, H, W]
                'weights': weights,                      # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]
                'positive': torch.tensor(positives),      # [T,]
                'caption': exp,
                'orig_size': torch.as_tensor([int(h), int(w)]), 
                'size': torch.as_tensor([int(h), int(w)])
            }

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
            
            # FIXME: handle "valid", since some box may be removed due to random crop
            # skip this. we sample all negative samples
            instance_check = True
            '''
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)
            '''
        return imgs, target


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        # edited by oyly, do not check, and transform weights
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    #T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    #T.Check(),
                ])
            ),
            normalize,
        ])
    
    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.actionvos_path)
    assert root.exists(), f'provided ActionVOS path {root} does not exist'
    ann_file = args.expression_file
    print('you are building actionvos {} set with {} , {}'.format(image_set,args.actionvos_path,ann_file))
    dataset = ActionVOSDataset(args.actionvos_path, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size), return_masks=args.masks, 
                           num_frames=args.num_frames, max_skip=args.max_skip, use_weights=args.use_weights, image_set=image_set)
    return dataset
