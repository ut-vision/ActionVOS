# change VOST and VSCOS to ActionVOS(RVOS) settings
# val ~ 
import numpy as np
import pandas as pd
import yaml
import json
import os
import functools
import cv2
import shutil
from tqdm import tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--VOST_PATH', type=str, required=True)
args = parser.parse_args()
VOST_PATH = args.VOST_PATH

SAVE_PATH = 'dataset_vost'
os.makedirs(SAVE_PATH,exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH,'JPEGImages'),exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH,'JPEGImages','train'),exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH,'JPEGImages','val'),exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH,'ImageSets'),exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH,'Annotations'),exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH,'Annotations','train'),exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH,'Annotations','val'),exist_ok=True)

palette = Image.open('annotations/00000.png').getpalette()

def copy_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    files = os.listdir(src_folder)
    for file_name in files:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copy2(src_path, dest_path)

def generate_action_with_mask(split='train',sampling='all'):
    '''
    return list of dicted RGB frames and annotation masks and weights
    [
        {
        'seq_id': int to identify sequence id,
        'src': str to identify video source
        'video': original video name from VOST/VSCOS
        'verb''verb_class''noun''noun_class': verb noun annotation from EK-100, -1 for not available
        'narration': narration from VOST/VSCOS,
        'start': start frame index from EK-100,
        'end': end frame index from EK-100,this 2 fields should be zero for VOST
        'object_classes': {1:{'name':xxx,'class_id':xxx,'positive':0/1,'hand_box':0/1,'narration_obj':0/1}, } for mask object labels in annotations,
        'sparse_frames': ['xxx.jpg','yyy.jpg'] for the frames indexs
        }, ...
    ]
    '''
    assert split in ['train','val']
    assert sampling in ['all']
    # read VOST
    with open(os.path.join(VOST_PATH,'ImageSets',split+'.txt')) as f:
        VOST_actions = f.readlines()
        VOST_actions = [a.strip() for a in VOST_actions]
    
    list_of_dict = []
    all_sparse_frames = 0
    seq_id = 0
    for video in tqdm(VOST_actions):
        seq_id += 1
        # copy images
        images_path = os.path.join(VOST_PATH,'JPEGImages',video)
        copy_files(images_path,os.path.join(SAVE_PATH,'JPEGImages',split,'{:08d}_{}'.format(seq_id,'_'.join(video.split('_')[1:]))))
        # copy masks, make ambigious to zeros, make instances as one
        src_mask_path = os.path.join(VOST_PATH,'Annotations_raw',video)
        dst_mask_path = os.path.join(SAVE_PATH,'Annotations',split,'{:08d}_{}'.format(seq_id,'_'.join(video.split('_')[1:])))
        src_masks = os.listdir(src_mask_path)
        os.makedirs(dst_mask_path,exist_ok=True)
        for src_mask in src_masks:
            mask = Image.open(os.path.join(src_mask_path,src_mask))
            mask_np = np.array(mask)
            mask_np[mask_np==255] = 0
            mask_np[mask_np>0] = 1
            mask_save = Image.fromarray(mask_np,mode='P')
            mask_save.putpalette(palette)
            mask_save.save(os.path.join(dst_mask_path,src_mask))
        # save dict
        dict = {}
        dict['seq_id'] = seq_id
        dict['video'] = video
        dict['start'] = 0
        dict['end'] = 0
        dict['narration'] = ' '.join(video.split('_')[1:])
        dict['verb'] = video.split('_')[1]
        dict['verb_class'] = -1
        dict['noun'] = ' '.join(video.split('_')[2:])
        dict['noun_class'] = -1
        dict['sparse_frames'] = sorted(os.listdir(images_path))
        dict['object_classes'] = {"1":{"name": dict['noun'], "class_id": -1, "handbox": 1, "narration": 1, "positive": 1}}
        list_of_dict.append(dict)
        all_sparse_frames += len(dict['sparse_frames'])
        #if seq_id>=2:
            #break

    # save json 
    json_object = json.dumps(list_of_dict) 
    with open(os.path.join(SAVE_PATH,'ImageSets',split+'_'+sampling+".json"), "w") as f:
        f.write(json_object) 
    print('finished for {}_{} set. {:d} actions, {:d} sparse frames'.format(split,sampling,seq_id,all_sparse_frames))

def main():
    generate_action_with_mask('val')

if __name__ == '__main__':
    main()