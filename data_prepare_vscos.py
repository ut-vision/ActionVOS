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
parser.add_argument('--VSCOS_PATH', type=str, required=True)
args = parser.parse_args()
VSCOS_PATH = args.VSCOS_PATH

SAVE_PATH = 'dataset_vscos'
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
    list_of_dict = []
    all_sparse_frames = 0
    seq_id = 0
    # read VSCOS
    with open(os.path.join(VSCOS_PATH,'EPIC_{}_split'.format(split),'EPIC100_state_positive_{}.yaml'.format(split)), 'r') as f:
        VSCOS_actions = yaml.safe_load(f)
    for video in tqdm(VSCOS_actions.keys()):
        P = VSCOS_actions[video]['participant_id']
        vid = VSCOS_actions[video]['video_id']
        seq_id += 1
        src_mask_path = os.path.join(VSCOS_PATH,'EPIC_{}_split'.format(split),P,'anno_masks',vid,video)
        src_img_path = src_mask_path.replace('anno_masks','rgb_frames')
        dst_mask_path = os.path.join(SAVE_PATH,'Annotations',split,'{:08d}_{}_{}'.format(seq_id,VSCOS_actions[video]['verb'],VSCOS_actions[video]['noun']))
        dst_img_path = dst_mask_path.replace('Annotations','JPEGImages')
        os.makedirs(dst_mask_path,exist_ok=True)
        os.makedirs(dst_img_path,exist_ok=True)
        for src_mask in os.listdir(src_mask_path):
            # copy masks
            mask = Image.open(os.path.join(src_mask_path,src_mask)).convert('P')
            mask.putpalette(palette)
            mask.save(os.path.join(dst_mask_path,src_mask))
            # copy images
            src_img = src_mask.replace('png','jpg')
            shutil.copy2(os.path.join(src_img_path,src_img),os.path.join(dst_img_path,src_img))    
        # save dict
        dict = {}
        dict['seq_id'] = seq_id
        dict['video'] = video
        dict['start'] = VSCOS_actions[video]['start_frame']
        dict['end'] = VSCOS_actions[video]['stop_frame']
        dict['narration'] = VSCOS_actions[video]['narration']
        dict['verb'] = VSCOS_actions[video]['verb']
        dict['verb_class'] = VSCOS_actions[video]['verb_class']
        dict['noun'] = VSCOS_actions[video]['noun']
        dict['noun_class'] = VSCOS_actions[video]['noun_class']
        dict['sparse_frames'] = sorted([m.replace('png','jpg') for m in os.listdir(src_mask_path)])
        dict['object_classes'] = {"1":{"name": dict['noun'], "class_id": dict['noun_class'], "handbox": 1, "narration": 1, "positive": 1}}
        list_of_dict.append(dict)
        all_sparse_frames += len(dict['sparse_frames'])

    # save json 
    json_object = json.dumps(list_of_dict) 
    with open(os.path.join(SAVE_PATH,'ImageSets',split+'_'+sampling+".json"), "w") as f:
        f.write(json_object)
    print('finished for {}_{} set. {:d} actions, {:d} sparse frames'.format(split,sampling,seq_id,all_sparse_frames))

def main():
    generate_action_with_mask('val')

if __name__ == '__main__':
    main()