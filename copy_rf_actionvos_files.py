# move modified files to referformer
file_list = [
    "datasets/__init__.py",
    "datasets/actionvos.py",
    "datasets/transforms_video_actionvos.py",
    "opts.py",
    "main_actionvos.py",
    "inference_actionvos.py",
    "scripts/train_actionvos.sh",
    "scripts/test_actionvos.sh",
    "models/referformer.py",
    "models/segmentation.py",
    "models/criterion.py",
]

import shutil
import os
import json
from tqdm import tqdm

def move_files(src_path='RF_ActionVOS',dst_path='ReferFormer'):
    for file in file_list:
        filename = file.split('/')[-1]
        src_file = os.path.join(src_path,filename)
        dst_file = os.path.join(dst_path,file)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.copy(src_file, dst_file)

move_files()

# generate objects_category.json for RF
# generate meta_expressions.json for RF
# if you want to change input expressions, 
# i.e., ablation study of language prompts, change meta_expressions.json files

def generate_meta_json(actionvos_path='dataset_visor',language_prompt='promptaction',split='train'):
    assert language_prompt in ['noaction','action','promptaction']
    # for meta_expressions.json
    datas = json.load(open(os.path.join(actionvos_path,'ImageSets',f'{split}.json')))
    meta_exp = {}
    meta_exp['videos'] = {}
    obj_cls = {}
    obj_cls['videos'] = {}
    for seq in tqdm(datas):
        exp_dict = {}
        exp_dict['expressions'] = {}
        obj_dict = {}
        obj_dict['objects'] = {}
        for i,k in enumerate(seq['object_classes'].keys()):
            exp_dict['expressions'][str(i)] = {}
            if language_prompt == 'noaction':
                exp_dict['expressions'][str(i)]['exp'] = seq['object_classes'][k]['name']
            elif language_prompt == 'action':
                exp_dict['expressions'][str(i)]['exp'] = seq['object_classes'][k]['name']+', '+seq['narration']
            elif language_prompt == 'promptaction':
                exp_dict['expressions'][str(i)]['exp'] = seq['object_classes'][k]['name']+' used in the action of '+seq['narration'] 
            else:
                pass
            exp_dict['expressions'][str(i)]['obj_id'] = k
            exp_dict['expressions'][str(i)]['positive'] = seq['object_classes'][k]['positive']
            exp_dict['expressions'][str(i)]['class_id'] = seq['object_classes'][k]['class_id']
            obj_dict['objects'][k]={'category':seq['object_classes'][k]['name'],'positive':seq['object_classes'][k]['positive'],'class_id':seq['object_classes'][k]['class_id']}
        exp_dict['frames'] = []
        for f in seq['sparse_frames']:
            exp_dict['frames'].append(f.replace('.jpg',''))
        folder_name = '{:08d}_{}_{}_{}'.format(seq['seq_id'],seq['video'],seq['verb'],seq['noun'])
        meta_exp['videos'][folder_name] = exp_dict
        obj_cls['videos'][folder_name] = obj_dict

    meta_exp_json_object = json.dumps(meta_exp)
    obj_cls_json_dist = os.path.join(actionvos_path,'ImageSets', f'{split}_meta_expressions_{language_prompt}.json')
    with open(obj_cls_json_dist, "w") as f:
        f.write(meta_exp_json_object)

    obj_cls_json_object = json.dumps(obj_cls)
    obj_cls_json_dist = os.path.join(actionvos_path,'ImageSets', f'{split}_objects_category.json')
    with open(obj_cls_json_dist, "w") as f:
        f.write(obj_cls_json_object)

generate_meta_json(split='train')
generate_meta_json(split='val')