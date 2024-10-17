<font size=5>**Instructions of ReferFormer Configs**</font>

```
bash scripts/train_actionvos.sh actionvos_dirs/r101 pretrained_weights/r101_refytvos_joint.pth 1 0 29500 --backbone resnet101 --expression_file train_meta_expressions_promptaction.json --use_weights --actionvos_path ../dataset_visor --epochs 6 --lr_drop 3 5 --save_interval 3
```

where **actionvos_dirs/r101** is the path for saving training logs and checkpoints.

**pretrained_weights/r101_refytvos_joint.pth** is the base RVOS model path.

**1** is number of gpus.
**0** is visible gpus. E.g., use **4 0,1,2,3** when using 4 gpus.

**29500** is the port index. If you are running parallel scripts, change this index to any other number.

For all other parameters, check [opts.py](opts.py) for explanations.
