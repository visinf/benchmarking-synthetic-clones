#!/bin/bash
base_str="python ./eval_calibration.py"
modelNames=("resnet50" "DeiT" "SwimT" "ConvNext" "syn_clone" "scaling_imagenet_sup" "syn_clone" "vit-b" "dino" "mae" "mocov3" "synclr")
seeds=(42)
dataset=("imagenet-r" "imagenet-a")
# dataset=("imagenet")
for seed in ${seeds[@]} 
do
    for mname in ${modelNames[@]} 
    do
        for dname in ${dataset[@]} 
        do
            echo "$dname"
            $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.batch_size=96 ++exp.backbone="False"
        done
    done
done