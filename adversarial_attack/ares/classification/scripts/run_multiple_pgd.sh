#!/bin/bash
base_str="python ./run_attack.py"
# modelNames=("resnet50" "DeiT" "SwimT" "ConvNext" "syn_clone" "scaling_imagenet_sup", "MAE", "mocov3", "CLIP", "synclr")
modelNames=("scaling_imagenet_sup")
seeds=(42)
dataset=("imagenet")
for seed in ${seeds[@]} 
do
    for mname in ${modelNames[@]} 
    do
        for dname in ${dataset[@]} 
        do
            echo "$dname"
            $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.batch_size=32 ++exp.backbone="False" ++exp.attack_name="pgd"
        done
    done
done



