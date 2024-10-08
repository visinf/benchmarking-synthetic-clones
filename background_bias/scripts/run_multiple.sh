#!/bin/bash
base_str="python ./backgound_eval.py"
modelNames=("resnet50" "DeiT" "SwimT" "ConvNext" "syn_clone" "scaling_imagenet_sup" "syn_clone" "vit-b" "dino" "mae" "mocov3" "synclr")
dataset=("imagenet")
seed=42
for mname in ${modelNames[@]} 
    do
        $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.batch_size=96 ++exp.backbone="False"
    done