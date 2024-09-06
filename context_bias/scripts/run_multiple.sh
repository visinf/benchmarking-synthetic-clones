#!/bin/bash
base_str="python ./evaluate_model.py"
modelNames=("resnet50" "DeiT" "SwimT" "ConvNext" "syn_clone" "scaling_imagenet_sup" "syn_clone" "vit-b" "dino" "mae" "mocov3" "synclr")
dataset=("imagenet")
for mname in ${modelNames[@]} 
    do
        echo $mname
        $base_str ++exp.model=$mname ++exp.seed=$seed ++exp.batch_size=96 ++exp.backbone="False"
    done