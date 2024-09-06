#!/bin/bash
base_str="python ./run_eval.py"
# modelNames=("CLIP" "resnet50" "DeiT" "SwimT" "ConvNext" "syn_clone" "scaling_imagenet_sup")
modelNames=("CLIP")
in_dataset="imagenet"
ood_dataset=("iNaturalist" "SUN" "places365")
seed=42
for mname in ${modelNames[@]} 
    do
        for od in ${ood_dataset[@]}
            do
                echo $od
                $base_str ++exp.model=$mname ++exp.in_dataset=$in_dataset ++exp.ood_dataset=$od ++exp.seed=$seed ++exp.batch_size=96 ++exp.backbone="True"
            done
    done