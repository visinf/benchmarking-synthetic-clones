#!/bin/bash
base_str="python /visinf/home/ksingh/syn-rep-learn/ood_detection/run_eval.py"
# modelNames=("CLIP" "resnet50" "DeiT" "SwimT" "ConvNext" "syn_clone" "scaling_imagenet_sup")

mname="scaling_clip"
seed=42
partition_type=("REAL" "Syn_Real" "Synthetic")
dataset_size=("64M" "128M" "256M" "371M")
in_dataset="imagenet"
od="SUN"

ablation="True"
for ptype in ${partition_type[@]} 
do
    for dsize in ${dataset_size[@]} 
    do
        echo "$dname"
        $base_str ++exp.model=$mname ++exp.in_dataset=$in_dataset ++exp.ood_dataset=$od ++exp.seed=$seed ++exp.batch_size=96 ++exp.backbone="False" ++data_params.num_data_points=$dsize ++data_params.prompt_type=$ptype ++exp.ablation=$ablation ++exp.backbone="True"
    done
done