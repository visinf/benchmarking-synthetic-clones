#!/bin/bash
base_str="python /visinf/home/ksingh/syn-rep-learn/ood_detection/run_eval.py"
mname="scaling_imagenet_sup"
seed=42
prompt_type=("captions" "classname" "CLIP_templates")
num_data_points=("1M" "4M" "8M" "16M")
in_dataset="imagenet"
od="SUN"
ablation="True"
for ptype in ${prompt_type[@]} 
do
    for dsize in ${num_data_points[@]} 
    do
        echo "$dname"
        $base_str ++exp.model=$mname ++exp.in_dataset=$in_dataset ++exp.ood_dataset=$od ++exp.seed=$seed ++exp.batch_size=96 ++exp.backbone="False" ++data_params.num_data_points=$dsize ++data_params.prompt_type=$ptype ++exp.ablation=$ablation ++exp.backbone="False"
    done
done