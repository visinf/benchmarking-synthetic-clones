#!/bin/bash
base_str="python /visinf/home/ksingh/syn-rep-learn/background_bias/backgound_eval.py"
mname="scaling_imagenet_sup"
dataset=("imagenet")
# prompt_type=("captions" "classname" "CLIP templates")
prompt_type=("CLIP_templates")

num_data_points=("1M" "4M" "8M" "16M")
dname="imagenet"
seed=42
for ptype in ${prompt_type[@]} 
do
    for dsize in ${num_data_points[@]} 
    do
        echo "$dname"
        echo "$ptype"
        echo "$dsize"
        $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.batch_size=96 ++exp.backbone="False" ++data_params.num_data_points=$dsize ++data_params.prompt_type=$ptype ++exp.ablation="True"
    done
done