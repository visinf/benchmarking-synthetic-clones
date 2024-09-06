#!/bin/bash
base_str="python /visinf/home/ksingh/syn-rep-learn/adversarial_attack/ares/classification/run_attack.py"
mname="scaling_imagenet_sup"
seed=42
prompt_type=("captions" "classname" "CLIP_templates")
# prompt_type=("CLIP_templates")
num_data_points=("1M" "4M" "8M" "16M")
dname="imagenet"
for ptype in ${prompt_type[@]} 
do
    for dsize in ${num_data_points[@]} 
    do
        echo "$dname"
        echo "$ptype"
        echo "$dsize"
        $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.batch_size=96 ++exp.backbone="False" ++data_params.num_data_points=$dsize ++data_params.prompt_type=$ptype ++exp.ablation="True" ++exp.attack_name="fgsm"
    done
done