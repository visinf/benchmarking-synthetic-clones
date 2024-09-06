#!/bin/bash
base_str="python ./run_attack.py"
mname="scaling_clip"
seed=42
partition_type=("REAL" "Syn_Real" "Synthetic")
dataset_size=("64M" "128M" "256M" "371M")
dname="imagenet"
ablation="True"
for ptype in ${partition_type[@]} 
do
    for dsize in ${dataset_size[@]} 
    do
        echo "$dname"
        $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.batch_size=32 ++exp.backbone="False" ++data_params.num_data_points=$dsize ++data_params.prompt_type=$ptype ++exp.ablation=$ablation ++exp.backbone="True" ++exp.attack_name="pgd"
    done
done