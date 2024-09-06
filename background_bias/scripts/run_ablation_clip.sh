#!/bin/bash
base_str="python /visinf/home/ksingh/syn-rep-learn/background_bias/backgound_eval.py"
mname="scaling_clip"
seed=42
# partition_type=("REAL" "Syn_Real" "Synthetic")
partition_type=("Syn_Real" "Synthetic")

dataset_size=("64M" "128M" "256M" "371M")
dname="imagenet"
ablation="True"
for ptype in ${partition_type[@]} 
do
    for dsize in ${dataset_size[@]} 
    do
        echo "$dname"
        echo $ptype
        $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.batch_size=96 ++exp.backbone="False" ++data_params.num_data_points=$dsize ++data_params.prompt_type=$ptype ++exp.ablation=$ablation ++exp.backbone="True"
    done
done