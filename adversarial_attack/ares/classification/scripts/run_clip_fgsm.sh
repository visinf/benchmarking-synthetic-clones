#!/bin/bash
base_str="python ./run_attack.py"
modelNames=("scaling_clip" "CLIP")
seed=42
dataset=("imagenet")
for dname in ${dataset[@]} 
    do
    for mname in ${modelNames[@]} 
        do 
            echo "$dname"
            $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.backbone="True" ++exp.attack_name="fgsm"
        done
    done