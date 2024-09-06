#!/bin/bash
base_str="python ./eval_calibration.py"
modelNames=("scaling_clip")
seed=42
# dataset=("imagenet")
dataset=("imagenet" "imagenet-r" "imagenet-a")
for dname in ${dataset[@]} 
    do
    for mname in ${modelNames[@]} 
        do 
            echo "$dname"
            $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.backbone="True" ++exp.batch_size=92
        done
    done