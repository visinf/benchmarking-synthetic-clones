#!/bin/bash
base_str="python ./run_model_evaluator.py"
modelNames=("scaling_clip" "CLIP")
seed=42
dataset=("imagenet")
for dname in ${dataset[@]} 
    do
    for mname in ${modelNames[@]} 
        do 
            echo "$dname"
            $base_str ++exp.model=$mname ++exp.seed=$seed ++exp.backbone="True"
        done
    done