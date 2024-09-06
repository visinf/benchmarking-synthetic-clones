#!/bin/bash
base_str="python ./2dcc.py"
modelNames=("CLIP" "scaling_clip")
seed=42
dname=("imagenet")
for mname in ${modelNames[@]} 
do 
    echo "$dname"
    echo "$mname"
    $base_str ++exp.model=$mname ++exp.dataset=$dname ++exp.seed=$seed ++exp.backbone="true" ++exp.ablation="False" ++exp.batch_size=32
done