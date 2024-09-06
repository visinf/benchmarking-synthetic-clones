#!/bin/bash
if [ "$#" -eq 0 ]
then
    python /visinf/home/ksingh/syn-rep-learn/cls_evaluation/calibration/eval_calibration.py
else
    base_str = "/visinf/home/ksingh/syn-rep-learn/cls_evaluation/calibration/eval_calibration.py"
    for var in "$@";do
        base_str = $base_str+$var
    done
fi