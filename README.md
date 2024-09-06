# Is Synthetic Data all We Need? Benchmarking the Robustness of Models Trained with Synthetic Images
CVPR 2024 Workshop SyntaGen: Harnessing Generative Models for Synthetic Visual Dataset

# Dependencies
The repo depends upon the following

Python 3.8.5

PyTorch 2.2.1

CUDA 12.1

# Environment
conda create -n bench-syn-clone python==3.8.5

conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt

# Model Files
model.py files loads the models files. 
You can add your model here in oder to evalute it. 

# Running evaluation on your own models
For evaluating your own put in the folder pretrained_models/Pretrained_Models folder and add edit the model.py file.

For each evaluation metric please change the dataset_path and save_path in the config/default.yaml file for each metric.


### Calibration
Please donwload the [ImageNet-A](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar) and [ImageNet-R ](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar) datasets.
```
cd calibration 
bash scripts/run_multiple.sh
bash scripts/run_clip.sh
```

###  Background Bias
Please download the ImageNet-9 dataset from (https://github.com/MadryLab/backgrounds_challenge)[https://github.com/MadryLab/backgrounds_challenge]
```
cd background_bias
bash scripts/run_multiple.sh
bash scripts/run_clip.sh
```

### Shape Bias
```
cd shape_bias
bash scripts/run_multiple.sh
bash scripts/run_clip.sh
```
### Context Bias
Please download the [FOCUS](https://umd.box.com/s/w7tvxer0wur7vtsoqcemfopgshn6zklv) dataset and then run, 
```
cd context_bias
bash scripts/run_mutliple.sh
```
### OOD Detection
Download iNaturalist, Places, and SUN dataset.
```
cd ood_detection
bash scripts/run_multiple.sh
bash scripts/run_clip.sh
```
### 2D corruptions
Download the 2D-corruptions dataset from [https://zenodo.org/records/2235448](https://zenodo.org/records/2235448)
```
cd corruptions
bash scripts/run_multiple_2dcc.sh
bash scripts/run_clip_2dcc.sh
```
### 3D corruptions
Download the 3D-corruptions dataset from [https://datasets.epfl.ch/3dcc/index.html](https://datasets.epfl.ch/3dcc/index.html)
```
cd corruptions
bash scripts/run_multiple_3dcc.sh
bash scripts/run_clip_3dcc.sh
```
### Adversarial Pertubations
We use the Ares package for running our attacks. 
```
cd adversarial_attack/ares/classification
bash scripts/run_multiple_fgsm.sh
bash scripts/run_multiple_pgd.sh
bash scripts/run_clip_fgsm.sh
bash scripts/run_clip_pdg.sh
```

## License
Apache2 license.

## Contact
Krishnakant Singh (firstname.lastname@visinf.tu-darmstadt.de)  
