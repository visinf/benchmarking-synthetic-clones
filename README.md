# Is Synthetic Data All We Need? Benchmarking the Robustness of Models Trained with Synthetic Images [CVPRW 2024]
CVPR 2024 Workshop SyntaGen: Harnessing Generative Models for Synthetic Visual Dataset

**[![arXiv](https://img.shields.io/badge/arXiv-2403.16292-b31b1b.svg)](https://arxiv.org/abs/2405.20469)** 

**[Project Page](https://synbenchmark.github.io/SynCloneBenchmark/)**

# 🔧  **Dependencies**
The repo depends upon the following

Python 3.8.5

PyTorch 2.2.1

CUDA 12.1

# 🚜 **Environment**
conda create -n bench-syn-clone python==3.8.5

conda activate bench-syn-clone

conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt

# 💻 **Model Files**
The pretrained models files that we used in our paper can be downloaded from [here](https://drive.google.com/file/d/1BYLwWGa6lPCGDXLPzH0H4dvW1FRMfQZV/view?usp=sharing).


# 🚀 **Running evaluation on your models**
To evaluate your own, put it in the pretrained_models/Pretrained_Models folder and edit the model.py file.

Please change the dataset_path and save_path in the config/default.yaml file for each evaluation metric.


### **Calibration**
Please donwload the [ImageNet-A](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar) and [ImageNet-R ](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar) datasets.
```
cd calibration 
bash scripts/run_multiple.sh
bash scripts/run_clip.sh
```

### **Background Bias**
Please download the [ImageNet-9](https://github.com/MadryLab/backgrounds_challenge) dataset.
```
cd background_bias
bash scripts/run_multiple.sh
bash scripts/run_clip.sh
```

### **Shape Bias**
```
cd shape_bias
bash scripts/run_multiple.sh
bash scripts/run_clip.sh
```
### **Context Bias**
Please download the [FOCUS](https://umd.box.com/s/w7tvxer0wur7vtsoqcemfopgshn6zklv) dataset and then run, 
```
cd context_bias
bash scripts/run_mutliple.sh
```
### **OOD Detection**
Download the iNaturalist, Places, and SUN datasets.
```
cd ood_detection
bash scripts/run_multiple.sh
bash scripts/run_clip.sh
```
### **2D corruptions**
Download the 2D-corruptions dataset from [https://zenodo.org/records/2235448](https://zenodo.org/records/2235448)
```
cd corruptions
bash scripts/run_multiple_2dcc.sh
bash scripts/run_clip_2dcc.sh
```
### **3D corruptions**
Download the 3D-corruptions dataset from [https://datasets.epfl.ch/3dcc/index.html](https://datasets.epfl.ch/3dcc/index.html)
```
cd corruptions
bash scripts/run_multiple_3dcc.sh
bash scripts/run_clip_3dcc.sh
```
### **Adversarial Perturbations**
We use the Ares package for running our attacks. 
```
cd adversarial_attack/ares/classification
bash scripts/run_multiple_fgsm.sh
bash scripts/run_multiple_pgd.sh
bash scripts/run_clip_fgsm.sh
bash scripts/run_clip_pdg.sh
```

**BibTex**
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@inproceedings{singh2024synthetic,
  title={Is Synthetic Data All We Need? Benchmarking the Robustness of Models Trained with Synthetic Images},
  author={Singh, Krishnakant and Navaratnam, Thanush and Holmer, Jannik and Schaub-Meyer, Simone and Roth, Stefan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2505--2515},
  year={2024}
}</code></pre>
  </div>
</section>


## **License**
Apache2 license.

## **Contact**
Krishnakant Singh (firstname.lastname@visinf.tu-darmstadt.de)  
