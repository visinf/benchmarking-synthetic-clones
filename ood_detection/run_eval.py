import os
import sys
from pathlib import Path

path_root = Path(__file__).resolve().parent.parent
sys.path.append(str(path_root))

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# Import metrics to compute
from torch import nn
from torch.utils.data import DataLoader

from datasets.load_datasets import load_data
from model import load_model
from utils import report_json, seed_everything
from ood_detection.ood_utils import get_ood_scores, get_measures
from ood_detection.datasets.pet37 import OxfordIIITPet
from ood_detection.datasets.food101 import Food101
from torchvision import transforms
import torchvision
from utils import obtain_ImageNet_classes, ClassificationModel, get_text_features


hydra.output_subdir="null"
from omegaconf import DictConfig, OmegaConf


def get_transform(clip=False):
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    INCEPTION_MEAN = (0.5, 0.5, 0.5)
    INCEPTION_STD = (0.5, 0.5, 0.5)
    
    mean = IMAGENET_MEAN if not clip else OPENAI_DATASET_MEAN
    std = IMAGENET_STD if not clip else OPENAI_DATASET_STD
   
    transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform



def set_ood_loader_ImageNet(out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)  
    elif out_dataset == 'placesbg': 
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)  
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join('/fastdata/ksingh/robust/transfer_datasets/', 'dtd', 'images'),
                                        transform=preprocess)
    return testsetout



def obtain_ImageNet_classes():
    loc = os.path.join('./data', 'imagenet')
    with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls


image_size = 224


# OmegaConf.register_new_resolver("bin_size", lambda x: 20)
@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(config):
    aurocs, auprs, fprs = [], [], []
    _, in_test_dset = load_data(config.exp.in_dataset, is_clip=config.exp.model in ['CLIP', 'scaling_clip', 'synthCLIP'])
    ood_transform  = get_transform(config.exp.model in ['CLIP', 'scaling_clip', 'synthCLIP'])
    ood_test_data = set_ood_loader_ImageNet(config.exp.ood_dataset, preprocess=ood_transform, root='/fastdata/ksingh/robust/OOD')
    
    in_test_loader = DataLoader(
        in_test_dset,
        batch_size=config.exp.batch_size,
        shuffle=False,
        num_workers=config.exp.n_workers,
        pin_memory=False,
    )
    ood_test_loader = DataLoader(
        ood_test_data,
        batch_size=config.exp.batch_size,
        shuffle=False,
        num_workers=config.exp.n_workers,
        pin_memory=False,
    )
    test_labels = obtain_ImageNet_classes()
    # ood_test_labels = ood_test_loader.dataset.class_names_str

    if config.exp.ablation:
        print('Running ablation...')
        pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone, prompt_type=config.data_params.prompt_type, size=config.data_params.num_data_points)
    else:
        pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone)

    # pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device)

    if config.exp.model in ['synthCLIP', 'CLIP', 'scaling_clip']:
        if config.exp.in_dataset == 'imagenet':
            class_labels = obtain_ImageNet_classes()

        text_features = get_text_features(pretrained_model, dataset_name=config.exp.in_dataset, class_labels=class_labels)
        pretrained_model= ClassificationModel(model=pretrained_model,text_embedding=text_features)
    seed_everything(config.exp.seed)

    in_score = get_ood_scores(config, pretrained_model, in_test_loader, test_labels=test_labels, in_dist=True)
    out_score = get_ood_scores(config, pretrained_model, ood_test_loader, test_labels=test_labels, in_dist=False)
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    result_dict = {
        'model': config.exp.model,
        'in_dist': config.exp.in_dataset,
        'ood_dist': config.exp.ood_dataset,
        'auroc': auroc,
        'aupr': aupr,
        'fpr': fpr,
        'score': config.exp.score
    } 
    if config.exp.ablation:
        config.exp.save_path = f'{config.exp.save_path}/ablations/'
        result_dict.update({'prompt_type': config.data_params.prompt_type, 'dataset_size': config.data_params.num_data_points})
        fn=f'{config.exp.model}_{config.exp.in_dataset}_{config.exp.ood_dataset}_{config.data_params.prompt_type}_{config.data_params.num_data_points}.json'
        report_json(result_dict, config.exp.save_path, fn)
    else:
        report_json(result_dict, config.exp.save_path, fn=f'{config.exp.model}_{config.exp.in_dataset}_{config.exp.ood_dataset}.json')


if __name__ == "__main__":
    main()