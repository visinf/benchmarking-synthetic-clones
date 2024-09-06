import sys
import torch
from torchvision import transforms
import math
import argparse
import tqdm
from pathlib import Path
path_root = Path(__file__).resolve().parent.parent
sys.path.append(str(path_root))
path_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(path_root))

from ares.utils.registry import registry
from ares.utils.metrics import AverageMeter, accuracy
from classification.attack_configs import attack_configs
from datasets.load_datasets import load_data
from model import load_model
from torch.utils.data import DataLoader
import hydra
from transformers import CLIPTokenizer
import json
import open_clip
from open_clip import get_tokenizer
import os
import numpy as np
from utils import ClassificationModel, get_text_features, obtain_ImageNet_classes, report_json, seed_everything

from torchattacks import PGD, FGSM
torch.backends.cudnn.deterministic = True

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(config):
    is_clip=config.exp.model in ['CLIP', 'scaling_clip', 'synthCLIP']
    _, test_dset = load_data(config.exp.dataset, is_clip=is_clip)
    test_loader = DataLoader(
        test_dset,
        batch_size=config.exp.batch_size,
        shuffle=False,
        num_workers=config.exp.n_workers,
        pin_memory=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone)
    if config.exp.ablation:
        print('Running ablation...')
        pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone, prompt_type=config.data_params.prompt_type, size=config.data_params.num_data_points)
    else:
        pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone)


    model = pretrained_model.to(device)
    seed_everything(config.exp.seed)
    # set device
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    if config.exp.model in ['synthCLIP', 'CLIP', 'scaling_clip']:
        if config.exp.dataset == 'imagenet':
            class_labels = obtain_ImageNet_classes()
        else:
            class_labels = test_loader.dataset.class_names_str

        text_features = get_text_features(model, dataset_name=config.exp.dataset, class_labels=class_labels)
        model = ClassificationModel(model=model,text_embedding=text_features)
        
    # create model
    # initialize attacker
    result_dict = []
    # pbar = tqdm.tqdm(total=len(config.exp.attack_name))
    attack_name = config.exp.attack_name
    if attack_name == 'pgd':
        print('Using PGD attack')
        atk = PGD(model, eps=1/255, alpha=2/225, steps=20, random_start=True)
    elif attack_name == 'fgsm':
        atk = FGSM(model, eps=1/255)
    
    if not is_clip:
        atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        atk.set_normalization_used(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    
    # attacker_cls = registry.get_attack(attack_name)
    # attack_config = attack_configs[attack_name]
    # attacker = attacker_cls(model=model, device=device, **attack_config)

    # attack process
    top1_m = AverageMeter()
    adv_top1_m = AverageMeter()
    pbar = tqdm.tqdm(total=len(test_loader))
    for i, (images, labels) in enumerate(test_loader):
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)
        
        # clean acc
        logits = model(images)
        clean_acc = accuracy(logits, labels)[0]
        top1_m.update(clean_acc.item(), batchsize)
        
        # robust acc
        adv_images = atk(images, labels)
        with torch.no_grad():
            adv_logits = model(adv_images)
        adv_acc = accuracy(adv_logits, labels)[0]
        adv_acc = adv_acc.item()
        adv_top1_m.update(adv_acc, batchsize)
        pbar.update(1)
    result_dict.append({
        'model': config.exp.model,
        'attack': attack_name,
        'clean_acc': round(top1_m.avg, 4),
        'robust_acc': round(adv_top1_m.avg, 4)
    })
    print(f"{__file__}, result_dict: {result_dict}")
    if config.exp.ablation:
        config.exp.save_path = f'{config.exp.save_path}/{attack_name}/ablations/'
        result_dict[-1].update({'prompt_type': config.data_params.prompt_type, 'dataset_size': config.data_params.num_data_points})
        fn=f'{config.exp.model}_{config.data_params.prompt_type}_{config.data_params.num_data_points}.json'
        report_json(result_dict, config.exp.save_path, fn=f'{fn}.json')
    else:
        report_json(result_dict, f'{config.exp.save_path}/{attack_name}', fn=f'{config.exp.model}_{config.exp.dataset}.json')

if __name__ == "__main__":
    main()

