import os
import sys
from pathlib import Path
path_root = Path(__file__).resolve().parent.parent
sys.path.append(str(path_root))
path_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(path_root))

from torchvision import transforms
import torch.nn as nn
import numpy as np
import json
import os
import time
from PIL import Image
from datasets import ImageNet9
from model_utils import adv_bgs_eval_model, eval_model
import hydra
from model import load_model
import torch
from utils import report_json, seed_everything
from utils import obtain_ImageNet_classes, ClassificationModel, get_text_features


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(config):
    if config.exp.ablation:
        print('Running ablation...')
        pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone, prompt_type=config.data_params.prompt_type, size=config.data_params.num_data_points)
    else:
        pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device)

    if config.exp.model in ['synthCLIP', 'CLIP', 'scaling_clip']:
        if config.exp.dataset == 'imagenet':
            class_labels = obtain_ImageNet_classes()
        text_features = get_text_features(pretrained_model, dataset_name=config.exp.dataset, class_labels=class_labels)
        pretrained_model= ClassificationModel(model=pretrained_model,text_embedding=text_features)
    pretrained_model.to(device)
    seed_everything(config.exp.seed)

    map_to_in9 = {}
    with open('in_to_in9.json', 'r') as f:
        map_to_in9.update(json.load(f))

    BASE_PATH_TO_EVAL = config.exp.eval_path
    BATCH_SIZE = 32
    WORKERS = 8
    variations = ['original', 'mixed_same', 'mixed_rand']    
    result_dict = {}
    for variation in variations:
        # Load model
        in9_trained = False
        in9_ds = ImageNet9(f'{BASE_PATH_TO_EVAL}/{variation}')
        val_loader = in9_ds.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS)
        acc = eval_model(val_loader, pretrained_model, map_to_in9, map_in_to_in9=(not in9_trained))
        print('Evaluation complete')
        result_dict.update({
            'model': config.exp.model,
            f'{variation}_acc': acc,
        })
    print(f"result_dict: {result_dict}")
    if config.exp.ablation:
        config.exp.save_path = f'{config.exp.save_path}/ablations/'
        result_dict.update({'prompt_type': config.data_params.prompt_type, 'dataset_size': config.data_params.num_data_points})
        fn=f'{config.exp.model}_{config.data_params.prompt_type}_{config.data_params.num_data_points}.json'
        report_json(result_dict, config.exp.save_path, fn=f'{fn}.json')
    else:
        report_json(result_dict, config.exp.save_path, fn=f'{config.exp.model}.json')


if __name__ == "__main__":
    main()