import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
path_root = Path(__file__).resolve().parent.parent
sys.path.append(str(path_root))
import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from helper import human_categories as hc
from torch import nn
import timm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torchvision.transforms as T
from collections import defaultdict

from model import load_model
from utils import report_json, seed_everything
from utils import obtain_ImageNet_classes, ClassificationModel, get_text_features

# Import metrics to compute

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Adapted from:
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __init__(self, *args, **kwargs):
        super(ImageFolderWithPaths, self).__init__(*args, **kwargs)


    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        # this is what ImageFolder normally returns
        (sample, target) = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        original_tuple = (sample, target)

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



class DecisionMapping(ABC):
    def check_input(self, probabilities):
        assert type(probabilities) is np.ndarray
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()

    @abstractmethod
    def __call__(self, probabilities):
        pass


class ImageNetProbabilitiesTo16ClassesMapping(DecisionMapping):
    """Return the 16 class categories sorted by probabilities"""

    def __init__(self, aggregation_function=None):
        if aggregation_function is None:
            aggregation_function = np.mean
        self.aggregation_function = aggregation_function
        self.categories = hc.get_human_object_recognition_categories()

    def __call__(self, probabilities):
        self.check_input(probabilities)

        aggregated_class_probabilities = []
        c = hc.HumanCategories()

        for category in self.categories:
            indices = c.get_imagenet_indices_for_category(category)
            values = np.take(probabilities, indices, axis=-1)
            aggregated_value = self.aggregation_function(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value)
        aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)
        sorted_indices = np.flip(np.argsort(aggregated_class_probabilities, axis=-1), axis=-1)
        return np.take(self.categories, sorted_indices, axis=-1)






cat_dict = {
    'airplane': 0, 'bear': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'car': 6,
    'cat': 7, 'chair': 8, 'clock': 9, 'dog': 10, 'elephant': 11, 'keyboard': 12, 'knife': 13, 
    'oven': 14, 'truck': 15
}

# OmegaConf.register_new_resolver("bin_size", lambda x: 20)
@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(config):
    fraction_correct_shape = 0.
    fraction_correct_shape_per_cat = {}
    fraction_correct_texture_per_cat = {}
    fraction_correct_texture = 0.
    cat_total = {}

    total = 0.
    input_size = 224
    test_transform = T.Compose(
        [
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if config.exp.model in ['synthCLIP', 'CLIP', 'scaling_clip']:
        OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
        test_transform = T.Compose(
            [
                timm.data.transforms.ResizeKeepRatio(input_size, interpolation='bicubic'),
                T.CenterCrop(input_size),
                T.ToTensor(),
                T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )


    test_dset = ImageFolderWithPaths(config.exp.base_path, transform=test_transform)
    # test_dset = torch.utils.data.Subset(test_dset, range(1, 100))
    test_loader = DataLoader(test_dset, batch_size=64, shuffle=True, pin_memory=True)
    decision_mapping = ImageNetProbabilitiesTo16ClassesMapping()
    print(f"{__file__}, len(test_set): {len(test_dset)}")

    # pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone)
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
        else:
            class_labels = test_loader.dataset.class_names_str

        text_features = get_text_features(pretrained_model, dataset_name=config.exp.dataset, class_labels=class_labels)
        pretrained_model= ClassificationModel(model=pretrained_model,text_embedding=text_features)
 
    seed_everything(config.exp.seed)
    correct_shape = 0.
    correct_texture = 0.
    correct_shape_per_cat = {k: 0 for k in cat_dict.keys()}
    correct_texture_per_cat = {k: 0 for k in cat_dict.keys()}
    total_per_cat = {k: 0 for k in cat_dict.keys()}
    with torch.no_grad():
        for images, target, img_path in tqdm(test_loader):
            images = images.to(device)
            logits = pretrained_model(images)
            softmax_output = torch.nn.functional.softmax(logits).detach().cpu().numpy()
            predictions = decision_mapping(softmax_output)
            for idx in range(0, len(predictions)):
                shape_target = target[idx]
                texture_target = img_path[idx].split('/')[-1].split('-')[-1][:-4]
                texture_target = ''.join((x for x in texture_target if not x.isdigit()))
                texture_target = cat_dict[texture_target]
                pred = cat_dict[predictions[idx][0]]
                if shape_target == texture_target:
                    continue
                elif shape_target == pred:
                    correct_shape+= 1
                    correct_shape_per_cat[predictions[idx][0]] += 1
                elif texture_target == pred:
                    correct_texture += 1
                    correct_texture_per_cat[predictions[idx][0]] += 1
                total_per_cat[predictions[idx][0]] += 1
                total+=1

    frac_correct_shape = correct_shape / total
    frac_correct_texture = correct_texture / total
    shape_bias = frac_correct_shape / (frac_correct_shape + frac_correct_texture)
    frac_correct_shape_per_cat = {k: 0 for k in cat_dict.keys()}
    frac_correct_texture_per_cat = {k: 0 for k in cat_dict.keys()}

    for key, val in correct_shape_per_cat.items():
        if not total_per_cat[key] == 0:
            frac_correct_shape_per_cat[key] = val / total_per_cat[key]
        else:
            frac_correct_shape_per_cat[key] = 0

    for key, val in correct_texture_per_cat.items():
        if not total_per_cat[key] == 0:
            frac_correct_texture_per_cat[key] = val / total_per_cat[key]
        else:
            frac_correct_texture_per_cat[key] = 0

    shape_bias_per_cat = {k: 0 for k in cat_dict.keys()}
    for key, val in correct_texture_per_cat.items():
        if not frac_correct_shape_per_cat[key] + frac_correct_texture_per_cat[key] == 0:
            shape_bias_per_cat[key] = frac_correct_shape_per_cat[key] / (frac_correct_shape_per_cat[key] + frac_correct_texture_per_cat[key])
        else:
            shape_bias_per_cat[key] = 0

    result_dict = {
        'model_name': config.exp.model,
        'dataset': 'cue_conflict',
        'shape_bias': shape_bias,
        'shape_bias_per_cat': shape_bias_per_cat
    }

    if config.exp.ablation:
        config.exp.save_path = f'{config.exp.save_path}/ablations/'
        result_dict.update({'prompt_type': config.data_params.prompt_type, 'dataset_size': config.data_params.num_data_points})
        fn=f'{config.exp.model}_{config.data_params.prompt_type}_{config.data_params.num_data_points}.json'
        report_json(result_dict, config.exp.save_path, fn=f'{fn}.json')
    else:
        report_json(result_dict, config.exp.save_path, fn=f'{config.exp.model}.json')
    

if __name__ == "__main__":
    main()
