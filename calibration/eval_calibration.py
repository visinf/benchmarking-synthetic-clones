import os
import sys
from pathlib import Path

path_root = Path(__file__).resolve().parent.parent
sys.path.append(str(path_root))
import hydra
import torch
import torch.backends.cudnn as cudnn
# Import metrics to compute
from Metrics.metrics import (AdaptiveECELoss, ClasswiseECELoss, ECELoss,
                             test_classification_net_logits)
from Metrics.plots import _populate_bins, bin_strength_plot, reliability_plot
from torch import nn
from torch.utils.data import DataLoader

from datasets.load_datasets import load_data
from model import load_model
from utils import report_json, seed_everything
from utils import obtain_ImageNet_classes, ClassificationModel, get_text_features



hydra.output_subdir="null"
from omegaconf import DictConfig, OmegaConf


def get_logits_labels(data_loader, net, mask=None):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for idx, item in enumerate(data_loader):
            data, label = item
            data = data.cuda()
            if mask is not None:
                logits = net(data)[:, mask]
            else:
                logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

    return logits, labels



# OmegaConf.register_new_resolver("bin_size", lambda x: 20)
@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(config):
    mask = None
    if config.exp.dataset in ['imagenet-a', 'imagenet-r']:
        mask, test_dset = load_data(config.exp.dataset, is_clip=config.exp.model in ['CLIP', 'scaling_clip', 'synthCLIP'])
    else:
        _, test_dset = load_data(config.exp.dataset, is_clip=config.exp.model in ['CLIP', 'scaling_clip', 'synthCLIP'])

    test_loader = DataLoader(
        test_dset,
        batch_size=config.exp.batch_size,
        shuffle=False,
        num_workers=config.exp.n_workers,
        pin_memory=False,
    )
    if config.exp.ablation:
        print('Running ablation...')
        pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone, prompt_type=config.data_params.prompt_type, size=config.data_params.num_data_points)
    else:
        pretrained_model = load_model(config.exp.model, backbone=config.exp.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device)
    if config.exp.model in ['synthCLIP', 'CLIP', 'scaling_clip']:
        print('Using CLIP')
        if config.exp.dataset.startswith('imagenet'):
            class_labels = obtain_ImageNet_classes()
        else:
            return None 

        text_features = get_text_features(pretrained_model, dataset_name='imagenet', class_labels=class_labels)
        pretrained_model= ClassificationModel(model=pretrained_model,text_embedding=text_features)
        pretrained_model.to(device)

    seed_everything(config.exp.seed)


    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss(n_bins=config.exp.num_bins).cuda()
    adaece_criterion = AdaptiveECELoss(n_bins=config.exp.num_bins).cuda()
    cece_criterion = ClasswiseECELoss(n_bins=config.exp.num_bins).cuda()
    logits, labels = get_logits_labels(test_loader, pretrained_model, mask)
    conf_matrix, p_accuracy, labels_list, predictions_list, confidences = test_classification_net_logits(logits, labels)


    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()

    print ('Test error: ' + str((1 - p_accuracy)))
    print ('Test NLL: ' + str(p_nll))
    print ('ECE: ' + str(p_ece))
    print ('AdaECE: ' + str(p_adaece))
    print ('Classwise ECE: ' + str(p_cece))
    result_dict = {
        'model': config.exp.model,
        'dataset': config.exp.dataset,
        'test_error': 1-p_accuracy,
        'test_nll': p_nll,
        'ece': p_ece,
        'ada ece': p_adaece,
        'class ece': p_cece
    }
    print(result_dict)

    if config.exp.ablation:
        config.exp.save_path = f'{config.exp.save_path}/ablations/'
        result_dict.update({'prompt_type': config.data_params.prompt_type, 'dataset_size': config.data_params.num_data_points})
        fn=f'{config.exp.model}_{config.exp.dataset}_{config.data_params.prompt_type}_{config.data_params.num_data_points}.json'
        report_json(result_dict, config.exp.save_path, fn)
    else:
        report_json(result_dict, config.exp.save_path, fn=f'{config.exp.model}_{config.exp.dataset}.json')
        bin_dict = _populate_bins(confidences, predictions_list, labels_list, num_bins=config.exp.num_bins)
        report_json(bin_dict, config.exp.save_path, fn=f'rel_plot_{config.exp.model}_{config.exp.dataset}.json')

    # fn = config.exp.save_path + f'{config.exp.model}_{config.exp.dataset}'  + '.tex'
    # reliability_plot(fn, confidences, predictions_list, labels, num_bins=config.exp.num_bins)
    # bin_strength_plot(confidences, predictions, labels, num_bins=config.exp.num_bins)


if __name__ == "__main__":
    main()