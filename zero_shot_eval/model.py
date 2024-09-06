import numpy as np
import torch
import torchvision
from third_party.Scaling.supervised.models_vit import create_model
from third_party.SynthCLIP.Training.models import CLIP_VITB16
from third_party.mocov3.vits import vit_base
from third_party.SynCLR.eval import models_vit as SynCLRVIT

# Models
# SimCLR, CLIP, DiNO, DiNov2, MoCov3, BeIT, MAE,  (All Unsupervised methods)
# Resnet, Deit, SwimTransformer, ConvNext (Supervised Models)

def load_model(model_name, backbone=False, **kwargs):
    # Supervised Models
    if model_name == 'DeiT':
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    if model_name == 'SwimT':
        model = torchvision.models.swin_b(pretrained=True)
    if model_name == 'ConvNext':
        model = torchvision.models.convnext_base(pretrained=True)
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    if 'dataset' in kwargs:
        model = torchvision.models.resnet50(pretrained=False)
        path = '/visinf/home/ksingh/syn-rep-learn/pretrained_models/Pretained_Models/supervised_models/'

        if kwargs['dataset'] == 'afhq':
            if kwargs['real']:
                ckpt = f'{path}/resent50_64_afhq_UNet_real_only'
            else:
                ckpt = f'{path}/resent50_64_afhq_UNet_synthetic_only'
        if kwargs['dataset'] == 'cars':
            if kwargs['real']:
                ckpt = f'{path}/resent50_64_cars_UNet_real_only'
            else:
                ckpt = f'{path}/resent50_64_cars_UNet_synthetic_only'
        if kwargs['dataset'] == 'flowers':
            if kwargs['real']:
                ckpt = f'{path}/resent50_64_flowers_UNet_real_only'
            else:
                ckpt = f'{path}/resent50_64_flowers_UNet_synthetic_only'
        if kwargs['dataset'] == 'cifar10':
            if kwargs['real']:
                ckpt = f'{path}/resent50_64_cifar10_UNet_real_only'
            else:
                ckpt = f'{path}/resent50_64_cifar10_UNet_synthetic_only'
        state_dict = torch.load(ckpt)
        model.load_state_dict(state_dict, strict=True)
 

    # Supervised Synthetic Models
    if model_name == 'syn_clone':
        path = '/visinf/home/ksingh/syn-rep-learn/pretrained_models/Pretrained_Models/synthetic_clone/imagenet_1k_sd.pth'
        ckpt = torch.load(path)
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(2048, 1000, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"
        model.load_state_dict(ckpt, strict=True)

    if model_name == 'scaling_imagenet_sup':
        if 'datasize' in  kwargs:
            path = f'/visinf/home/ksingh/syn-rep-learn/pretrained_models/Pretrained_Models/scaling/supervised/classname/{kwargs["datasize"]}.pt'
        else:
            path = f'/visinf/home/ksingh/syn-rep-learn/pretrained_models/Pretrained_Models/scaling/supervised/classname/16M.pt'
        model = create_model("vit_base_patch16_224", num_classes=1000)
        for name, param in model.named_parameters():
            param.requires_grad = False
        state_dict = torch.load(path, map_location='cuda:0')
        model.load_state_dict(state_dict, strict=True)
    

    # Unsupervised (Self-Supervised) Models 
    if model_name == 'CLIP':
        if not backbone:
            pass
            # model = get_model('vit-base-p16_clip-openai-pre_3rdparty_in1k', pretrained=True)
        else:
            pass
    if model_name == 'DiNov2':
        if not backbone:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
        else:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

    if model_name == 'MAE':
        pass
    if model_name == 'mocov3':
        base_dir = '/visinf/home/ksingh/pretrained_models/Pretrained_Models/mocov3/'
        model = vit_base
        if not backbone:
            model = vit_base()
            chkpt = torch.load(f'{base_dir}/mocov3_linear.tar')['state_dict']
            model.load_state_dict(chkpt, strict=True)
        else:
            model = vit_base()
            chkpt = torch.load(f'{base_dir}/vit-b-300ep.pth.tar')['state_dict']
            model.load_state_dict(chkpt, strict=True)

    if model_name == 'synthCLIP':
        if not backbone:
            path = ''
        else:
            path = './pretrained_models/Pretrained_Models/synthclip/vitb16-synthclip-30M/checkpoint_best.pt'
        model = CLIP_VITB16()
        ckpt = torch.load(path)
        for name, param in model.named_parameters():
            param.requires_grad = False
        state_dict = torch.load(ckpt)
        model = model.load_state_dict(state_dict, strict=True)

    if model_name == 'synclr':
        if not backbone:
            path = '/visinf/home/ksingh/checkpoints/linear_synclr_vitb/_bn/linear_best.pt'
        else:
            path = './pretrained_models/Pretrained_Models/synclr/synclr_vit_b_16.pth'

        model = SynCLRVIT.create_model('vit_base_patch16', num_classes=1000)
        ckpt = torch.load(path)
        for name, param in model.named_parameters():
            param.requires_grad = False
        state_dict = torch.load(ckpt)
        model = model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model