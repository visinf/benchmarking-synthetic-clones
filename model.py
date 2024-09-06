import numpy as np
import torch
import torchvision
from third_party.Scaling.supervised.models_vit import create_model
from third_party.SynthCLIP.Training.models import CLIP_VITB16
from third_party.mocov3.vits import vit_base
from third_party.SynCLR.eval import models_vit as SynCLRVIT
from transformers import CLIPModel
from third_party.mae import models_vit
from open_clip import create_model_and_transforms, get_tokenizer
import torch.nn as nn
import timm


class SynCLR(torch.nn.Module):
    def __init__(self, model, linear_classifier) -> None:
        super().__init__()
        self.model = model
        self.linear_classifier = linear_classifier
        self.key = 'classifier_lr_0_0050' # best acc 80.45
        # print(f"{__file__}, key: {self.key}")

    def forward(self, images):
        features = self.model.forward_features(images)
        outputs = self.linear_classifier(features)[self.key]
        return outputs

# Models
# SimCLR, CLIP, DiNO, DiNov2, MoCov3, BeIT, MAE,  (All Unsupervised methods)
# Resnet, Deit, SwimTransformer, ConvNext (Supervised Models)
def convert_to_finetune(ckpt):
    # this is to convert simclr pre-trained model
    if 'visual.pos_embed' in ckpt.keys():
        new_ckpt = {}
        keyword = 'visual.'
        for k, v in ckpt.items():
            if k.startswith(keyword):
                new_k = k.replace(keyword, '')
                new_ckpt[new_k] = v
        return new_ckpt
    elif 'module.visual.pos_embed' in ckpt.keys():
        new_ckpt = {}
        keyword = 'module.visual.'
        for k, v in ckpt.items():
            if k.startswith(keyword):
                new_k = k.replace(keyword, '')
                new_ckpt[new_k] = v
        return new_ckpt

    return ckpt

class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)



def add_linear_classifier(feat_dim, num_classes, use_bn=False):
    learning_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    linear_classifier_dict = nn.ModuleDict()
    for blr in learning_rates:

        linear_classifier = nn.Linear(feat_dim, num_classes)
        linear_classifier.weight.data.normal_(mean=0.0, std=0.01)
        linear_classifier.bias.data.zero_()
        if use_bn:
            linear_classifier = nn.Sequential(
                torch.nn.SyncBatchNorm(feat_dim, affine=False, eps=1e-6),
                linear_classifier
            )
        linear_classifier.cuda()

        name = f"{blr:.4f}".replace('.', '_')
        linear_classifier_dict[f"classifier_lr_{name}"] = linear_classifier
    # add to ddp mode
    linear_classifiers = AllClassifiers(linear_classifier_dict)
    return linear_classifiers 





def load_model(model_name, backbone=False, **kwargs):
    base_path = '/visinf/home/ksingh/benchmarking-synthetic-clones/pretrained_models'

    # Supervised Models
    if model_name == 'DeiT':
        model = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True)
    if model_name == 'SwimT':
        model = timm.create_model('swin_base_patch4_window7_224.ms_in1k', pretrained=True)
    if model_name == 'ConvNext':
        model = timm.create_model('convnext_base', pretrained=True)
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True)
    if model_name == 'vit-b':
        model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)
    if 'dataset' in kwargs:
        model = torchvision.models.resnet50(pretrained=False)
        path = '/visinf/home/ksingh/syn-rep-learn/benchmarking-synthetic-clones/pretrained_models/Pretained_Models/supervised_models/'

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
        path = f'{base_path}/Pretrained_Models/synthetic_clone/imagenet_1k_sd.pth'
        ckpt = torch.load(path)
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(2048, 1000, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"
        model.load_state_dict(ckpt, strict=True)

    if model_name == 'scaling_imagenet_sup':
        if 'prompt_type' in kwargs:
            path = f'{base_path}/Pretrained_Models/scaling/supervised/{kwargs["prompt_type"]}/{kwargs["size"]}.pt'
        else:
            path = f'{base_path}/Pretrained_Models/scaling/supervised/classname/16M.pt'
        print(f'Loading model from :{path}')
        model = create_model("vit_base_patch16_224", num_classes=1000)
        for name, param in model.named_parameters():
            param.requires_grad = False
        state_dict = torch.load(path, map_location='cuda:0')
        model.load_state_dict(state_dict, strict=True)
    

    # Unsupervised (Self-Supervised) Models 

    if model_name == 'dino':
        if not backbone:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
        else:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

    if model_name == 'mae':
        base_dir = f'{base_path}/Pretrained_Models/mae/' 
        model = models_vit.vit_base_patch16(num_classes=1000, global_pool=False)
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        if not backbone:
            checkpoint = torch.load(f'{base_dir}/mae_lp_e90.pth')
            checkpoint_model = checkpoint['model']
            # load pre-trained model
            model.load_state_dict(checkpoint_model, strict=True)
        else:
            checkpoint = torch.load(f'{base_dir}/mae_pretrain_vit_base.pth', map_location='cpu')
            checkpoint_model = checkpoint['model']
            # load pre-trained model
            model.load_state_dict(checkpoint_model, strict=True)

    if model_name == 'mocov3':
        base_dir = f'{base_path}/Pretrained_Models/mocov3/'
        model = vit_base
        if not backbone:
            model = vit_base()
            state_dict = torch.load(f'{base_dir}/mocov3_linear.tar')['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model.load_state_dict(state_dict, strict=True)
        else:
            model = vit_base()
            chkpt = torch.load(f'{base_dir}/vit-b-300ep.pth.tar')['state_dict']
            model.load_state_dict(chkpt, strict=True)

    if model_name == 'synclr':
        if not backbone:
            path  = f'{base_path}/Pretrained_Models/synclr/linear_best.pt'
        else:
            path  = f'{base_path}/Pretrained_Models/synclr/synclr_vit_b_16.pt'

        model = SynCLRVIT.create_model('vit_base_patch16', num_classes=1000)
        for name, param in model.named_parameters():
            param.requires_grad = False
        checkpoint = torch.load(path)
        linear_keyword = 'head'
        if 'model' in checkpoint.keys():
                state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']

        state_dict = convert_to_finetune(state_dict)
        if 'module.visual.cls_token' in state_dict.keys():
            visual_keyword = 'module.visual.'
        elif 'visual.cls_token' in state_dict.keys():
            visual_keyword = 'visual.'
        else:
            visual_keyword = None

        if visual_keyword is not None:
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith(visual_keyword) and not k.startswith(visual_keyword + linear_keyword):
                    # remove prefix
                    # state_dict[k[len(visual_keyword):]] = torch.from_numpy(state_dict[k])
                    state_dict[k[len(visual_keyword):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        del model.head

        feat_dim = model.cls_token.shape[-1]
        cls_state_dict = checkpoint['linear_classifiers']
        linear_classifiers = add_linear_classifier(feat_dim, num_classes=1000, use_bn=True)
        visual_keyword = 'module.'
        if visual_keyword is not None:
            for k in list(cls_state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith(visual_keyword):
                    # remove prefix
                    # state_dict[k[len(visual_keyword):]] = torch.from_numpy(state_dict[k])
                    cls_state_dict[k[len(visual_keyword):]] = cls_state_dict[k]
                # delete renamed or unused k
                del cls_state_dict[k]
        model.load_state_dict(state_dict, strict=True)
        linear_classifiers.load_state_dict(cls_state_dict, strict=True)
        model = SynCLR(model, linear_classifiers)

    if model_name == 'CLIP':
        print(f"{__file__}, backbone: {backbone}")
        if not backbone:
            pass
            # model = get_model('vit-base-p16_clip-openai-pre_3rdparty_in1k', pretrained=True)
        else:
            # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            model, _, _ = create_model_and_transforms(
                'ViT-B-16',
                'openai')
            device = "cuda:0" if torch.cuda.is_available() else "cpu"


    
    if model_name == 'scaling_clip':
        path = f'{base_path}/Pretrained_Models/scaling/CLIP/Synthetic/'
        model, _, _ = create_model_and_transforms(
                            'ViT-B-16',
                            '',
                            precision='amp',
                            device='cuda',
                            jit=False,
                            force_quick_gelu=True,
                            force_custom_text=False,
                            force_patch_dropout=None,
                            force_image_size=224,
                            pretrained_image=False,
                            image_mean=None,
                            image_std=None,
                            aug_cfg={},
                            output_dict=True,
                        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if not backbone:
            pass 
        else:
            if 'prompt_type' in kwargs:
                ckpt = f'{base_path}/Pretrained_Models/scaling/CLIP/{kwargs["prompt_type"]}/{kwargs["size"]}.pt'
                state_dict = torch.load(ckpt, map_location=device)
            else:
                path = f'{base_path}/Pretrained_Models/scaling/CLIP/Synthetic'
                ckpt = f'{path}/371M.pt'
                state_dict = torch.load(ckpt, map_location=device)
                # logit_scale = np.exp(state_dict['logit_scale'].item())
                # print(f"{__file__}, logit_scale: {logit_scale}")
        model.load_state_dict(state_dict, strict=True)

    if model_name == 'synthCLIP':
        if not backbone:
            path = ''
        else:
            path = f'{base_path}/Pretrained_Models/synthclip/vitb16-synthclip-30M/checkpoint_best.pt'
        model = CLIP_VITB16()
        ckpt = torch.load(path)
        for name, param in model.named_parameters():
            param.requires_grad = False
        state_dict = torch.load(ckpt)
        model = model.load_state_dict(state_dict, strict=True)


    model.eval()
    return model