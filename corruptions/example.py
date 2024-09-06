import torch 
from robustbench.data import load_imagenet3dcc
from robustbench.utils import clean_accuracy, load_model
PATH_IMAGENET_3DCC = '/fastdata/ksingh/robust/ImageNet-3DCC/'
corruptions_3dcc = ['near_focus', 'far_focus', 'bit_error', 'color_quant', 
                   'flash', 'fog_3d', 'h265_abr', 'h265_crf',
                   'iso_noise', 'low_light', 'xy_motion_blur', 'z_motion_blur'] # 12 corruptions in ImageNet-3DCC

device = torch.device("cuda:0")
model = load_model('Standard_R50', dataset='imagenet', threat_model='corruptions').to(device)
for corruption in corruptions_3dcc:
    for s in [1, 2, 3, 4, 5]:  # 5 severity levels
        x_test, y_test = load_imagenet3dcc(n_examples=5000, corruptions=[corruption], severity=s, data_dir=PATH_IMAGENET_3DCC)
        acc = clean_accuracy(model, x_test.to(device), y_test.to(device), device=device)
        print(f'Model: {model_name}, ImageNet-3DCC corruption: {corruption} severity: {s} accuracy: {acc:.1%}')




