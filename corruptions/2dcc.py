import sys
from pathlib import Path
path_root = Path(__file__).resolve().parent.parent
sys.path.append(str(path_root))
import torch 
import hydra
from torchvision import transforms
from loaders import CustomDatasetFolder, CustomImageFolder
from helper import AverageMeter, accuracy
from torch.utils.data import DataLoader
from torchvision import datasets
import tqdm
from model import load_model
from utils import ClassificationModel, get_text_features, obtain_ImageNet_classes
from utils import report_json, seed_everything


def get_transform(is_clip):
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    INCEPTION_MEAN = (0.5, 0.5, 0.5)
    INCEPTION_STD = (0.5, 0.5, 0.5)
    is_clip = False
    mean = IMAGENET_MEAN if not is_clip else OPENAI_DATASET_MEAN
    std = IMAGENET_STD if not is_clip else OPENAI_DATASET_STD

    normalize = transforms.Normalize(mean, std)
    print(f"{__file__}, mean: {mean}")
    print(f"{__file__}, std: {std}")

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform




# corruptions = ("snow", 
#                "gaussian_noise", "defocus_blur", "brightness", "fog",
#                "frost", "glass_blur", "contrast",
#                "jpeg_compression")
corruptions = ("shot_noise", "motion_blur", "snow", "pixelate", "jpeg_compression")
            #    "gaussian_noise", "defocus_blur", "brightness", "fog",
            #    "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
            #    "jpeg_compression", "elastic_transform")




device = torch.device("cuda:0")

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(config):
    is_clip=config.exp.model in ['CLIP', 'scaling_clip', 'synthCLIP']
    test_transform = get_transform(is_clip=is_clip)

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
        

    top1_m = AverageMeter()
    result_dict = []
    for corruption in corruptions:
        for s in [1, 2, 3, 4, 5]:  # 5 severity levels
            data_dir = f'/fastdata/ksingh/robust/Imagenet-C/{corruption}/{s}/'
            # test_dset = CustomImageFolder(root=data_dir, transform=test_transform)
            test_dset = datasets.ImageFolder(data_dir, test_transform)

            test_loader = DataLoader(
                test_dset,
                batch_size=config.exp.batch_size,
                shuffle=False,
                num_workers=config.exp.n_workers,
                pin_memory=False,
            )
            pbar = tqdm.tqdm(total=len(test_loader))
            for i, (images, labels) in enumerate(test_loader):
                # load data
                batchsize = images.shape[0]
                images, labels = images.to(device), labels.to(device)
                # clean acc
                logits = model(images)
                clean_acc = accuracy(logits, labels)[0]
                top1_m.update(clean_acc.item(), batchsize)
                pbar.update(1)
            result_dict.append({
                'model': config.exp.model,
                'corruption': corruption,
                'severity': s,
                'acc': round(top1_m.avg, 4)
            })
    
    if config.exp.ablation:
        config.exp.save_path = f'{config.exp.save_path}/2DCC/ablations/'
        result_dict[4].update({'prompt_type': config.data_params.prompt_type, 'dataset_size': config.data_params.num_data_points})
        fn=f'{config.exp.model}_{config.exp.dataset}_{config.data_params.prompt_type}_{config.data_params.num_data_points}.json'
        report_json(result_dict, config.exp.save_path, fn)
    else:
        report_json(result_dict, f'{config.exp.save_path}/2DCC/', fn=f'{config.exp.model}.json')

if __name__ == '__main__':
    main()