# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import argparse
import os
import sys
from time import time
from tqdm import tqdm

import data
import modeling
import torch
from torch.utils.data import DataLoader

import utils
import numpy as np
from external.robustness.imagenet_c import corrupt
import cv2


def main(args):
    """
    Main routine to extract features.
    """

    print("==> Initializing the pretrained resnet-model")
    model_resnet = modeling.build_model(
        args.arch, ckpt_file=args.ckpt_file_resnet, ckpt_key=args.ckpt_key, device=args.device
    )

    print("==> Initializing the pretrained sd-model")
    model_sd = modeling.build_model(
        args.arch, ckpt_file=args.ckpt_file_sd, ckpt_key=args.ckpt_key, device=args.device
    )

    for corruption in [
            "gaussian_noise", 
            "shot_noise", 
            "impulse_noise", 
            "defocus_blur",
            "glass_blur",
            "zoom_blur", 
            "snow", 
            "frost", 
            "fog",
            "brightness", 
            "contrast", 
            "pixelate", 
            "jpeg_compression",
            "speckle_noise",
            "gaussian_blur", 
            "spatter", 
            "saturate",
            "motion_blur",
            "elastic_transform",
        ]:
        for severity in [1, 2, 3, 4, 5]:
            # extract features from training and test sets
            #for split in ("trainval", "test"):
            for split in ["trainval", "test"]:
                dataset = data.load_dataset(
                    args.dataset,
                    args.dataset_dir,
                    split,
                    args.dataset_image_size,
                    cog_levels_mapping_file=args.cog_levels_mapping_file,
                    cog_concepts_split_file=args.cog_concepts_split_file,
                )
                print(
                    "==> Extracting features from {} / {} (size: {})".format(
                        args.dataset, split, len(dataset)
                    )
                )
                print(" Data loading pipeline: {}".format(dataset.transform))
                
                # Corrupt images only for the test dataset
                if split == "test":
                    print("==> Corrupting images in the test dataset")
                    corrupt_dataset(dataset, corruption_name=corruption, severity=severity)

                    dataset = data.load_dataset(
                        args.dataset,
                        args.dataset_dir_cc,
                        split,
                        args.dataset_image_size,
                        cog_levels_mapping_file=args.cog_levels_mapping_file,
                        cog_concepts_split_file=args.cog_concepts_split_file,
                    )
                    print(
                        "==> Extracting features from {} / {} (size: {})".format(
                            args.dataset, split, len(dataset)
                        )
                    )

                for model in [model_resnet, model_sd]:
                    #X, Y = extract_features_loop(
                    #    model, dataset, args.batch_size, args.n_workers, args.device
                    #)
                    #if model == model_resnet:
                    #    features_file = os.path.join(args.output_dir_resnet, f"fv_{args.dataset}_{corruption}_{severity}", "features_{}.pth".format(split))
                    #elif model == model_sd:
                    #    features_file = os.path.join(args.output_dir_sd, f"fv_{args.dataset}_sd_{corruption}_{severity}", "features_{}.pth".format(split))
                    # Create parent directories if they don't exist
                    #os.makedirs(os.path.dirname(features_file), exist_ok=True)
                    #print(" Saving features under {}".format(features_file))
                    #torch.save({"X": X, "Y": Y}, features_file)


                    X, Y = extract_features_loop(
                        model, dataset, args.batch_size, args.n_workers, args.device
                    )
                    if model == model_resnet:
                        base_dir = args.output_dir_resnet
                    elif model == model_sd:
                        base_dir = args.output_dir_sd

                    # Create parent directories if they don't exist
                    if model == model_resnet:
                        parent_dir = os.path.join(base_dir, f"fv_{args.dataset}_{corruption}_{severity}")
                    if model == model_sd:
                        parent_dir = os.path.join(base_dir, f"fv_{args.dataset}_sd_{corruption}_{severity}")
                    os.makedirs(parent_dir, exist_ok=True)

                    # Form the complete features_file path
                    features_file = os.path.join(parent_dir, f"features_{split}.pth")

                    print(" Saving features under {}".format(features_file))
                    torch.save({"X": X, "Y": Y}, features_file)


def corrupt_dataset(dataset, corruption_name, severity=1):
    corrupted_dataset = []
    print(corruption_name, "severity: ", severity)
    for idx, (image, label) in enumerate(tqdm(dataset, desc="Corrupting Dataset")):
        image_path = dataset.get_path(idx)  # Assuming a get_path method exists in your dataset class
        #print(f"Image {idx + 1}: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping {image_path}. Unable to read image.")
            continue

        # Resize the image only when corruption_name is in the specified list
        if corruption_name in ["glass_blur", "zoom_blur", "snow", "frost", "fog"]:
            resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        else:
            resized_image = image

        #print("image_path", image_path)
        output_path = get_output_path(image_path)
        #print("output_path", output_path)
        
        #print(f"Image {idx + 1}: {output_path}")
        if resized_image is not None:
            corrupted_image = corrupt(resized_image, severity=severity, corruption_name=corruption_name)
            cv2.imwrite(output_path, corrupted_image)
    return

def get_output_path(image_path):
    parts = image_path.split(os.sep)
    #vilab10_index = parts.index("vilab10")
    vilab10_index = parts.index("transfer_datasets")
    next_directory = parts[vilab10_index + 1]
    parts[vilab10_index + 1] = f"{next_directory}_cc"
    output_path = os.path.join(*parts)

    # Ensure the output path starts with a slash
    if not output_path.startswith(os.sep):
        output_path = os.sep + output_path
    return output_path


def extract_features_loop(
    model, dataset, batch_size=128, n_workers=12, device="cuda", print_iter=50
):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=False,
    )

    # (feature, label) pairs to be stored under args.output_dir
    X = None
    Y = None
    six = 0  # sample index

    t_per_sample = utils.AverageMeter("time-per-sample")
    t0 = time()

    with torch.no_grad():
        for bix, batch in enumerate(dataloader):
            assert (
                len(batch) == 2
            ), "Data loader should return a tuple of (image, label) every iteration."
            image, label = batch
            feature = model(image.to(device))

            if X is None:
                print(
                    " Size of the first batch: {} and features {}".format(
                        list(image.shape), list(feature.shape)
                    ),
                    flush=True,
                )
                X = torch.zeros(
                    len(dataset), feature.size(1), dtype=torch.float32, device="cpu"
                )
                Y = torch.zeros(len(dataset), dtype=torch.long, device="cpu")

            bs = feature.size(0)
            X[six : six + bs] = feature.cpu()
            Y[six : six + bs] = label
            six += bs

            t1 = time()
            td = t1 - t0
            t_per_sample.update(td / bs, bs)
            t0 = t1

            if (bix % print_iter == 0) or (bix == len(dataloader) - 1):
                print(
                    " {:6d}/{:6d} extracted, {:5.3f} secs per sample, {:5.1f} mins remaining".format(
                        six,
                        len(dataset),
                        t_per_sample.avg,
                        (t_per_sample.avg / 60) * (len(dataset) - six),
                    ),
                    flush=True,
                )

    assert six == len(X)
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet50"],
        help="The architecture of the pretrained model",
    )
    parser.add_argument(
        "--ckpt_file_resnet",
        type=str,
        required=True,
        help="Model checkpoint file",
    )
    parser.add_argument(
        "--ckpt_file_sd",
        type=str,
        required=True,
        help="sd-Model checkpoint file",
    )
    parser.add_argument(
        "--ckpt_key",
        type=str,
        default="",
        help="""Key in the checkpoint dictionary that corresponds to the model's state_dict
        For instance, if the checkpoint dictionary contains
        {'optimizer': optimizer.state_dict(),
         'model': model.state_dict(),
         ...}
        then this argument should be 'model'.
        If the checkpoint dictionary is the model state_dict itself, leave this argument empty.""",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="in1k",
        choices=[
            "in1k",
            "cog_l1",
            "cog_l2",
            "cog_l3",
            "cog_l4",
            "cog_l5",
            "aircraft",
            "cars196",
            "dtd",
            "eurosat",
            "flowers",
            "pets",
            "food101",
            "sun397",
            "inat2018",
            "inat2019",
        ],
        help="From which datasets to extract features",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--dataset_dir_cc",
        type=str,
        required=True,
        help="Path to the corrupted dataset",
    )
    parser.add_argument(
        "--dataset_image_size",
        type=int,
        default=224,
        help="Size of images given as input to the network before extracting features",
    )
    parser.add_argument(
        "--output_dir_resnet",
        type=str,
        required=True,
        help="Where to extract features.",
    )
    parser.add_argument(
        "--output_dir_sd",
        type=str,
        required=True,
        help="Where to extract features.",
    )
    parser.add_argument(
        "--cog_levels_mapping_file",
        type=str,
        help="Pickle file containing a list of concepts in each level (5 lists in total)."
        'This is an optional argument that needs to be set if --dataset is one of CoG levels, i.e., one of ("cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")',
    )
    parser.add_argument(
        "--cog_concepts_split_file",
        type=str,
        help="Pickle file containing training and test splits for each concept in ImageNet-CoG."
        'This is an optional argument that needs to be set if --dataset is one of CoG levels, i.e., one of ("cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size used during feature extraction",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers run for the data loader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Whether to use CUDA during feature extraction",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt_file_resnet):
        print(
            "Checkpoint file ({}) not found. "
            "Please provide a valid checkpoint file path for the pretrained model.".format(
                args.ckpt_file_resnet
            )
        )
        sys.exit(-1)
    if not os.path.isfile(args.ckpt_file_sd):
        print(
            "Checkpoint file ({}) not found. "
            "Please provide a valid checkpoint file path for the pretrained model.".format(
                args.ckpt_file_sd
            )
        )
        sys.exit(-1)

    if not os.path.isdir(args.dataset_dir):
        print(
            "Dataset not found under {}. "
            "Please provide a valid dataset path".format(args.dataset_dir)
        )
        sys.exit(-1)

    if args.dataset in ["cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5"] and not (
        os.path.isfile(args.cog_levels_mapping_file)
        and os.path.isfile(args.cog_concepts_split_file)
    ):
        print(
            "ImageNet-CoG files are not found. "
            "Please check the <cog_levels_mapping_file> and <cog_concepts_split_file> arguments."
        )
        sys.exit(-1)

    if (not torch.cuda.is_available()) or (torch.cuda.device_count() == 0):
        print("No CUDA-compatible device found. " "We will only use CPU.")
        args.device = "cpu"

    os.makedirs(args.output_dir_resnet, exist_ok=True)
    os.makedirs(args.output_dir_sd, exist_ok=True)
    utils.print_program_info(args)

    main(args)