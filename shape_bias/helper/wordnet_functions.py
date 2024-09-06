from os.path import join as pjoin
import numpy as np
from shutil import copyfile
import os
import linecache as lc

import os
from os.path import join as pjoin

##################################################################
# DIRECTORIES
##################################################################

PROJ_DIR = str(os.environ.get("MODELVSHUMANDIR", "model-vs-human"))
assert (PROJ_DIR != "None"), "Please set the 'MODELVSHUMANDIR' environment variable as described in the README"
CODE_DIR = pjoin(PROJ_DIR, "modelvshuman")
DATASET_DIR = pjoin(PROJ_DIR, "datasets")
FIGURE_DIR = pjoin(PROJ_DIR, "figures")
RAW_DATA_DIR = pjoin(PROJ_DIR, "raw-data")
PERFORMANCES_DIR = pjoin(RAW_DATA_DIR, "performances")
REPORT_DIR = pjoin(PROJ_DIR, "latex-report/")
ASSETS_DIR = pjoin(PROJ_DIR, "assets/")
ICONS_DIR = pjoin(ASSETS_DIR, "icons/")

##################################################################
# CONSTANTS
##################################################################

IMG_SIZE = 224  # size of input images for most models

##################################################################
# DATASETS
##################################################################

NOISE_GENERALISATION_DATASETS = ["colour",
                                 "contrast",
                                 "high-pass",
                                 "low-pass",
                                 "phase-scrambling",
                                 "power-equalisation",
                                 "false-colour",
                                 "rotation",
                                 "eidolonI",
                                 "eidolonII",
                                 "eidolonIII",
                                 "uniform-noise"]

TEXTURE_SHAPE_DATASETS = ["original", "greyscale",
                          "texture", "edge", "silhouette",
                          "cue-conflict"]

DEFAULT_DATASETS = ["edge", "silhouette", "cue-conflict"] + \
                   NOISE_GENERALISATION_DATASETS + ["sketch", "stylized"]
##################################################################
# PLOT TYPES
##################################################################

PLOT_TYPE_TO_DATASET_MAPPING = {
    # default plot types:
    "shape-bias": ["cue-conflict"],
    "accuracy": NOISE_GENERALISATION_DATASETS,
    "nonparametric-benchmark-barplot": ["edge", "silhouette", "sketch", "stylized"],
    "benchmark-barplot": DEFAULT_DATASETS,
    "scatterplot": DEFAULT_DATASETS,
    "error-consistency-lineplot": NOISE_GENERALISATION_DATASETS,
    "error-consistency": ["cue-conflict", "edge", "silhouette", "sketch", "stylized"],
    # 'unusual' plot types:
    "entropy": NOISE_GENERALISATION_DATASETS,
    "confusion-matrix": DEFAULT_DATASETS,
    }

DEFAULT_PLOT_TYPES = list(PLOT_TYPE_TO_DATASET_MAPPING.keys())
DEFAULT_PLOT_TYPES.remove("entropy")
DEFAULT_PLOT_TYPES.remove("confusion-matrix")

##################################################################
# MODELS
##################################################################

TORCHVISION_MODELS = ["alexnet",
                      "vgg11_bn",
                      "vgg13_bn",
                      "vgg16_bn",
                      "vgg19_bn",
                      "squeezenet1_0",
                      "squeezenet1_1",
                      "densenet121",
                      "densenet169",
                      "densenet201",
                      "inception_v3",
                      "resnet18",
                      "resnet34",
                      "resnet50",
                      "resnet101",
                      "resnet152",
                      "shufflenet_v2_x0_5",
                      "mobilenet_v2",
                      "resnext50_32x4d",
                      "resnext101_32x8d",
                      "wide_resnet50_2",
                      "wide_resnet101_2",
                      "mnasnet0_5",
                      "mnasnet1_0"]

BAGNET_MODELS = ["bagnet9", "bagnet17", "bagnet33"]

SHAPENET_MODELS = ["resnet50_trained_on_SIN",
                   "resnet50_trained_on_SIN_and_IN",
                   "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN"]

SIMCLR_MODELS = ["simclr_resnet50x1", "simclr_resnet50x2", "simclr_resnet50x4"]

PYCONTRAST_MODELS = ["InsDis", "MoCo", "PIRL", "MoCoV2", "InfoMin"]

SELFSUPERVISED_MODELS = SIMCLR_MODELS + PYCONTRAST_MODELS

EFFICIENTNET_MODELS = ["efficientnet_b0", "noisy_student"]

ADV_ROBUST_MODELS = ["resnet50_l2_eps0", "resnet50_l2_eps0_5",
                     "resnet50_l2_eps1", "resnet50_l2_eps3",
                     "resnet50_l2_eps5"]

VISION_TRANSFORMER_MODELS = ["vit_small_patch16_224", "vit_base_patch16_224",
                             "vit_large_patch16_224"]

BIT_M_MODELS = ["BiTM_resnetv2_50x1", "BiTM_resnetv2_50x3", "BiTM_resnetv2_101x1",
                "BiTM_resnetv2_101x3", "BiTM_resnetv2_152x2", "BiTM_resnetv2_152x4"]

SWAG_MODELS = ["swag_regnety_16gf_in1k", "swag_regnety_32gf_in1k", "swag_regnety_128gf_in1k",
               "swag_vit_b16_in1k", "swag_vit_l16_in1k", "swag_vit_h14_in1k"]



categories = None
WNIDs = None
IMAGENET_CATEGORIES_FILE = pjoin(CODE_DIR, "helper", "categories.txt")

def get_filenames_of_category(category, image_labels_path, categories):
    """Return a list of filenames of all images belonging to a category.

    category - a string specifying a (perhaps broad) category
    image_labels_path - a filepath to a file with all image labels,
                        formatted in the ilsvrc2012 format
    categories - a list of all categories of the dataset. The order of
                 categories has to be the same as used for the labelling.

    """

    # get indices of all subcategories that belong to the category
    subcategories_list = []
    counter = 0
    for cat in categories:
        if is_hypernym(cat, category):
            subcategories_list.append(counter)
        counter += 1


    image_list = []
    with open(image_labels_path) as labels_file:
        for line in labels_file:
            image_name, image_label = line.split(" ")

            if int(image_label) in subcategories_list:
                image_list.append(image_name)

    return image_list


def hypernyms_in_ilsvrc2012_categories(entity):
    """Return all hypernyms of categories.txt for a given entity.

    entity - a string, e.g. "furniture"

    Returns the children of the entity, e.g. "bed" and "chair" if there were
    both a "bed" and a "chair" in categories.txt (the imagenet categories).
    If the entity itself is contained, it will be returned as well.
    """

    return get_hypernyms("categories.txt", entity)


def get_hypernyms(categories_file, entity):
    """Return all hypernyms of categories for a given entity.

    entity - a string, e.g. "furniture"

    Returns the children of the entity, e.g. "bed" and "chair" if there were
    both a "bed" and a "chair" in the categories.
    If the entity itself is contained, it will be returned as well.
    """

    hypers = []
    with open(categories_file) as f:
        for line in f:
            category = get_category_from_line(line)
            cat_synset = wn.synsets(category)[0]
            if is_hypernym(category, entity):
                hypers.append(category)

    return hypers


def get_ilsvrc2012_training_WNID(entity):
    """Return a WNID for each hypernym of entity.

    entity - a string, e.g. "furniture"

    Returns the WNIDs of the children of the entity,
    e.g. "bed" and "chair" if there were
    both a "bed" and a "chair" in the ilsvrc2012 categories.
    If the entity itself is contained, it will be returned as well.
    """

    results = []

    hypernyms = hypernyms_in_ilsvrc2012_categories(entity)

    for hyper in hypernyms:

        with open("WNID_synsets_mapping.txt") as f:
            for line in f:
                category = get_category_from_line(line)

                if category == hyper:
                    print(line[:9])
                    results.append(line[:9])

    return results


def num_hypernyms_in_ilsvrc2012(entity):
    """Return number of hypernyms in the ilsvrc2012 categories for entity."""

    return len(hypernyms_in_ilsvrc2012_categories(entity))


def get_ilsvrc2012_categories():
    """
        Return the first item of each synset of the ilsvrc2012 categories.
        Categories are lazy-loaded the first time they are needed.
    """

    global categories
    if categories is None:
        categories = []
        with open(IMAGENET_CATEGORIES_FILE) as f:
            for line in f:
                categories.append(get_category_from_line(line))

    return categories


def get_ilsvrc2012_WNIDs():
    """
        Return the first item of each synset of the ilsvrc2012 categories.
        Categories are lazy-loaded the first time they are needed.
    """

    global WNIDs
    if WNIDs is None:
        WNIDs = []
        with open(IMAGENET_CATEGORIES_FILE) as f:
            for line in f:
                WNIDs.append(get_WNID_from_line(line))

    return WNIDs


def get_category_from_line(line):
    """Return the category without anything else from categories.txt"""

    category = line.split(",")[0][10:]
    category = category.replace(" ", "_")
    category = category.replace("\n", "")
    return category


def get_WNID_from_line(line):
    """Return the WNID without anything else from categories.txt"""
    
    WNID = line.split(" ")[0]
    return WNID


def get_WNID_from_index(index):
    """Return WNID given an index of categories.txt"""
    assert (index >= 0 and index < 1000), "index needs to be within [0, 999]"

    file_path = IMAGENET_CATEGORIES_FILE
    assert(os.path.exists(file_path)), "path to categories.txt wrong!"
    line = lc.getline(file_path, index+1)
    return line.split(" ")[0]

