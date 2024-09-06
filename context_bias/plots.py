#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import trange
import tikzplotlib as tpl
import pandas as pd
import os
import argparse

def rename_fn(s):
    result = []
    rename_dict = {'CLIP': 'CLIP', 'resnet50': 'Resnet50', 
                    'SwimT': 'Swin-B', 
                    'syn_clone': 'Resnet50 (Syn.)', 
                    'scaling_imagenet_sup': 'ViT-B (Syn.)', 
                    'dino': 'DINOv2', 
                    'mocov3': 'MOCOv3', 
                    'synclr': 'SimCLR (Syn.)', 
                    'mae': 'MAE', 
                    'scaling_clip': 'CLIP (Syn.)', 
                    'vit-b': 'ViT-B'} 
        
    for i in s:
        if i in rename_dict.keys():
            result.append(rename_dict[i])
        else:
            result.append(i)
    return result

def sortDF(df, lst, colName='col1'):
    df['order'] = df[colName].apply(lambda x: lst.index(x))
    return df.sort_values(['order']).drop(columns=['order'])


def plot_accuracy_ablation():
    accuracies = {}
    categories = [
        "truck",
        "car",
        "plane",
        "ship",
        "cat",
        "dog",
        "horse",
        "deer",
        "frog",
        "bird",
    ]

    model_architectures = ['captions_1M', 'captions_4M', 'captions_8M', 'captions_16M',
                           'classname_1M', 'classname_4M', 'classname_8M', 'classname_16M',
                           'CLIP_templates_1M', 'CLIP_templates_4M', 'CLIP_templates_8M', 'CLIP_templates_16M',
                           'REAL_64M', 'REAL_128M', 'REAL_256M', 'REAL_371M', 
                           'Syn_Real_64M', 'Syn_Real_128M', 'Syn_Real_256M', 'Syn_Real_371M', 
                           'Synthetic_64M', 'Synthetic_128M', 'Synthetic_256M', 'Synthetic_371M', 
                           ]
    arch_labels = ['captions_1M', 'captions_4M', 'captions_8M', 'captions_16M',
                           'classname_1M', 'classname_4M', 'classname_8M', 'classname_16M',
                           'CLIP_templates_1M', 'CLIP_templates_4M', 'CLIP_templates_8M', 'CLIP_templates_16M',
                           'REAL_64M', 'REAL_128M', 'REAL_256M', 'REAL_371M', 
                           'Syn_Real_64M', 'Syn_Real_128M', 'Syn_Real_256M', 'Syn_Real_371M', 
                           'Synthetic_64M', 'Synthetic_128M', 'Synthetic_256M', 'Synthetic_371M', 
                           ]


    for model_architecture in model_architectures:
        predictions = np.load(f"/visinf/home/ksingh/syn-rep-learn/context_bias/results/ablations/{model_architecture}/predictions.npy")

        accuracies[model_architecture] = {c: np.zeros((2, 4)) for c in (categories + ["total"])}
        for idx in trange(predictions.shape[0] - 2): # last 2 are not images (leftover from batch_size * num_batches)
            try:
                category = np.where(predictions[idx, 0:10])[0][0]
            except IndexError as e:
                print(model_architecture, predictions[idx, 0:10])
                raise e
            num_uncommon = int(np.sum(predictions[idx, 10:13]))
            accuracies[model_architecture][categories[category]][0, num_uncommon] += predictions[idx, 13]
            accuracies[model_architecture][categories[category]][1, num_uncommon] += 1
            accuracies[model_architecture]["total"][0, num_uncommon] += predictions[idx, 13]
            accuracies[model_architecture]["total"][1, num_uncommon] += 1

    labels = ["$P_0$", "$P_1$", "$P_2$"]
    acc = np.zeros((len(labels), len(arch_labels)))
    for idx in range(len(arch_labels)):
        acc[:, idx] = 100 * accuracies[model_architectures[idx]]["total"][0, :3] / accuracies[model_architectures[idx]]["total"][1, :3]
    labels = ["$P_0$", "$P_1$", "$P_2$"]
    arch_labels = ['captions_1M', 'captions_4M', 'captions_8M', 'captions_16M',
                           'classname_1M', 'classname_4M', 'classname_8M', 'classname_16M',
                           'CLIP_templates_1M', 'CLIP_templates_4M', 'CLIP_templates_8M', 'CLIP_templates_16M',
                           'REAL_64M', 'REAL_128M', 'REAL_256M', 'REAL_371M', 
                           'Syn_Real_64M', 'Syn_Real_128M', 'Syn_Real_256M', 'Syn_Real_371M', 
                           'Synthetic_64M', 'Synthetic_128M', 'Synthetic_256M', 'Synthetic_371M', 
                           ]


    data_dict = {'model': [], 'P0': [], 'P1': [], 'P2': [], 'prompt': [], 'num_data_points': []}
    for idx, label in enumerate(arch_labels):
        if label.startswith('Syn_'):
            prompt = 'Syn+Real'
        elif label.startswith('CLIP_templates'):
            prompt = 'CLIP_templates'
        else:
            prompt = label.split('_')[-2]
        num_data_points = label.split('_')[-1]
        data_dict['num_data_points'].append(num_data_points)
        data_dict['prompt'].append(prompt)
        if prompt in ['captions', 'classname', 'CLIP_templates']:
            data_dict['model'].append('ViT-B (Syn.)')
        else:
            data_dict['model'].append('CLIP (Syn.)')

    data_dict['P0'] = acc[0]
    data_dict['P1'] = acc[1]
    data_dict['P2'] = acc[2]
    
    res_0 = []
    res_1 = []
    for i in range(0, len(data_dict['P0'])):
        val = (data_dict['P1'][i]) / data_dict['P0'][i]
        val = round(val*100, 2)

        res_0.append(str(round(data_dict['P1'][i], 2)) + ' (' + str(val) + ')')
        val = (data_dict['P2'][i]) / data_dict['P0'][i]
        val = round(val*100, 2)
        res_1.append(str(round(data_dict['P2'][i], 2)) + ' (' + str(val) + ')')
    data_dict['P1'] = res_0
    data_dict['P2'] = res_1
    data_df = pd.DataFrame(data_dict)

    # data_df['model'] = rename_fn(data_df['model'])
    fn = f"/visinf/home/ksingh/syn-rep-learn/context_bias/results/ablations/"
    os.makedirs(fn, exist_ok=True)
    # data_df = sortDF(data_df, ['ViT-B (Syn.)', 'CLIP (Syn.)'], 'model')

    data_df.to_latex(f'{fn}/all_acc_plots.tex', index=False, float_format="{:.2f}".format)



def plot_accuracy():
    accuracies = {}
    categories = [
        "truck",
        "car",
        "plane",
        "ship",
        "cat",
        "dog",
        "horse",
        "deer",
        "frog",
        "bird",
    ]

    model_architectures = ["resnet50", "DeiT", "SwimT", "ConvNext", "syn_clone", "scaling_imagenet_sup", "vit-b", "dino", "mae", "mocov3", "synclr", "CLIP", "scaling_clip"]
    arch_labels = ["Resnet-50", "DeiT", "Swin-B", "ConvNext", "Resnet-50 (Syn.)", "ViT-B (Syn.)", "ViT-B", "DINOv2", "MAE", "MOCOv3", "SimCLR (Syn.)", "CLIP", "CLIP (Syn.)"]

    for model_architecture in model_architectures:
        predictions = np.load(f"/visinf/home/ksingh/syn-rep-learn/context_bias/results/{model_architecture}/predictions.npy")

        accuracies[model_architecture] = {c: np.zeros((2, 4)) for c in (categories + ["total"])}
        for idx in trange(predictions.shape[0] - 2): # last 2 are not images (leftover from batch_size * num_batches)
            try:
                category = np.where(predictions[idx, 0:10])[0][0]
            except IndexError as e:
                print(model_architecture, predictions[idx, 0:10])
                raise e
            num_uncommon = int(np.sum(predictions[idx, 10:13]))
            accuracies[model_architecture][categories[category]][0, num_uncommon] += predictions[idx, 13]
            accuracies[model_architecture][categories[category]][1, num_uncommon] += 1
            accuracies[model_architecture]["total"][0, num_uncommon] += predictions[idx, 13]
            accuracies[model_architecture]["total"][1, num_uncommon] += 1

    labels = ["$P_0$", "$P_1$", "$P_2$"]
    acc = np.zeros((len(labels), len(arch_labels)))
    for idx in range(len(arch_labels)):
        acc[:, idx] = 100 * accuracies[model_architectures[idx]]["total"][0, :3] / accuracies[model_architectures[idx]]["total"][1, :3]
    labels = ["$P_0$", "$P_1$", "$P_2$"]
    arch_labels = ["Resnet50", "DeiT", "Swin-B", "ConvNext", "Resnet50 (Syn.)", "ViT-B (Syn.)", "ViT-B", "DINOv2", "MAE", "MOCOv3", "SimCLR (Syn.)", "CLIP", "CLIP (Syn.)"]
    data_dict = {'model': [], 'P0': [], 'P1': [], 'P2': []}
    for idx, label in enumerate(arch_labels):
        data_dict['model'].append(label)
    data_dict['P0'] = acc[0]
    data_dict['P1'] = acc[1]
    data_dict['P2'] = acc[2]
    
    res_0 = []
    res_1 = []
    for i in range(0, len(data_dict['P0'])):
        val = (data_dict['P1'][i]) / data_dict['P0'][i]
        val = round(val*100, 2)

        res_0.append(str(round(data_dict['P1'][i], 2)) + ' (' + str(val) + ')')
        val = (data_dict['P2'][i]) / data_dict['P0'][i]
        val = round(val*100, 2)
        res_1.append(str(round(data_dict['P2'][i], 2)) + ' (' + str(val) + ')')
    data_dict['P1'] = res_0
    data_dict['P2'] = res_1
    data_df = pd.DataFrame(data_dict)

    data_df['model'] = rename_fn(data_df['model'])
    fn = f"/visinf/home/ksingh/syn-rep-learn/context_bias/results/all_acc_plots.tex"

    data_df = sortDF(data_df, ['Resnet50', 'Resnet50 (Syn.)', 'Swin-B', 
                               'ConvNext', 'DeiT', 'ViT-B', 
                               'ViT-B (Syn.)', 'MAE', 'DINOv2', 
                               'MOCOv3', 'SimCLR (Syn.)', 'CLIP', 'CLIP (Syn.)'], 'model')
    data_df.to_latex(fn, index=False, float_format="{:.2f}".format)






    # width = 0.5
    # x = np.arange(len(arch_labels))
    # fig, ax = plt.subplots()
    # fig.set_size_inches(18, 6)
    # offsets = np.zeros((len(labels), len(arch_labels)))
    # for i in range(len(labels)):
    #     for idx in range(len(arch_labels)):
    #         offsets[i, idx] = (5 - i - len(arch_labels) // 2) * width / 3
    # for idx in range(len(labels)):
    #     ax.bar(x - offsets[idx], acc[idx], width / 3, label=labels[idx])
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_xticks(x)
    # ax.set_xticklabels(arch_labels)
    # plt.xticks(rotation=45)
    # ax.set_ylabel("Accuracy")
    # ax.legend(loc="upper left", ncol=3, framealpha=1.0)
    # plt.tick_params(labelsize=12.5)
    # fig.tight_layout()
    # plt.gcf().savefig("/visinf/home/ksingh/syn-rep-learn/context_bias/results/all_accuracy_plots.pdf")
    # fn = f"/visinf/home/ksingh/syn-rep-learn/context_bias/results/all_acc_plots.tex"
    # tpl.save(fn, axis_width=r'\figwidth', axis_height=r'\figheight')







# # arch_labels = ["ResNet50", "Wide-ResNet50-2", "MobileNet-v3-large", "EfficientNet-b4", "EfficientNet-b7", "CLIP", "ViT-B/16", "ResNeXt-50 (32x4d)"]
# arch_labels = ["Resnet-50"]




# for idx in range(len(model_architectures)):
#     correct_common = accuracies[model_architectures[idx]]["total"][0, 0]
#     total_common = accuracies[model_architectures[idx]]["total"][1, 0]
#     correct_uncommon = np.sum(accuracies[model_architectures[idx]]["total"][0, 1:3])
#     total_uncommon = np.sum(accuracies[model_architectures[idx]]["total"][1, 1:3])
#     print(f"{model_architectures[idx]}: {100 * (correct_common / total_common - correct_uncommon / total_uncommon)} ")




# for idx in range(len(model_architectures)):
#     plt.plot(range(3), 100 * accuracies[model_architectures[idx]]["total"][0, :3] / accuracies[model_architectures[idx]]["total"][1, :3], label=arch_labels[idx])
# plt.xticks(range(3))
# plt.gca().set_xticklabels(["$P_0$", "$P_1$", "$P_2$"])
# plt.xlabel("Partition")
# plt.ylabel("Accuracy")
# plt.legend(loc="upper right")
# fig = plt.gcf()
# fig.set_size_inches(7, 7)
# fig.savefig("./logs/all_accuracy_plots.pdf")




# labels = ["$P_0$", "$P_1$", "$P_2$"]
# acc = np.zeros((len(labels), len(arch_labels)))
# for idx in range(8):
#     acc[:, idx] = 100 * accuracies[model_architectures[idx]]["total"][0, :3] / accuracies[model_architectures[idx]]["total"][1, :3]
# labels = ["$P_0$", "$P_1$", "$P_2$"]
# # arch_labels = ["ResNet50", "Wide-ResNet50-2", "MobileNet-v3-large", "EfficientNet-b4", "EfficientNet-b7", "CLIP", "ViT-B/16", "ResNeXt-50 (32x4d)"]
# arch_labels = ["Resnet-50"]

# width = 0.5
# x = np.arange(len(arch_labels))
# fig, ax = plt.subplots()
# fig.set_size_inches(18, 6)
# offsets = np.zeros((len(labels), len(arch_labels)))
# for i in range(len(labels)):
#     for idx in range(len(arch_labels)):
#         offsets[i, idx] = (5 - i - len(arch_labels) // 2) * width / 3
# for idx in range(len(labels)):
#     ax.bar(x - offsets[idx], acc[idx], width / 3, label=labels[idx])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks(x)
# ax.set_xticklabels(arch_labels)
# # plt.xticks(rotation=45)
# ax.set_ylabel("Accuracy")
# ax.legend(loc="upper left", ncol=3, framealpha=1.0)
# plt.tick_params(labelsize=12.5)
# # fig.tight_layout()
# plt.gcf().savefig("./logs/all_accuracy_plots.pdf")




# num_rows = 2
# num_cols = 5
# fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, 4))
# for cat_idx, category in enumerate(categories):
#     i = cat_idx // num_cols
#     j = cat_idx % num_cols
    
#     for idx in range(4, len(model_architectures)):
#         n = accuracies[model_architectures[idx]][category][0, :3]
#         N = accuracies[model_architectures[idx]][category][1, :3]
#         p = n / N
#         yerr = 100 * 1.96 * np.sqrt(p * (1 - p) / N)
#         ax[i, j].errorbar(range(3), 100 * p, yerr=yerr, label=arch_labels[idx])
#         break
#     ax[i, j].set_title(category)
#     ax[i, j].set_xticks(range(3))
#     ax[i, j].set_xticklabels(["$P_0$", "$P_1$", "$P_2$"])
#     ax[i, j].set_ylabel("Accuracy")

# # plt.legend(arch_labels, loc="lower right")
# plt.tight_layout()
# plt.gcf().savefig("./logs/efficientnet_b7_classwise_accuracy_plots.pdf")




# labels = ["Common Settings", "1 Uncommon Attribute", "2 Uncommon Attributes"]
# width = 0.5
# x = np.arange(len(labels))
# fig, ax = plt.subplots()
# data = np.zeros((len(model_architectures), 3))
# offsets = np.zeros((len(model_architectures), 3))
# for idx in range(len(model_architectures)):
#     data[idx] = 100 * accuracies[model_architectures[idx]]["total"][0, :3] / accuracies[model_architectures[idx]]["total"][1, :3]
# sorted_idxs = np.argsort(data, axis=0)
# for i in range(3):
#     for idx in range(len(model_architectures)):
#         offsets[idx, i] = (np.where(sorted_idxs[:, i] == idx)[0][0] - len(model_architectures) // 2) * width / 6
# for idx in range(len(model_architectures)):
#     ax.bar(x - offsets[idx][0], data[idx], width / 6, label=arch_labels[idx])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# #     plt.plot(range(3), 100 * accuracies[model_architectures[idx]]["total"][0, :3] / accuracies[model_architectures[idx]]["total"][1, :3], label=arch_labels[idx])
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.set_ylabel("Accuracy")
# ax.legend(loc="upper right", framealpha=1.0, bbox_to_anchor=(1, 1.05))
# fig.tight_layout()
# plt.gcf().savefig("./logs/accuracy_bar_plots.pdf")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_ablation', type=bool)
    args = parser.parse_args()
    print(f"{__file__}, args: {args}")
    if not args.is_ablation:
        plot_accuracy()
    else:
        plot_accuracy_ablation()

if __name__ == "__main__":
    main()

