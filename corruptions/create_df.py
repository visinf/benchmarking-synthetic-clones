import numpy as np
import pandas as pd
import glob
import argparse
import json
import os
from collections import defaultdict

tab_first = (1, 0.7, 0.7)
tab_second = (1, 0.85, 0.7)
tab_third = (1, 1, 0.7)

def highlight_top3(s):
    result = []
    is_large = sorted(s.nlargest(3).values)
    colors = [tab_first, tab_second, tab_third]
    idx = 0 
    for i in s:
        if i in is_large:
            result.append('\cellcolor[rgb]{'+str(colors[idx])[1:-1]+'} '+str(round(i*100, 2)))
            idx+=1
        else:
            result.append(i*100)
    return result
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    return args

def load_files(inp_dir, out_path):
    ablation = False
    if not ablation:
        files = glob.glob(f'{inp_dir}/2DCC/*.json')
        files.extend(glob.glob(f'{inp_dir}/3DCC/*.json'))
    else:
        files = glob.glob(f'{inp_dir}/2DCC/ablations/*.json')

    data_dict = []
    corruption_dict = defaultdict()
    all_cc = []
    for file in files:
        if file.split('/')[-1].startswith('rel_plot'):
            continue
        with open(file, 'r') as fn:
            if ablation:
                num_imgs = file.split('.')[1].split('_')[-1]
                prompt_type = file.split('.')[1].split('_')[-2]
            content = fn.read()
            content_dict = json.loads(content)
            for contents in content_dict:
                # print(contents)
                if contents['model'] in corruption_dict.keys():
                    if contents['corruption'] in corruption_dict[contents['model']].keys():
                        corruption_dict[contents['model']][contents['corruption']] += contents['acc'] 
                    else:
                        corruption_dict[contents['model']].update({contents['corruption']: contents['acc']})
                else:
                    corruption_dict.update({contents['model']: {contents['corruption']: contents['acc']}})
            if ablation:
                corruption_dict[contents['model']].update({
                    'num_imgs': num_imgs,
                    'prompt_type': prompt_type
                })
        print('Here...')
        if ablation:
            print(f"{__file__}, corruption_dict: {corruption_dict}")
            for model_names in corruption_dict.keys():
                for corruption in corruption_dict[model_names]:
                    if corruption not in ['num_imgs', 'prompt_type']:
                        corruption_dict[model_names][corruption] /= 5

            empty_dict = {'model': contents['model']}
            for k, v in corruption_dict[contents['model']].items():
                empty_dict.update({k:v})
            all_cc.append(empty_dict)
            corruption_dict = {}
    

    for model_names in corruption_dict.keys():
        for corruption in corruption_dict[model_names]:
            corruption_dict[model_names][corruption] /= 5

    if not ablation:
        data_df = pd.DataFrame.from_dict(corruption_dict)
    else:
        data_df = pd.DataFrame.from_dict(all_cc)
    os.makedirs(out_path, exist_ok=True)
    if ablation:
        data_df['Avg'] = data_df[['shot_noise', 'motion_blur', 'snow', 'pixelate', 'jpeg_compression']].mean(axis=1)
    import ipdb; ipdb.set_trace()    

    # if 'model_name' not in data_df.columns:
    #     data_df['model'] = rename_fn(data_df['model'])
    # else:
    #     data_df['model'] = rename_fn(data_df['model_name'])
    if not ablation:
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
        data_df.rename(mapper=rename_dict, inplace=True, axis=1)
        data_df = data_df[['Resnet50', 'Resnet50 (Syn.)', 'Swin-B', 'ConvNext', 'DeiT', 'ViT-B',  'ViT-B (Syn.)', 'MAE', 'DINOv2', 'MOCOv3', 'SimCLR (Syn.)', 'CLIP', 'CLIP (Syn.)']]
        data_df.loc['Average Acc'] = data_df.mean()
        data_df = data_df.reset_index()
        out_path = f'{out_path}/table.tex'

    else:
        data_df = data_df[['model', 'num_imgs', 'prompt_type', 'Avg']]
        out_path = f'{out_path}/ablation.tex'
   # data_df = sortDF(data_df, ['Resnet50', 'Resnet50 (Syn.)', 'Swin-B', 
    #                            'ConvNext', 'DeiT', 'ViT-B', 
    #                            'ViT-B (Syn.)', 'MAE', 'DINOv2', 
    #                            'MOCOv3', 'SimCLR (Syn.)', 'CLIP', 'CLIP (Syn.)'], 'model')
    data_df.to_latex(out_path, index=False, float_format="{:.2f}".format)

def main():
    args = parse_args()
    load_files(args.inp_dir, args.out_path)

if __name__ == "__main__":
    main()






0.39 & 0.42 & 0.43 & 0.42 & 0.37 & 0.43 & 0.44 & 0.44 & 0.44 & 0.51 & 0.54 & 0.54



0.44 & 0.46 & 0.51 & 0.52 & 0.29 & 0.28 & 0.31 & 0.31 & 0.46 & 0.48 & 0.52 & 0.52