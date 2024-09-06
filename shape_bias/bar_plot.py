import matplotlib.pyplot as plt
import json

path =  '/visinf/home/ksingh/syn-rep-learn/shape_bias/results/'
# fns = [f'{path}/resnet50.json', f'{path}/scaling_imagenet_sup.json', f'{path}/vit-b.json', f'{path}/syn_clone.json',
#        f'{path}/mocov3.json', f'{path}/scaling_clip.json',f'{path}/CLIP.json', f'{path}/synclr.json', 
#        ]
fns = [f'{path}/scaling_clip.json',f'{path}/CLIP.json']

result_dict = []
for fn in fns:
    with open(fn, 'r') as f:
        data_dict = json.load(f)
        result_dict.append(
            {
                'model_name': data_dict['model_name'],
                'shape_bias': data_dict['shape_bias'],
                'shape_bias_per_cat': data_dict['shape_bias_per_cat'],
            }
        )


fig, ax = plt.subplots()
