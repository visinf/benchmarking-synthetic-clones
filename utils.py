import json
import os
import numpy as np
import torch
import random
import open_clip
from open_clip import get_tokenizer
from transformers import CLIPTokenizer
from pathlib import Path


def report_json(result_dict, path, fn='metrics.csv'):
    os.makedirs(path, exist_ok=True)
    print(f'Writing results to {os.path.join(path, fn)}')
    with open(os.path.join(path, fn), 'w+', encoding='utf-8') as fn:
        json.dump(result_dict, fn, ensure_ascii=False, indent=4)

def seed_everything(seed):
    print(f'Seeding everything with seed: {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def obtain_ImageNet_classes():
    base_path = f'{Path.home()}/benchmarking-synthetic-clones/'
    loc = os.path.join(f'{base_path}/ood_detection/data', 'imagenet')
    with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls

    
def get_text_features(model, dataset_name='imagenet1k', class_labels=[]):
    model = model.eval()
    base_path = f'{Path.home()}/benchmarking-synthetic-clones/'
    with torch.no_grad():
        with open(f'{base_path}/zero_shot_eval/zeroshot-templates.json', 'r') as f:
            templates = json.load(f)
        templates = templates[dataset_name]
        embedding_text_labels_norm = []
        tokenizer = get_tokenizer('ViT-B-16')
        for c in class_labels:
            texts = [template.format(c=c) for template in templates]
            if 'encode_text' in model.__dir__():
                text_tokens = tokenizer(texts).to('cuda')
                class_embeddings = model.encode_text(text_tokens)
                class_embedding = torch.nn.functional.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                embedding_text_labels_norm.append(class_embedding.to())
        text_features = torch.stack(embedding_text_labels_norm, dim=1).to('cuda')
                        

    return text_features


class ClassificationModel(torch.nn.Module):
    def __init__(self, model, text_embedding):
        super().__init__()
        self.model = model
        self.text_features = text_embedding
        self.logit_scale = 100.# value from the checkpoint for scaling clip

    def forward(self, images):
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logits = image_features @ self.text_features
        logits = self.logit_scale*logits

        return logits

