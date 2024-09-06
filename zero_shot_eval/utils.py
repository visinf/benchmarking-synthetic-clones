import json
import os
import numpy as np
import torch
import random


def report_json(result_dict, path, fn='metrics.csv'):
    os.makedirs(path, exist_ok=True)
    print(f'Writing results to {os.path.join(path, fn)}')
    with open(os.path.join(path, fn), 'w', encoding='utf-8') as fn:
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
    
class ClassificationModel(torch.nn.Module):
    def __init__(self, model, text_embedding, args, input_normalize, resizer=None, logit_scale=True):
        super().__init__()
        self.model = model
        self.args = args
        self.input_normalize = input_normalize
        self.resizer = resizer if resizer is not None else lambda x: x
        self.text_embedding = text_embedding
        self.logit_scale = logit_scale

    def forward(self, vision, output_normalize=True):
        assert output_normalize
        embedding_norm_ = self.model.encode_image(
            self.input_normalize(self.resizer(vision)),
            normalize=True
        )
        logits = embedding_norm_ @ self.text_embedding
        if self.logit_scale:
            logits *= self.model.logit_scale.exp()
        return logits