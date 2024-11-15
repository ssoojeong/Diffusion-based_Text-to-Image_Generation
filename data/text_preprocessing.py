
import os
import argparse
import torch.nn as nn

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='./data/captions.json', type=str, help='custom_data_path')
parser.add_argument('--save_dir', default='./data/text', type=str, help='data_save_path(dir)')
args = parser.parse_args()


import json
with open(args.json_path, 'r') as f:
    data = json.load(f)

path = args.save_dir
for idx, p in enumerate(data['annotations']):
    if not os.path.exists(path):
        os.makedirs(path)

    if p['image_id'] <= 136:
        title = str(p['image_id']).zfill(12)
        with open(os.path.join(path, title+'.txt'), 'a') as f:
            f.write(p['caption'])
            f.write('\n')
            f.close()