from torchmetrics.functional.multimodal import clip_score
from functools import partial
import os
import cv2
import torch
import numpy as np

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

import argparse
parser = argparse.ArgumentParser(description='설명입니다.')
parser.add_argument("-c", "--gpus", default="", type=str, help="GPU to use (leave blank for CPU only)")
parser.add_argument('--model', type=str)
parser.add_argument('--classifier',  action='store_true') #default == Fasle
args = parser.parse_args()
#parser.add_argument('--data_type', default=None, type=str, help='cub') 
if (args.model != 'glide'):
    parser.add_argument('--image_path', type=str, default=f'./data_gen/{args.model}/image')
    parser.add_argument('--save_results', type=str, default=f'./results/{args.model}.txt')
elif (args.classifier):
    parser.add_argument('--image_path', type=str, default=f'./data_gen/{args.model}/image/cond')
    parser.add_argument('--save_results', type=str, default=f'./results/{args.model}_cond.txt')
else:
    parser.add_argument('--image_path', type=str, default=f'./data_gen/{args.model}/image/free')
    parser.add_argument('--save_results', type=str, default=f'./results/{args.model}_free.txt')
parser.add_argument('--txt_path', default=f'./data_gen/{args.model}/text', type=dir_path)
 
args = parser.parse_args()

'''
import logging
logger = logging.getLogger('coco')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename=f'coco_clip_{args.data_type}.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)
'''

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

device = 'cuda'
if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach().to(device)
    return round(float(clip_score), 4)


captions_list = os.listdir(os.path.join(args.txt_path))
images_list = os.listdir(os.path.join(args.image_path))

captions_list = sorted(captions_list) #정렬 필수!!!
images_list = sorted(images_list)


clip_score=0.0
std_list = []
for i, _ in enumerate(captions_list):
    with open(os.path.join(args.txt_path, captions_list[i]), 'r') as f:
        prompts = f.readlines()
    images = cv2.imread(os.path.join(args.image_path, images_list[i]))
    w, h, c = images.shape[0], images.shape[1], images.shape[2]
    images = images.reshape(1, w, h, c) #images.shape

    sd_clip_score = calculate_clip_score(images, prompts[0][:200]) #prompt 개수에 영향?
    clip_score += sd_clip_score
    print(f"{i}_CLIP score: {sd_clip_score}")
    #logger.info(f"{i}_CLIP score: {sd_clip_score}")
    # CLIP score: 35.7038'''
    std_list.append(sd_clip_score)

avg_clip_score = clip_score/len(captions_list)
std_clip_score = np.std(std_list)
print(f'average CLIP score: {avg_clip_score}, std CLIP score: {std_clip_score}')

# Save result to file
with open(args.save_results, "a") as f:
    f.write(f"CLIP score: {avg_clip_score}, std CLIP score: {std_clip_score}")
    f.write('\n')