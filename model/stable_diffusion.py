import torch
from diffusers import StableDiffusionPipeline

#특정 gpu 사용하기
import os
import random
import argparse

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir_image', default='./data_gen/stable/image', type=str, help='data_save_path(dir)')
parser.add_argument('--save_dir_txt', default='./data_gen/stable/text', type=str, help='data_save_path(dir)')
parser.add_argument('--txt_path', default='./data/text', type=str, help='data_txt_path(dir)')
parser.add_argument('--gpus', default=0, type=int, help='gpu_number')
args = parser.parse_args()

mkdir(args.save_dir_image)
mkdir(args.save_dir_txt)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
#os.environ["CUDA_VISIBLE_DEVICES"]= f"{args.gpus}"  # Set the GPU 2 to use

device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())



model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)



GENERATED_IMAGE_DIR = args.save_dir_image
mkdir(GENERATED_IMAGE_DIR)
mkdir(args.save_dir_txt)

total_text = []
for i in sorted(os.listdir(args.txt_path)):
    #print(i)
    with open(os.path.join(args.txt_path, i),'r') as f:
        lines = f.readlines() #이미지 하나당 캡션 10개
    mini_text = []
    for l in lines:
        #print(l.strip('\n'))
        mini_text.append(l.strip('\n'))
    input_data = (i.split('.')[0], random.choice(mini_text))
    total_text.append(input_data)



for i, (name, word) in enumerate(total_text):
    with open(os.path.join(args.save_dir_txt, name+'.txt'), 'a') as f:
        f.write(word)
        f.write('\n')
        f.close()
    #이미지 로드 및 저장
    caption_id = name +'.jpg'
    generated_image = pipe(word).images[0]
    generated_image.save(f'{GENERATED_IMAGE_DIR}/{caption_id}')

print('이미지 생성 끝!')

