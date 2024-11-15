###cd 경로 위치 주의###


from PIL import Image
from IPython.display import display
import torch as th
import torchvision.transforms as T
import random

from model_t2i.download import load_checkpoint
from model_t2i.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)


#특정 gpu 사용하기
import os
import torch

import argparse


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
#parser.add_argument('--data_path', default=None, type=dir_path, help='custom_data_dir')
parser.add_argument('--save_dir_free', default='./data_gen/glide/image/free', type=str, help='data_save_path(dir)')
parser.add_argument('--save_dir_cond', default='./data_gen/glide/image/cond', type=str, help='data_save_path(dir)')
parser.add_argument('--save_dir_txt', default='./data_gen/glide/text', type=str, help='data_save_path(txt)')
parser.add_argument('--txt_path', default='./data/text', type=dir_path, help='data_txt_path(dir)')
parser.add_argument('--gpus', default=0, type=int, help='gpu_number')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
#os.environ["CUDA_VISIBLE_DEVICES"]= f"{args.gpus}"  # Set the GPU 2 to use
#os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"  # Set the GPU 2 to use


device = torch.device(f"cuda:{args.gpus}") #무조건 cuda 만 쓰게끔
torch.cuda.set_device(device)
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

# This notebook supports both CPU and GPU.
# On CPU, generating one sample may take on the order of 20 minutes.
# On a GPU, it should be under a minute.

has_cuda = th.cuda.is_available()
#device = th.device('cpu' if not has_cuda else 'cuda')

# Create base model.
options = model_and_diffusion_defaults() 
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))


# Create upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

import numpy as np
def show_images(batch: th.Tensor, name, path):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))
    img = Image.fromarray(reshaped.numpy())
    img.save(os.path.join(path, f'{name}.jpg'))
    print(f'saved image: {name}.jpg')
    
#텍스트 쓰기
def save_tokens(txt_path, prompt):
    with open(os.path.join(txt_path, name+'.txt'), 'a') as f:
        f.write(prompt)
        f.write('\n')
        f.close()


# Create a classifier-free guidance sampling function
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)
  

print('text loading...')
# Sampling parameters
#prompt text list 생성
total_text = []
mini_text = []
for i in sorted(os.listdir(args.txt_path)):
    #print(i)
    with open(os.path.join(args.txt_path, i),'r') as f:
        lines = f.readlines() #이미지 하나당 캡션 10개
    for l in lines:
        #print(l.strip('\n'))
        mini_text.append(l.strip('\n'))
    input_data = (i.split('.')[0], random.choice(mini_text))
    total_text.append(input_data)


print('text prompt input...')
for i, (name, text) in enumerate(total_text): 
    prompt = text
    batch_size = 1
    guidance_scale = 3.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997


    ##############################
    # Sample from the base model #
    ##############################

    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model.del_cache()

    makedir(args.save_dir_free)
    # Show the output
    show_images(samples, name, args.save_dir_free)
    
    #2. condition 이미지 생성 및 저장
    ##############################
    # Upsample the 64x64 samples #
    ##############################

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

    # Sample from the base model.
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model_up.del_cache()

    makedir(args.save_dir_cond)
    # Show the output
    show_images(up_samples, name, args.save_dir_cond)
    makedir(args.save_dir_txt)
    save_tokens(args.save_dir_txt, text)
        
