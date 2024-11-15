import cv2
from sewar.full_ref import psnr, ssim

psnr_v = 0.0
ssim_v = 0.0
num = 0

import argparse
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

        
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='설명입니다.')
parser.add_argument('--GT_path', default='./data/image', type=str, help='GT')
parser.add_argument('--model', type=str)
parser.add_argument('--classifier',  action='store_true') #default == Fasle
args = parser.parse_args()
#parser.add_argument('--data_type', default=None, type=str, help='cub') 
if (args.model != 'glide'):
    parser.add_argument('--P_path', type=str, default=f'./data_gen/{args.model}/image')
    parser.add_argument('--save_results', type=str, default=f'./results/{args.model}.txt')
elif (args.classifier):
    parser.add_argument('--P_path', type=str, default=f'./data_gen/{args.model}/image/cond')
    parser.add_argument('--save_results', type=str, default=f'./results/{args.model}_cond.txt')
else:
    parser.add_argument('--P_path', type=str, default=f'./data_gen/{args.model}/image/free')
    parser.add_argument('--save_results', type=str, default=f'./results/{args.model}_free.txt') 

args = parser.parse_args()



path1, path2 = args.GT_path, args.P_path
gt_lists = sorted(os.listdir(path1)[:10000])
p_lists = sorted(os.listdir(path2)[:10000])
for gt, p in zip(gt_lists, p_lists ):
    img1 = cv2.imread(os.path.join(args.GT_path, gt))
    img2 = cv2.imread(os.path.join(args.P_path,p))

    w1, h1 = img1.shape[0], img1.shape[1]
    w2, h2 = img2.shape[0], img2.shape[1]

    w_min, h_min = min(w1, w2), min(h1, h2)
    print(w_min, h_min)

    img1 = cv2.resize(img1, dsize=(h_min, w_min), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, dsize=(h_min, w_min), interpolation=cv2.INTER_AREA)

    psnr_v += psnr(img1, img2)
    ssim_v += ssim(img1, img2)[0]
    
    num+=1

avg_psnr = psnr_v/num
avg_ssim = ssim_v/num
print(f'avg_psnr: {avg_psnr} / avg_ssim: {avg_ssim}')

# Save result to file
with open(args.save_results, "a") as f:
    f.write(f'PSNR: {avg_psnr}')
    f.write('\n')
    f.write(f'SSIM: {avg_ssim}')
    f.write('\n')
