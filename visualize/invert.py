# Copyright (C) 2022 ByteDance Inc.
# All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# The software is made available under Creative Commons BY-NC-SA 4.0 license
# by ByteDance Inc. You can use, redistribute, and adapt it
# for non-commercial purposes, as long as you (a) give appropriate credit
# by citing our paper, (b) indicate any changes that you've made,
# and (c) distribute any derivative works under the same license.

# THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import os
from pickletools import uint8
import sys
import shutil
import math
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image
from imageio import imwrite, mimwrite
import cv2 as cv
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
sys.path.insert(1, os.getcwd())
from criteria.lpips import lpips
from models import make_model
from visualize.utils import tensor2image, tensor2seg

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def get_transformation(args):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                ])
    return transform

def calc_lpips_loss(im1, im2):
    img_gen_resize = F.adaptive_avg_pool2d(im1, (256,256))
    target_img_tensor_resize = F.adaptive_avg_pool2d(im2, (256,256))
    p_loss = percept(img_gen_resize, target_img_tensor_resize).mean()
    return p_loss

def optimize_latent(args, g_ema, target_img_tensor, target_seg_tensor=None, 
                    latent_in=None, latent_mean=None, noises=None):

    if noises == None:
        noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
                    
    for noise in noises:
        noise.requires_grad = True
        
    # initialization
    if latent_mean == None or latent_in == None or noises == None:
        with torch.no_grad():
            noise_sample = torch.randn(10000, 512, device=device)
            latent_mean = g_ema.style(noise_sample).mean(0)
            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(args.batch_size, 1)
            if args.w_plus:
                latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    latent_in.requires_grad = True

    if args.no_noises:
        optimizer = optim.Adam([latent_in], lr=args.lr)
    else:
        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)


    mask = 0
    # glasses, eyes, eyebrows mask + dilation
    
    if target_seg_tensor != None:
        target = target_seg_tensor.squeeze().cpu().detach()
        target_im = torch.permute(target_img_tensor.squeeze().cpu().detach(),(1, 2, 0))
        #print(target_im.shape)
        #print(target.shape)
        mask = np.zeros_like(target, dtype=np.uint8)
        w = np.full(13, 1) #13 no frames, 14 with frames, 15 with 2 types of lenses
        
        for i in [2,10]: #2,10,13
            mask[target==i] = 255
        for i in [2,3,5,10]: #2,3,5,10,13
            w[i] = 5
        w[10] = 10
        #w[13] = 15
        mask = np.asarray(cv.dilate(mask, kernel=np.ones((5, 5), np.uint8), iterations=10), dtype=bool)
        imwrite(os.path.join(args.outdir, f'{image_basename}_seg_mask.png'), mask*target.numpy())
        l = np.dstack([mask,mask,mask])*target_im.numpy()
        #print(l.min(), l.max(), l.dtype, l[30,30])
        imwrite(os.path.join(args.outdir, f'{image_basename}_im_mask.png'), l)

        mask = torch.from_numpy(mask).unsqueeze(0).to(device)

        #print("target:", target_seg_tensor.size(), target_seg_tensor.max(), target_seg_tensor.min())
        #w = np.zeros(13)
        #for i in range(len(w)):
        #    w[i] = (args.size*args.size)/(sum(target_seg_tensor[target_seg_tensor==i])+1)
        #w[w<1] = w[w<1]/w.min() 

        #print(w)
        seg_weights = torch.tensor(w, dtype=torch.float, device=device)

    latent_path = [latent_in.detach().clone()]
    pbar = tqdm(range(args.step))
    for i in pbar:
        seg_loss = 0
        seg_mask_loss = 0
        mask_loss = 0
        optimizer.param_groups[0]['lr'] = get_lr(float(i)/args.step, args.lr)
        
        img_gen, seg_gen = g_ema([latent_in], input_is_latent=True, randomize_noise=False, noise=noises)
        
        #print(img_gen.min(), img_gen.max())
        #a = tensor2image(img_gen.detach().cpu()*((1-target)*-1)).squeeze()
        #print(a.min(), a.max())
        #imwrite(os.path.join(args.outdir, f'{image_basename}_masked_im.png'), a)

        p_loss = calc_lpips_loss(img_gen, target_img_tensor)
        mse_loss = F.mse_loss(img_gen, target_img_tensor)
        n_loss = torch.mean(torch.stack([noise.pow(2).mean() for noise in noises]))


        if args.w_plus == True:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.unsqueeze(0).repeat(latent_in.size(0), g_ema.n_latent, 1))
        else:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.repeat(latent_in.size(0), 1))

        if target_seg_tensor != None and i<len(pbar)/2:
            seg_loss = F.cross_entropy(seg_gen, target_seg_tensor, weight=seg_weights)
            seg_mask_loss = F.cross_entropy(seg_gen*mask, target_seg_tensor*mask, weight=seg_weights)
            mask_loss = F.mse_loss(img_gen*mask, target_img_tensor*mask)
        if target_seg_tensor != None and i<len(pbar)/2:
            # main loss function
            loss = (n_loss * args.noise_regularize + 
                    p_loss * args.lambda_lpips + 
                    mse_loss * args.lambda_mse + 
                    latent_mean_loss * args.lambda_mean + 
                    seg_loss * args.lambda_seg + 
                    seg_mask_loss * args.lambda_segmask +
                    mask_loss * args.lambda_mask)
        else:
            loss = (n_loss * args.noise_regularize + 
                    p_loss * args.lambda_lpips + 
                    mse_loss * args.lambda_mse + 
                    latent_mean_loss * args.lambda_mean + 
                    seg_loss * args.lambda_seg/5 + 
                    seg_mask_loss * args.lambda_segmask/5 +
                    mask_loss * args.lambda_mask)

        if args.segdir and target_seg_tensor != None and i<len(pbar)/2:
            pbar.set_description(f'perc: {p_loss.item():.4f} noise: {n_loss.item():.4f} mse: {mse_loss.item():.4f} latent: {latent_mean_loss.item():.4f} seg: {seg_loss.item():.4f} mask: {mask_loss.item():.4f}')
        else:
            pbar.set_description(f'perc: {p_loss.item():.4f} noise: {n_loss.item():.4f} mse: {mse_loss.item():.4f} latent: {latent_mean_loss.item():.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # noise_normalize_(noises)
        latent_path.append(latent_in.detach().clone())
    
    #imwrite(os.path.join(args.outdir, f'{image_basename}_masked_im.png'), tensor2image(img_gen.detach()).squeeze()*target.numpy())
    return latent_path, noises, latent_mean


def optimize_weights(args, g_ema, target_img_tensor, latent_in, noises=None):

    for p in g_ema.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(g_ema.parameters(), lr=args.lr_g)

    pbar = tqdm(range(args.finetune_step))
    for i in pbar:        
        img_gen, _ = g_ema([latent_in], input_is_latent=True, randomize_noise=False, noise=noises)

        p_loss = calc_lpips_loss(img_gen, target_img_tensor)
        mse_loss = F.mse_loss(img_gen, target_img_tensor)

        # main loss function
        loss = (p_loss * args.lambda_lpips +
                mse_loss * args.lambda_mse
        )

        pbar.set_description(f'perc: {p_loss.item():.4f} mse: {mse_loss.item():.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return g_ema


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parse_boolean = lambda x: not x in ["False","false","0"]
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--imgdir', type=str, required=True)
    parser.add_argument('--segdir', type=str, default=None)
    parser.add_argument('--outdir', type=str, required=True)

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--no_noises', type=parse_boolean, default=True)
    parser.add_argument('--w_plus', type=parse_boolean, default=True, help='optimize in w+ space, otherwise w space')

    parser.add_argument('--save_steps', type=parse_boolean, default=False, help='if to save intermediate optimization results')

    parser.add_argument('--truncation', type=float, default=1, help='truncation tricky, trade-off between quality and diversity')

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--step', type=int, default=400, help='latent optimization steps')
    parser.add_argument('--finetune_step', type=int, default=0, help='pivotal tuning inversion (PTI) steps (200-400 should give good result)')
    parser.add_argument('--noise_regularize', type=float, default=10)
    parser.add_argument('--lambda_mse', type=float, default=0.1)
    parser.add_argument('--lambda_lpips', type=float, default=1.0)
    parser.add_argument('--lambda_mean', type=float, default=1.0)
    #added losses
    parser.add_argument('--lambda_mask', type=float, default=1.0)
    parser.add_argument('--lambda_seg', type=float, default=1.0) #7
    parser.add_argument('--lambda_segmask', type=float, default=1.0) #12

    args = parser.parse_args()
    print(args)

    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    g_ema = make_model(ckpt['args'])
    g_ema.to(device)
    g_ema.eval()
    g_ema.load_state_dict(ckpt['g_ema'])

    percept = lpips.LPIPS(net_type='vgg').to(device)
    

    img_list = sorted(os.listdir(args.imgdir))
    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(os.path.join(args.outdir, 'recon'), exist_ok=True)
    if args.finetune_step > 0:
        os.makedirs(os.path.join(args.outdir, 'recon_finetune'), exist_ok=True)
    if args.save_steps:
        os.makedirs(os.path.join(args.outdir, 'steps'), exist_ok=True)
        
    os.makedirs(os.path.join(args.outdir, 'latent'), exist_ok=True)
    if not args.no_noises:
        os.makedirs(os.path.join(args.outdir, 'noise'), exist_ok=True)
    if args.finetune_step > 0:
        os.makedirs(os.path.join(args.outdir, 'weights'), exist_ok=True)

    transform_im = get_transformation(args)

    for image_name in img_list:
        image_basename = os.path.splitext(image_name)[0]
        img_path = os.path.join(args.imgdir, image_name)

        # Reload the model
        if args.finetune_step > 0:
            g_ema.load_state_dict(ckpt['g_ema'], strict=True)
            g_ema.eval()

        # load target image
        target_pil = Image.open(img_path).convert('RGB').resize((args.size,args.size), resample=Image.LANCZOS)
        target_img_tensor = transform_im(target_pil).unsqueeze(0).to(device)
        target_seg_tensor = None
        if args.segdir:
            seg_path = os.path.join(args.segdir, f'{image_basename}.png')
            target_seg = np.array(Image.open(seg_path).resize((args.size,args.size), resample=Image.NEAREST))
            target_seg_tensor = torch.as_tensor(target_seg, dtype=torch.int64).unsqueeze(0).to(device)


        latent_path, noises, latent_m = optimize_latent(args, g_ema, target_img_tensor, target_seg_tensor)
        

        # save results
        with torch.no_grad():
            img_gen, seg_gen = g_ema([latent_path[-1]], input_is_latent=True, randomize_noise=False, noise=noises)
            # Image

            img_gen = tensor2image(img_gen).squeeze()
            imwrite(os.path.join(args.outdir, 'recon/', f'{image_basename}.jpg'), img_gen)

            # Segmentation
            seg_gen = tensor2seg(seg_gen).squeeze()
            imwrite(os.path.join(args.outdir, 'recon/', f'{image_basename}.png'), seg_gen)

            # Latents
            latent_np = latent_path[-1].detach().cpu().numpy()
            np.save(os.path.join(args.outdir, 'latent/', f'{image_basename}.npy'), latent_np)
            if not args.no_noises:
                for i in range(len(noises)):
                    #noises_np = torch.stack(noises[i], dim=1).detach().cpu().numpy()
                    noises_np = noises[i].detach().cpu().numpy()
                    np.save(os.path.join(args.outdir, 'noise/', f'{image_basename}_{i}.npy'), noises_np)


            if args.save_steps:
                total_steps = args.step
                images = []
                masks = []
                for i in range(0, total_steps, 10):
                    img_gen, seg_gen = g_ema([latent_path[i]], input_is_latent=True, randomize_noise=False, noise=noises)
                    img_gen = tensor2image(img_gen).squeeze()
                    seg_gen = tensor2seg(seg_gen).squeeze()
                    images.append(img_gen)
                    masks.append(seg_gen)
                mimwrite(os.path.join(args.outdir, 'steps/', f'{image_basename}_im.mp4'), images, fps=10)
                mimwrite(os.path.join(args.outdir, 'steps/', f'{image_basename}_seg.mp4'), masks, fps=10)

        if args.finetune_step > 0:
            g_ema = optimize_weights(args, g_ema, target_img_tensor, latent_path[-1], noises=noises)
            with torch.no_grad():
                img_gen, seg_gen = g_ema([latent_path[-1]], input_is_latent=True, randomize_noise=False, noise=noises)
                img_gen = tensor2image(img_gen).squeeze()
                seg_gen = tensor2seg(seg_gen).squeeze()

                imwrite(os.path.join(args.outdir, 'recon_finetune/', f'{image_basename}.jpg'), img_gen)
                imwrite(os.path.join(args.outdir, 'recon_finetune/', f'{image_basename}.png'), seg_gen)

                # Weights
                image_basename = os.path.splitext(image_name)[0]
                ckpt_new = {"g_ema": g_ema.state_dict(), "args": ckpt["args"]}
                torch.save(ckpt_new, os.path.join(args.outdir, 'weights/', f'{image_basename}.pt'))

        
        # # 2

        # latent_path, noises, _ = optimize_latent(args, g_ema, target_img_tensor, \
        #                                          latent_in=latent_path[-1], latent_mean=latent_m, noises=noises)

        # # save results
        # with torch.no_grad():
        #     img_gen, seg_gen = g_ema([latent_path[-1]], input_is_latent=True, randomize_noise=False, noise=noises)
        #     # Image
        #     img_gen = tensor2image(img_gen).squeeze()
        #     imwrite(os.path.join(args.outdir, 'recon/', f'{image_basename}_2.jpg'), img_gen)
        #     # Segmentation
        #     seg_gen = tensor2seg(seg_gen).squeeze()
        #     imwrite(os.path.join(args.outdir, 'recon/', f'{image_basename}_2.png'), seg_gen)
        #     # Latents
        #     latent_np = latent_path[-1].detach().cpu().numpy()
        #     np.save(os.path.join(args.outdir, 'latent/', f'{image_basename}_2.npy'), latent_np)
        #     if not args.no_noises:
        #         for i in range(len(noises)):
        #             #noises_np = torch.stack(noises[i], dim=1).detach().cpu().numpy()
        #             noises_np = noises[i].detach().cpu().numpy()
        #             np.save(os.path.join(args.outdir, 'noise/', f'{image_basename}_{i}.npy'), noises_np)


        #     if args.save_steps:
        #         total_steps = args.step
        #         images = []
        #         masks = []
        #         for i in range(0, total_steps, 10):
        #             img_gen, seg_gen = g_ema([latent_path[i]], input_is_latent=True, randomize_noise=False, noise=noises)
        #             img_gen = tensor2image(img_gen).squeeze()
        #             seg_gen = tensor2seg(seg_gen).squeeze()
        #             images.append(img_gen)
        #             masks.append(seg_gen)
        #         mimwrite(os.path.join(args.outdir, 'steps/', f'{image_basename}_im_2.mp4'), images, fps=10)
        #         mimwrite(os.path.join(args.outdir, 'steps/', f'{image_basename}_seg_2.mp4'), masks, fps=10)