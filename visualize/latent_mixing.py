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
import sys
import argparse
import shutil
import numpy as np
import imageio
import torch
sys.path.insert(1, os.getcwd())
from models import make_model
from visualize.utils import generate, cubic_spline_interpolate
from visualize.utils import tensor2image, tensor2seg

latent_dict_celeba = {
    2:  "bcg_1",
    3:  "bcg_2",
    4:  "face_shape",
    5:  "face_texture",
    6:  "eye_shape",
    7:  "eye_texture",
    8:  "eyebrow_shape",
    9:  "eyebrow_texture",
    10: "mouth_shape",
    11: "mouth_texture",
    12: "nose_shape",
    13: "nose_texture",
    14: "ear_shape",
    15: "ear_texture",
    16: "hair_shape",
    17: "hair_texture",
    18: "neck_shape",
    19: "neck_texture",
    20: "cloth_shape",
    21: "cloth_texture",
    22: "glasses_frames_shape",
    23: "glasses_frames_texture",
    24: "hat",
    26: "earing",
    28: "glasses_lens_shape",
    29: "glasses_lens_texture",
    30: "glasses_sunlens_shape",
    31: "glasses_sunlens_texture",
    0:  "coarse_1",
    1:  "coarse_2",
}

def read_noises(latent_path, noise_dir_path):
    noises = []
    latent_name = os.path.splitext(latent_path)[0].split("/")[-1]
    for n in sorted(os.listdir(noise_dir_path)):
        if n.endswith(".npy") and n.startswith(latent_name):
            noise = torch.tensor(np.load(os.path.join(noise_dir_path,n)), device=args.device)
            noises.append(noise)
    return noises

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('ckpt', type=str, help="path to the model checkpoint")
    parser.add_argument('--latent', type=str, default=None,
        help="path to the latent numpy")
    parser.add_argument('--latent_style', type=str, default=None,
        help="path to the latent style numpy")
    parser.add_argument('--noise', type=str, default=None,
        help="path to the noises dir with numpys")
    parser.add_argument('--outdir', type=str, default='./results/style_mixing/', 
        help="path to the output directory")
    parser.add_argument("--dataset_name", type=str, default="celeba",
        help="used for finding mapping between latent indices and names")
    parser.add_argument('--device', type=str, default="cuda", 
        help="running device for inference")
    args = parser.parse_args()

    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    g_ema = make_model(ckpt['args'])
    g_ema.to(args.device)
    g_ema.eval()
    g_ema.load_state_dict(ckpt['g_ema'])

    assert os.path.exists(args.noise)
    noises = read_noises(args.latent, args.noise)
    #mean_latent = model.style(torch.randn(args.truncation_mean, model.style_dim, device=args.device)).mean(0)
    
    print("Generating original image ...")
    with torch.no_grad():
        assert args.latent != None
        style = torch.tensor(np.load(args.latent), device=args.device)
        image, seg = g_ema([style], input_is_latent=True, randomize_noise=False, noise=noises)
        image = tensor2image(image).squeeze()
        seg = tensor2seg(seg).squeeze()
        imageio.imwrite(f'{args.outdir}/image.jpeg', image)
        imageio.imwrite(f'{args.outdir}/seg.jpeg', seg)

    print("Generating styled image ...")
    if args.dataset_name == "celeba":
        latent_dict = latent_dict_celeba
    else:
        raise ValueError("Unknown dataset name: f{args.dataset_name}")

    with torch.no_grad():
        assert args.latent_style != None
        style_mix = torch.tensor(np.load(args.latent_style), device=args.device)

        for latent_index, latent_name in latent_dict.items():
            if 'glasses_frames' in latent_name or 'glasses_lens' in latent_name:
                try:
                    style_new = style
                    style_new[:,latent_index] = style_mix[:,latent_index]
                    image, seg = g_ema([style_new], input_is_latent=True, randomize_noise=False, noise=noises)
                    #images, segs = generate(model, style_new, mean_latent=mean_latent, 
                    #                        randomize_noise=False)
                    image = tensor2image(image).squeeze()
                    seg = tensor2seg(seg).squeeze()
                    imageio.imwrite(f'{args.outdir}/{latent_index:02d}_{latent_name}_image.jpeg', image)
                    imageio.imwrite(f'{args.outdir}/{latent_index:02d}_{latent_name}_seg.jpeg', seg)
                except IndexError as ie:
                    print(ie)
                    print("{} not in this model".format(latent_name))