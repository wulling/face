import argparse
import torch
import numpy as np
import sys
import os
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from HFGI.configs import data_configs, paths_config
from HFGI.datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from HFGI.utils.model_utils import setup_model
from HFGI.utils.common import tensor2im
from PIL import Image
from HFGI.editings import latent_editor

device = "cuda"

def process(img):
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    imgs = tf(img)
    return imgs


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


# def get_all_latents(net, data_loader, n_images=None, is_cars=False):
#     all_latents = []
#     i = 0
#     with torch.no_grad():
#         for batch in data_loader:
#             if n_images is not None and i > n_images:
#                 break
#             x = batch
#             inputs = x.to(device).float()
#             latents = get_latents(net, inputs, is_cars)
#             all_latents.append(latents)
#             i += len(latents)
#     return torch.cat(all_latents)


# def save_image(img, save_dir, idx):
#     result = tensor2im(img)
#     im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
#     Image.fromarray(np.array(result)).save(im_save_path)

def edit(img_path,attribute):
    parser = argparse.ArgumentParser(description="Inference")
    # parser.add_argument("--images_dir", type=str, default="", help="The directory to the images")
    # parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
    # parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    # parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    # parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
    parser.add_argument("--edit_degree", type=float, default=3, help="edit degreee")
    parser.add_argument("--ckpt", type=str, default="HFGI/checkpoint/ckpt.pt", help="path to generator checkpoint")


    args = parser.parse_args()
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'car' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    aligner = net.grid_align
    editor = latent_editor.LatentEditor(net.decoder, is_cars)

    # initial inversion


    # set the editing operation
    if attribute == 'inversion':
        pass
    elif attribute == 'age' or attribute == 'smile':
        interfacegan_directions = {
            'age': 'HFGI/editings/interfacegan_directions/age.pt',
            'smile': 'HFGI/editings/interfacegan_directions/smile.pt'}
        edit_direction = torch.load(interfacegan_directions[attribute]).to(device)
    else:
        ganspace_pca = torch.load('HFGI/editings/ganspace_pca/ffhq_pca.pt')
        ganspace_directions = {
            'eyes': (54, 7, 8, 20),
            'beard': (58, 7, 9, -20),
            'lip': (34, 10, 11, 20)}
        edit_direction = ganspace_directions[attribute]

    image = Image.open(img_path).convert('RGB')
    # img = np.array(image)
    # perform high-fidelity inversion or editing

    pro_img = process(image)
    pro_imgs = pro_img.unsqueeze(0)
    # print(pro_imgs.shape)
    # print(type(pro_img))
    x = pro_imgs.to(device).float()

    latent_codes = get_latents(net, x, is_cars=is_cars)
        # calculate the distortion map 计算失真图
    imgs, _ = generator([latent_codes.to(device)], None, input_is_latent=True,
                            randomize_noise=False, return_latents=True)
        # test_5_16_img_fake = imgs
        # result = tensor2im(test_5_16_img_fake[0])
        # im_save_path = os.path.join('./img', f"{i:05d}.jpg")
        # Image.fromarray(np.array(result)).save(im_save_path)
    res = x - torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256, 256), mode='bilinear')

        # produce initial editing image
        # edit_latents = editor.apply_interfacegan(latent_codes[i].to(device), interfacegan_direction, factor_range=np.linspace(-3, 3, num=40))
    if attribute == 'inversion':
        img_edit = imgs
        edit_latents = latent_codes.to(device)
    elif attribute == 'age' or attribute == 'smile':
        img_edit, edit_latents = editor.apply_interfacegan(latent_codes.to(device), edit_direction,
                                                               factor=args.edit_degree)
    else:
        img_edit, edit_latents = editor.apply_ganspace(latent_codes.to(device), ganspace_pca,
                                                           [edit_direction])

        # align the distortion map  #对齐失真图
    img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=(256, 256), mode='bilinear')
    res_align = net.grid_align(torch.cat((res, img_edit), 1))
        # test_5_16_img_res =  res_align
        # result = tensor2im(test_5_16_img_res[0])
        # im_save_path = os.path.join('./res', f"{i:05d}.jpg")
        # Image.fromarray(np.array(result)).save(im_save_path)

        #####  torch.clamp将输入input张量每个元素的范围限制到区间
        # consultation fusion
    conditions = net.residue(res_align)
    imgs, _ = generator([edit_latents], conditions, input_is_latent=True, randomize_noise=False,
                            return_latents=True)
    if is_cars:
        imgs = imgs[:, :, 64:448, :]

        # save images
    imgs = torch.nn.functional.interpolate(imgs, size=(256, 256), mode='bilinear')
    result = tensor2im(imgs[0])

    return result