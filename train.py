from argparse import ArgumentParser
import random
import string
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import load_data, IMG_SIZE
from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import torch
from torch import nn
import torch.optim as optim
from transformers import AutoTokenizer, BertModel
from blocks import UNet
from diffusion import DiffusionModel, DiffusionModelConfig

DEVICE = 'cuda'
EPOCHS = 600
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 20
NUM_THREAD = 4

'''CosineAnnealingLR'''
LR = 2e-4

'''CyclicLR-triangular2'''
#LR_min = 1e-5
#LR_max = 1e-2


def collate_fn(samples):
    images = []
    img_names = []
    letters = []
    impressions = []
    for sample in samples:
        images.append(sample['image'])
        img_names.append(sample['img_name'])
        letters.append(sample['letter'])
        impressions.append(sample['impression'])
    images = torch.stack(images)
    return {'images': images, 'letters': letters, 'impressions': impressions, 'img_name': img_names}

def load_dataloader():
    data_train, data_test = load_data()
    train_dataloader = DataLoader(data_train, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_THREAD, shuffle=True)
    test_dataloader = DataLoader(data_test, batch_size=EVAL_BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_THREAD, shuffle=True)
    return train_dataloader, test_dataloader

def initialize_weights(m):
    if hasattr(m, 'weight'):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

class Main:
    def __init__(self):
        self.train_dataloader, self.test_dataloader = load_dataloader()
        unet = UNet(1, 128, (1, 2, 4, 8))
        model = DiffusionModel(
            nn_module=unet,
            input_shape=(1, IMG_SIZE[0], IMG_SIZE[1]),
            config=DiffusionModelConfig(
                num_timesteps=1000,
                target_type="pred_x_0", # target_type="pred_eps",
                gamma_type="ddim", # ddpm
                noise_schedule_type="cosine",
            ),
        )
        model.apply(initialize_weights)
        self.model = model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=False)
        self.text_encoder = BertModel.from_pretrained('google-bert/bert-base-uncased').to(DEVICE)
        self.text_encoder.requires_grad_(False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        if not os.path.exists('outputs'):
            os.makedirs('outputs')


    def save_model(self, epoch):
        if not os.path.exists('weights'):
            os.makedirs('weights')
        torch.save(self.model.state_dict(), f'weights/font_dif-{epoch}.model')

    def train(self, start_epoch=0):
        if start_epoch > 0:
            checkpoint = torch.load(f'weights/font_dif-{start_epoch}.model')
            self.model.load_state_dict(checkpoint)
            print(f'Loading pretrained model on Epoch {start_epoch}')
        for epoch in range(start_epoch, EPOCHS):
            total_loss = []
            for i, batch in tqdm(enumerate(self.train_dataloader)):
                images = batch['images'].to(DEVICE)
                letters = batch['letters']
                impressions = batch['impressions']
                res_let = self.tokenizer(letters, padding=True, truncation=True, return_tensors='pt')
                res_imp = self.tokenizer(impressions, padding=True, truncation=True, return_tensors='pt')
                let_ids, let_masks = res_let.input_ids.to(DEVICE), res_let.attention_mask.to(DEVICE)
                imp_ids, imp_masks = res_imp.input_ids.to(DEVICE), res_imp.attention_mask.to(DEVICE)
                let_feats = self.text_encoder(let_ids, let_masks).last_hidden_state # b, len, 768
                imp_feats = self.text_encoder(imp_ids, imp_masks).last_hidden_state # b, len, 768
                self.optimizer.zero_grad()
                loss = self.model.loss(images, [let_feats, imp_feats]).mean()
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
            if (epoch+1) % 10 == 0:
                self.save_model(epoch+1) 
            print(f'Epoch {epoch+1}, loss: {np.mean(total_loss):.3f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
            self.scheduler.step()
            self.eval(epoch+1)

    def _eval_uno(self, impression, epoch, timesteps=None, load_from_file=False):
        if load_from_file:
            checkpoint = torch.load(f'weights/font_dif-{epoch}.model')
            self.model.load_state_dict(checkpoint)

        letters = list(string.ascii_uppercase)
        impressions = [impression] * len(letters)

        res_let = self.tokenizer(letters, padding=True, truncation=True, return_tensors='pt')
        res_imp = self.tokenizer(impressions, padding=True, truncation=True, return_tensors='pt')
        let_ids, let_masks = res_let.input_ids.to(DEVICE), res_let.attention_mask.to(DEVICE)
        imp_ids, imp_masks = res_imp.input_ids.to(DEVICE), res_imp.attention_mask.to(DEVICE)
        let_feats = self.text_encoder(let_ids, let_masks).last_hidden_state # b, len, 768
        imp_feats = self.text_encoder(imp_ids, imp_masks).last_hidden_state # b, len, 768

        self.model.eval()
        samples = self.model.sample([let_feats, imp_feats], bsz=len(letters), num_sampling_timesteps=timesteps, device=DEVICE).cpu().numpy()
        gen_full, gen_list = self.vis_a2z(samples)
        return gen_list

    def _eval_uno_xN(self, impression, epoch, timesteps=None, load_from_file=False, N=5):
        if load_from_file:
            checkpoint = torch.load(f'weights/font_dif-{epoch}.model')
            self.model.load_state_dict(checkpoint)

        letters = list(string.ascii_uppercase) * N
        impressions = [impression] * len(letters)

        res_let = self.tokenizer(letters, padding=True, truncation=True, return_tensors='pt')
        res_imp = self.tokenizer(impressions, padding=True, truncation=True, return_tensors='pt')
        let_ids, let_masks = res_let.input_ids.to(DEVICE), res_let.attention_mask.to(DEVICE)
        imp_ids, imp_masks = res_imp.input_ids.to(DEVICE), res_imp.attention_mask.to(DEVICE)
        let_feats = self.text_encoder(let_ids, let_masks).last_hidden_state # b, len, 768
        imp_feats = self.text_encoder(imp_ids, imp_masks).last_hidden_state # b, len, 768

        self.model.eval()
        samples = self.model.sample([let_feats, imp_feats], bsz=len(letters), num_sampling_timesteps=timesteps, device=DEVICE).cpu().numpy()
        gen_full, gen_list = self.vis_a2z(samples)
        return gen_list


    def eval_one_letter_impression(self, letters, impression, epoch, timesteps=None, category=None, real_img=None, load_from_file=False, show_full_impression=True):
        if load_from_file:
            checkpoint = torch.load(f'weights/font_dif-{epoch}.model')
            self.model.load_state_dict(checkpoint)

        impressions = [impression] * len(letters)

        res_let = self.tokenizer(letters, padding=True, truncation=True, return_tensors='pt')
        res_imp = self.tokenizer(impressions, padding=True, truncation=True, return_tensors='pt')
        let_ids, let_masks = res_let.input_ids.to(DEVICE), res_let.attention_mask.to(DEVICE)
        imp_ids, imp_masks = res_imp.input_ids.to(DEVICE), res_imp.attention_mask.to(DEVICE)
        let_feats = self.text_encoder(let_ids, let_masks).last_hidden_state # b, len, 768
        imp_feats = self.text_encoder(imp_ids, imp_masks).last_hidden_state # b, len, 768

        self.model.eval()
        samples = self.model.sample([let_feats, imp_feats], bsz=len(letters), num_sampling_timesteps=timesteps, device=DEVICE).cpu().numpy()
        gen_full, gen_list = self.vis_a2z(samples)
        if real_img is not None:
            real_full, real_list = real_img
            final = np.vstack([real_full, gen_full])
        else:
            final = gen_full
        if show_full_impression:
            imp = impression
        else:
            imp = ''
        if category:
            name = f'letter_a2z_{category}_{imp}_{epoch}'
        else:
            name = f'letter_a2z_{imp}_{epoch}'
        if not os.path.exists(f'outputs/{name}'):
            os.makedirs(f'outputs/{name}')
        filename = f'outputs/{name}.png'
        plt.imsave(filename, final, vmin=0, vmax=255, cmap='gray')
        if real_img is not None:
            for l, img in zip(list(string.ascii_uppercase), real_list):
                plt.imsave(f'outputs/{name}/GT_{l}.png', np.array(img), vmin=0, vmax=255, cmap='gray')
        for l, img in zip(list(string.ascii_uppercase), gen_list):
            plt.imsave(f'outputs/{name}/gen_{l}.png', img, vmin=0, vmax=255, cmap='gray')

    def eval_one_letters(self, letters, impression, epoch, timesteps=None, category=None, real_img=None, load_from_file=False, show_full_impression=True):
        if load_from_file:
            checkpoint = torch.load(f'weights/font_dif-{epoch}.model')
            self.model.load_state_dict(checkpoint)

        letters = list(letters)
        impressions = [impression] * len(letters)

        res_let = self.tokenizer(letters, padding=True, truncation=True, return_tensors='pt')
        res_imp = self.tokenizer(impressions, padding=True, truncation=True, return_tensors='pt')
        let_ids, let_masks = res_let.input_ids.to(DEVICE), res_let.attention_mask.to(DEVICE)
        imp_ids, imp_masks = res_imp.input_ids.to(DEVICE), res_imp.attention_mask.to(DEVICE)
        let_feats = self.text_encoder(let_ids, let_masks).last_hidden_state # b, len, 768
        imp_feats = self.text_encoder(imp_ids, imp_masks).last_hidden_state # b, len, 768

        self.model.eval()
        samples = self.model.sample([let_feats, imp_feats], bsz=len(letters), num_sampling_timesteps=timesteps, device=DEVICE).cpu().numpy()
        gen_full, gen_list = self.vis_a2z(samples)
        if real_img is not None:
            real_full, real_list = real_img
            final = np.vstack([real_full, gen_full])
        else:
            final = gen_full
        if show_full_impression:
            imp = impression[:100] # fix the file name too long error
        else:
            imp = ''
        if category:
            name = f'a2z_{category}_{imp}_{epoch}'
        else:
            name = f'a2z_{imp}_{epoch}'
        if not os.path.exists(f'outputs/{name}'):
            os.makedirs(f'outputs/{name}')
        filename = f'outputs/{name}.png'
        plt.imsave(filename, final, vmin=0, vmax=255, cmap='gray')
        if real_img is not None:
            for l, img in zip(list(string.ascii_uppercase), real_list):
                plt.imsave(f'outputs/{name}/GT_{l}.png', np.array(img), vmin=0, vmax=255, cmap='gray')
        for l, img in zip(letters, gen_list):
            plt.imsave(f'outputs/{name}/gen_{l}_{random.randrange(1000,9999)}.png', img, vmin=0, vmax=255, cmap='gray')

    def eval_one(self, impression, epoch, timesteps=None, category=None, real_img=None, load_from_file=False, show_full_impression=True):
        if load_from_file:
            checkpoint = torch.load(f'weights/font_dif-{epoch}.model')
            self.model.load_state_dict(checkpoint)

        letters = list(string.ascii_uppercase)
        impressions = [impression] * len(letters)

        res_let = self.tokenizer(letters, padding=True, truncation=True, return_tensors='pt')
        res_imp = self.tokenizer(impressions, padding=True, truncation=True, return_tensors='pt')
        let_ids, let_masks = res_let.input_ids.to(DEVICE), res_let.attention_mask.to(DEVICE)
        imp_ids, imp_masks = res_imp.input_ids.to(DEVICE), res_imp.attention_mask.to(DEVICE)
        let_feats = self.text_encoder(let_ids, let_masks).last_hidden_state # b, len, 768
        imp_feats = self.text_encoder(imp_ids, imp_masks).last_hidden_state # b, len, 768

        self.model.eval()
        samples = self.model.sample([let_feats, imp_feats], bsz=len(letters), num_sampling_timesteps=timesteps, device=DEVICE).cpu().numpy()
        gen_full, gen_list = self.vis_a2z(samples)
        if real_img is not None:
            real_full, real_list = real_img
            final = np.vstack([real_full, gen_full])
        else:
            final = gen_full
        if show_full_impression:
            imp = impression
        else:
            imp = ''
        if category:
            name = f'a2z_{category}_{imp}_{epoch}'
        else:
            name = f'a2z_{imp}_{epoch}'
        if not os.path.exists(f'outputs/{name}'):
            os.makedirs(f'outputs/{name}')
        filename = f'outputs/{name}.png'
        plt.imsave(filename, final, vmin=0, vmax=255, cmap='gray')
        if real_img is not None:
            for l, img in zip(list(string.ascii_uppercase), real_list):
                plt.imsave(f'outputs/{name}/GT_{l}.png', np.array(img), vmin=0, vmax=255, cmap='gray')
        for l, img in zip(list(string.ascii_uppercase), gen_list):
            plt.imsave(f'outputs/{name}/gen_{l}.png', img, vmin=0, vmax=255, cmap='gray')


    def eval(self, epoch, timesteps=None, load_from_file=False):
        if load_from_file:
            checkpoint = torch.load(f'weights/font_dif-{epoch}.model')
            self.model.load_state_dict(checkpoint)

        test_data = next(iter(self.test_dataloader))
        x_vis = test_data['images']
        letters = test_data['letters']
        impressions = test_data['impressions']
        res_let = self.tokenizer(letters, padding=True, truncation=True, return_tensors='pt')
        res_imp = self.tokenizer(impressions, padding=True, truncation=True, return_tensors='pt')
        let_ids, let_masks = res_let.input_ids.to(DEVICE), res_let.attention_mask.to(DEVICE)
        imp_ids, imp_masks = res_imp.input_ids.to(DEVICE), res_imp.attention_mask.to(DEVICE)
        let_feats = self.text_encoder(let_ids, let_masks).last_hidden_state # b, len, 768
        imp_feats = self.text_encoder(imp_ids, imp_masks).last_hidden_state # b, len, 768

        self.model.eval()
        samples = self.model.sample([let_feats, imp_feats], bsz=EVAL_BATCH_SIZE, num_sampling_timesteps=timesteps, device=DEVICE).cpu().numpy()
        self.vis(x_vis, samples, f"outputs/fonts_{epoch}.png")

    def vis_a2z(self, gen_samples):
        samples = rearrange(gen_samples, "t b () h w -> t b (h w)")
        samples = samples * 127.5 + 127.5

        samples = samples[0]
        gen_list = rearrange(samples, "b (h w) -> b h w", h=IMG_SIZE[0])
        count = samples.shape[0]

        nrows, ncols = 4, 7
        raster = np.zeros((nrows * IMG_SIZE[0], ncols * IMG_SIZE[1]), dtype=np.float32)

        for i in range(nrows * ncols):
            if i >= count:
                break
            row, col = i // ncols, i % ncols
            raster[IMG_SIZE[0] * row : IMG_SIZE[0] * (row + 1), IMG_SIZE[1] * col : IMG_SIZE[1] * (col + 1)] = samples[i].reshape(IMG_SIZE[0], IMG_SIZE[1])

        raster = 255 - raster
        gen_list = 255 - gen_list
        return raster, gen_list

    def vis(self, real_imgs, gen_samples, filename):
        x_vis = rearrange(real_imgs, 'b () h w -> b (h w)')
        x_vis = x_vis * 127.5 + 127.5
        samples = rearrange(gen_samples, "t b () h w -> t b (h w)")
        samples = samples * 127.5 + 127.5

        nrows, ncols = 10, 2
        percents = (100, 75, 50, 25, 0)
        raster = np.zeros((nrows * IMG_SIZE[0], ncols * IMG_SIZE[1] * (len(percents) + 1)), dtype=np.float32)

        for i in range(nrows * ncols):
            row, col = i // ncols, i % ncols
            raster[IMG_SIZE[0] * row : IMG_SIZE[0] * (row + 1), IMG_SIZE[1] * col : IMG_SIZE[1] * (col + 1)] = x_vis[i].reshape(IMG_SIZE[0], IMG_SIZE[1])
        for percent_idx, percent in enumerate(percents):
            itr_num = int(round(0.01 * percent * (len(samples) - 1)))
            for i in range(nrows * ncols):
                row, col = i // ncols, i % ncols
                offset = IMG_SIZE[1] * ncols * (percent_idx + 1)
                raster[IMG_SIZE[0] * row : IMG_SIZE[0] * (row + 1), offset + IMG_SIZE[1] * col : offset + IMG_SIZE[1] * (col + 1)] = samples[itr_num][i].reshape(IMG_SIZE[0], IMG_SIZE[1])
        plt.imsave(filename, raster, vmin=0, vmax=255, cmap='gray')


if __name__ == "__main__":
    main = Main()
    main.train(0)

    '''EVAL'''
    class_id = 'dinfun-pro-effects'
    with open(f'/home/lkang/datasets/Impression_Fonts_Dataset/dataset/taglabel/{class_id}', 'r') as _f:
        imps = _f.read().strip().split(' ')
    imps = ','.join(imps)
    main.eval_one(imps, 380, 100, category=class_id, real_img=None, load_from_file=True)
