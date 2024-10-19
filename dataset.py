import os
import string
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision

IMG_SIZE = (32, 32)

TRAIN = '/home/lkang/datasets/Impression_Fonts_Dataset/dataset/font_impression_tro.train.v4'
TEST = '/home/lkang/datasets/Impression_Fonts_Dataset/dataset/font_impression_tro.test.v4'
IMG = '/home/lkang/datasets/Impression_Fonts_Dataset/dataset/fontimage/'


def vis_real(font_id):
    letters = list(string.ascii_uppercase)
    nrows, ncols = 4, 7
    out = np.ones((nrows * IMG_SIZE[0], ncols * IMG_SIZE[1]), dtype=np.float32) * 255
    imgs = []
    
    for i in range(nrows * ncols):
        if i >= len(letters):
            break
        row, col = i // ncols, i % ncols
        img = Image.open(f'{IMG}/{font_id}_{letters[i]}{letters[i]}.png').convert('L')
        width, height = img.size
        img = img.crop((0, 0, IMG_SIZE[1]*height/IMG_SIZE[0], height))
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
        imgs.append(img)
        out[IMG_SIZE[0] * row : IMG_SIZE[0] * (row + 1), IMG_SIZE[1] * col : IMG_SIZE[1] * (col + 1)] = img
    return out, imgs


def load_data():
    font_dir = dict()
    font_dir['train'] = TRAIN
    font_dir['test'] = TEST
    data_train = Font(font_dir, IMG, split='train')
    data_test = Font(font_dir, IMG, split='test')
    return data_train, data_test

class Font(Dataset):
    def __init__(self, font_dir, img_dir, split):
        with open(font_dir[split], 'r') as _f:
            data = _f.readlines()
        self.fonts = [line.strip().split(' ') for line in data]
        self.img_dir = img_dir
        self.process = torchvision.transforms.Compose(
                    [torchvision.transforms.Resize(IMG_SIZE),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: 2*(x-0.5))
                    ])

    def __len__(self):
        return len(self.fonts)

    def get_random_k(self, impressions, k=5):
        k_items = np.random.choice(impressions, k, replace=False)
        return ','.join(k_items)

    def crop_rev(self, img):
        width, height = img.size

        im = img.crop((0, 0, IMG_SIZE[1]*height/IMG_SIZE[0], height))
        im = ImageOps.invert(im)
        return im

    def __getitem__(self, idx):
        img_name, letter, impressions = self.fonts[idx]
        image = Image.open(f'{self.img_dir}{img_name}').convert('L')
        image = self.crop_rev(image)
        imps = self.get_random_k(impressions.split(','))
        image = self.process(image)

        sample_info = {'image': image,
                       'img_name': img_name,
                       'letter': letter,
                       'impression': imps
                       }

        return sample_info


if __name__ == '__main__':
    pass
