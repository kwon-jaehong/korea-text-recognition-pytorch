import os
import sys
import re
import six
import math
import torch
from natsort import natsorted
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter
import pandas as pd
from collections import Counter
import cv2
import random as rnd


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img



class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        transform = ResizeNormalize((self.imgW, self.imgH))
        image_tensors = [transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels



class custom_dataset(Dataset):
    def __init__(self,dict_path,font_path):
        
        ## 폰트 파일 리스트 및 폰트 경로 저장
        self.font_path = font_path
        self.font_list = os.listdir(font_path)
        
        
        f = open(dict_path, 'r')
        lines = f.readlines()
        f.close()
        lines.sort(key=len)
        for i,line in enumerate(lines):
            lines[i] = line.replace('\n','')
        
        self.dict = lines
        self.len_dict = len(lines)

    def __len__(self):
        return self.len_dict
    
    def __getitem__(self, index):
        return index
        # 딕트를 기준으로 이미지를 생성 이때는 정상 이미지여야함
    
    
    

# class make_custom_dataset():
#     def __init__(self,dict_path="./dict/dict.txt",font_path="./font"):        
#         self.font_list = os.listdir(font_path)
#         self.dict_list = load_dict_file(dict_path)
      

    
#     def load_dict_file(path):
#         f = open(path, 'r')
#         lines = f.readlines()
#         f.close()
#         lines.sort(key=len)

#         return lines
    
    
    
    
    
    
## dict 텍스트 파일 불러오기 

    
    
    

def generate_text_img(text,font):
    
    ## 빈칸 일경우 너비 설정
    space_width = 2
    

    image_font = ImageFont.truetype(font = font, size = 40)
    words = text.split(' ')
    space_width = image_font.getsize(' ')[0] * space_width
    words_width = [image_font.getsize(w)[0] for w in words]
    
    text_width = sum(words_width) + int(space_width) * (len(words) - 1)
    text_height = max([image_font.getsize(w)[1] for w in words])
    
    txt_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_img)
    
    
#     text_color = "black,red"
#     colors = [ImageColor.getrgb(c) for c in text_color.split(',')]    
    ## 흰색이 나오면 안되니 픽셀 255,255,127 사이값이 랜덤하게 나옴
    colors = [(0, 0, 0), (255, 255, 127)]
#     print(colors)
    
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2]))
    )
    
    for i, w in enumerate(words):
        txt_draw.text((sum(words_width[0:i]) + i * int(space_width), 0), w, fill=fill, font=image_font)
    return txt_img


def _apply_func_distorsion(image, vertical, horizontal, max_offset, func):
    # Nothing to do!
    if not vertical and not horizontal:
        return image

    rgb_image = image.convert('RGBA')
    
    img_arr = np.array(rgb_image)

    vertical_offsets = [func(i) for i in range(img_arr.shape[1])]
    horizontal_offsets = [
        func(i)
        for i in range(
            img_arr.shape[0] + (
                (max(vertical_offsets) - min(min(vertical_offsets), 0)) if vertical else 0
            )
        )
    ]

    new_img_arr = np.zeros((
                        img_arr.shape[0] + (2 * max_offset if vertical else 0),
                        img_arr.shape[1] + (2 * max_offset if horizontal else 0),
                        4
                    ))

    new_img_arr_copy = np.copy(new_img_arr)
    
    if vertical:
        column_height = img_arr.shape[0]
        for i, o in enumerate(vertical_offsets):
            column_pos = (i + max_offset) if horizontal else i
            new_img_arr[max_offset+o:column_height+max_offset+o, column_pos, :] = img_arr[:, i, :]

    if horizontal:
        row_width = img_arr.shape[1]
        for i, o in enumerate(horizontal_offsets):
            if vertical:
                new_img_arr_copy[i, max_offset+o:row_width+max_offset+o,:] = new_img_arr[i, max_offset:row_width+max_offset, :]
            else:
                new_img_arr[i, max_offset+o:row_width+max_offset+o,:] = img_arr[i, :, :]

    return Image.fromarray(np.uint8(new_img_arr_copy if horizontal and vertical else new_img_arr)).convert('RGBA')

def sin(image, vertical=False, horizontal=False):
    max_offset = int(image.height ** 0.5)
    return _apply_func_distorsion(image, vertical, horizontal, max_offset, (lambda x: int(math.sin(math.radians(x)) * max_offset)))

def cos(image, vertical=False, horizontal=False):
    max_offset = int(image.height ** 0.5)
    return _apply_func_distorsion(image, vertical, horizontal, max_offset, (lambda x: int(math.cos(math.radians(x)) * max_offset)))
