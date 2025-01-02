import torch
import numpy as np
import torch.nn as nn
import cv2
import math
import sys
import os
import pickle
from pathlib import  Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

CUR_DIR = Path(os.getcwd()).resolve()
sys.path.append(str(CUR_DIR))
import hyper_dl
from hyper_dl.models.necks import LSSCylinViewTransformer

if __name__ == "__main__":
    """
img_view_transformer=dict(
        type='LSSCylinViewTransformer',
        cam_config_path='data/samples/bev3d/2023-03-24-17-43-48/calib/',
        src_size=data_config['src_size'],
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        downsample=16),
    """
    voxel_size = [0.1, 0.1, 0.2]

    numC_Trans = 64
    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 1.0],
    }
    model = LSSCylinViewTransformer(cam_config_path = '/home/gpu/works/fk/task/hyper_dl/data/samples/bev3d/2023-03-24-17-43-48/calib/',
        src_size = (576,704),
        grid_config = grid_config,
        input_size = (576,704),
        in_channels = 256,
        out_channels = numC_Trans,
        downsample = 16)
    model = model.cuda()
    with open('/home/gpu/works/fk/task/hyper_dl/sample_data.pkl', 'rb') as file:
        input = pickle.load(file)
    model = model.cuda()
    img_inputs = input['img_inputs']
    img_inputs = list(img_inputs)
    img_inputs[0]=  torch.randn(1,4,256,36,64)
    img_inputs[1]= img_inputs[1].unsqueeze(0)
    for i in range(len(img_inputs)):
         img_inputs[i] = img_inputs[i].to('cuda')
    print(img_inputs[0].device)
    output = model(img_inputs)#256+
    print(output[0].shape)
    i = 1
