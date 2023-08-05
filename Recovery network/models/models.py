### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
from .pix2pixHD_model import Pix2PixHDModel
from .ui_model import  UIModel

def create_model(opt):
    if opt.model == 'pix2pixHD':
        model = Pix2PixHDModel()
        model.initialize(opt)
        print("model [%s] was created" % (model.name()))
        if opt.isTrain and len(opt.gpu_ids):
            model = torch.nn.DataParallel(model.to(torch.device("cuda:0")), device_ids=opt.gpu_ids)

    else:
        model = UIModel()

    return model
