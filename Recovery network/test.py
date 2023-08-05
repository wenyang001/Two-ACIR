### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html, convert, metric
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # mytest code only supports nThreads = 1
opt.batchSize = 1  # mytest code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

path = web_dir + '.txt'
f = open(path, 'a')
print(path)

t0 = time.time()
test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []

for i, data in enumerate(dataset):
    if i >= 10:
        break

    w = data['label'].shape[3]
    generated = model.inference(data['label'], data['image'], data['gt_thumbnail'], data['gt_thumbnail_original'])
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0][:, :, :w], opt.label_nc)),
                           ('thumbnail', util.tensor2im(data['gt_thumbnail'][0][:, :, :w])),
                           ('pix2pixHD', util.tensor2im(generated[0].data[0])),
                           ('pix2pixHD_edge', util.tensor2im(generated[1].data[0])),
                           ('Ours', util.tensor2im(generated[2].data[0])),
                           ('Ours_edge', util.tensor2im(generated[3].data[0])),
                           ('real_image', util.tensor2im(data['image'][0][:, :, :w])),
                           ('real_image_edge', util.tensor2im(data['gt_image_edge'][0][:, :, :w])),
                           ])

    # Convert images to YCbCr space
    Ours = visuals['Ours']
    Real_image = visuals['real_image']

    Ours = convert.rgb2y(Ours)
    Real_image = convert.rgb2y(Real_image)

    show = False
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(Ours)
        plt.show()
        plt.imshow(Real_image)
        plt.show()

    psnr = metric.psnr(Ours, Real_image)
    ssim = metric.ssim(Ours, Real_image)
    test_results['psnr'].append(psnr)
    test_results['ssim'].append(ssim)

    img_path = data['path']
    print('%d process image... %s' % (i, img_path))
    if i < 50:
        visualizer.save_images(webpage, visuals, img_path)

t1 = time.time()
print(str((t1-t0)/60) + ' minutes')
webpage.save()

for key, values in test_results.items():
    avg_value = sum(values) / len(values)
    tmp_str = 'avg %s: %.4f \n' % (key, avg_value)
    f.write(tmp_str)
    print(tmp_str, end='')

f.close()


