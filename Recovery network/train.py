### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import os
import torch
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import numpy as np
from torch.autograd import Variable



opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):

    if epoch > opt.niter:
        model.module.update_learning_rate()

    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size  # epoch_iter 一个epoch中当前训练到哪里了

    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize  # total_steps 总体当前训练到哪里了
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0

        ############## Forward Pass ######################
        losses, generated = model(Variable(data['label']), Variable(data['image']), Variable(data['gt_thumbnail']), Variable(data['gt_thumbnail_original']), infer=save_fake)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]  # 如果vgg loss没有赋值就不会执行torch.mean(x)
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN_Feat'] + loss_dict['G_GAN'] + loss_dict['G_Edge']
        loss_Enhance = loss_dict['G_L2'] + loss_dict['G_Edge2'] + loss_dict['G_VGG']


        ############### Backward Pass ####################
        for p in model.module.netD.parameters():  # 不更新D
            p.requires_grad = False

        # update generator weights
        optimizer_G.zero_grad()
        loss_G.backward(retain_graph=True)
        loss_Enhance.backward()
        optimizer_G.step()

        for p in model.module.netD.parameters():
            p.requires_grad = True

        # update discriminator weights
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        ############## Display results and errors ##########
        if total_steps % 200 == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            visualizer.plot_current_errors(errors, total_steps)

        w = data['label'].shape[3]
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0][:,:,:w], opt.label_nc)),
                                   ('thumbnail', util.tensor2im(data['gt_thumbnail'][0][:,:,:w])),
                                   ('pix2pixHD', util.tensor2im(generated[0].data[0])),
                                   ('Ours', util.tensor2im(generated[2].data[0])),
                                   ('real_image', util.tensor2im(data['image'][0][:,:,:w]))])

            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        # if epoch > 170:
        #     model.module.save(epoch)
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()