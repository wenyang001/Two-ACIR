### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import io
import matplotlib.pyplot as plt
import torch

from visdom import Visdom
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Steps',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

    def plot_image(self, id, title, image):
        self.viz.image(
            image,
            win=id,
            opts=dict(title=title, caption='Result', store_history=True)
        )


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.vis = False

        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
        self.writer = SummaryWriter(self.log_dir)

        if self.tf_log:
            from tensorflow import summary
            self.summary = summary
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = self.summary.create_file_writer(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        if self.vis:
            self.plotter = VisdomLinePlotter(env_name='Image Plots')

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = self.tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = self.tf.expand_dims(image, 0)
        return image

    def image_grid(self, train_labels, train_images):
        """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
        # Create a figure to contain the plot.
        figure = plt.figure()
        for i in range(4):
            # Start next subplot.
            plt.subplot(1, 4, i + 1, title=train_labels[i])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i])

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        train_labels, train_images = [], []
        for label, image_numpy in visuals.items():
            # Create an Image object
            if len(image_numpy.shape) == 3:
                train_labels.append(label)
                train_images.append(torch.from_numpy(image_numpy))

        # h, w, c = train_images[0].shape
        # train_images = np.reshape(train_images, (-1, h, w, c))
        #img_grid = vutils.make_grid(train_images, nrow=5, padding=2, normalize=False, scale_each=True)

        # self.writer.add_images("Training images/1", train_images[0], epoch, dataformats='HWC')
        # self.writer.add_images("Training images/2", train_images[1], epoch, dataformats='HWC')
        # self.writer.add_images("Training images/3", train_images[2], epoch,  dataformats='HWC')
        # self.writer.add_images("Training images/4", train_images[3], epoch, dataformats='HWC')
        # self.writer.add_images("Training images/5", train_images[4], epoch,   dataformats='HWC')


        if self.tf_log: # show images in tensorboard output
            train_labels, train_images = [], []
            for label, image_numpy in visuals.items():
                # Create an Image object
                if len(image_numpy.shape) == 3:
                    train_labels.append(label)
                    train_images.append(image_numpy)

            h, w, c = train_images[0].shape
            train_images = np.reshape(train_images, (-1, h, w, c))
            with self.writer.as_default():
                self.summary.image("Training data", train_images, max_outputs=5, step=step)


        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

        if self.vis:
            for label, image_numpy in visuals.items():
                if len(image_numpy.shape) == 3:
                    image_numpy = np.transpose(image_numpy, (2, 0, 1))
                    # self.plotter.plot_image(label+str(epoch), label, image_numpy)
                    self.plotter.plot_image(label, label, image_numpy)



    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                with self.writer.as_default():
                    self.summary.scalar(tag, value, step=step)

        if self.vis:
            loss_D = (errors['D_fake'] + errors['D_real']) * 0.5
            loss_G = errors['G_GAN_Feat'] + errors['G_GAN'] + errors['G_Edge']
            loss_Enhance = errors['G_L2'] + errors['G_Edge2'] + errors['G_VGG']

            self.plotter.plot('Total Loss', 'loss_D', 'Loss', step, loss_D)
            self.plotter.plot('Total Loss', 'loss_G', 'Loss', step, loss_G)
            self.plotter.plot('Total Loss', 'loss_Enhance', 'Loss', step, loss_Enhance)

            for tag, value in errors.items():
                self.plotter.plot('Single Loss', tag, 'Loss', step, value)

        loss_D = (errors['D_fake'] + errors['D_real']) * 0.5
        loss_G = errors['G_GAN_Feat'] + errors['G_GAN'] + errors['G_Edge']
        loss_Enhance = errors['G_L2'] + errors['G_Edge2'] + errors['G_VGG']

        self.writer.add_scalar("Total Loss/loss_D", loss_D, step)
        self.writer.add_scalar("Total Loss/loss_G", loss_G, step)
        self.writer.add_scalar("Total Loss/loss_Enhance", loss_Enhance, step)

        for tag, value in errors.items():
            self.writer.add_scalar('Single Loss/' + str(tag), value, step)




    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)


    # save image to the disk
    def save_output_images(self, image_dir, visuals, image_path):
        short_path = ntpath.basename(image_path[0])
        #name = os.path.splitext(short_path)[0]
        for label, image_numpy in visuals.items():
            image_name = short_path
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)



