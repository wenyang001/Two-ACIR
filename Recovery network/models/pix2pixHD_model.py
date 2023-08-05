### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import sobel
from torch.nn.functional import interpolate
from skimage.feature import canny
import torchvision.transforms.functional as FL
import torchvision.transforms as T
from .refined_guiding import L1_Charbonnier_loss

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        self.edge = sobel.GradLayer()
        input_nc = opt.label_nc if opt.label_nc != 0 else 3

        ##### define networks
        # Generator network
        netG_input_nc = input_nc + 3
        netG_output_nc = opt.output_nc

        self.netG = networks.define_G(netG_input_nc, netG_output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, scen=opt.scen)

        # Dis　criminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 3
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

        print('---------- Networks initialized -------------')

        # load networks  如果是test或者微调模型时进入，载入训练好的参数
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:  # 训练的时候开启，测试的时候关闭
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)  # 创建了ImagePool
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionMse = torch.nn.MSELoss()

            # define edge loss functions
            self.criterionEdge = torch.nn.MSELoss()
            self.criterionL1_Charbonnier_loss = L1_Charbonnier_loss()

            # define vgg loss
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        
            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_Edge', 'D_real', 'D_fake','G_L2', 'G_Edge2']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0: # number of epochs that we only train the outmost local enhancer 默认为0
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]
                params += list(self.netG.dehaze.parameters())
            else:
                params = list(self.netG.parameters())

            if self.gen_features:              
                params += list(self.netE.parameters())         

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


    def encode_input(self, label, image, gt_thumbnail, gt_thumbnail_original, infer=False):
        label = label.data.cuda()
        image = Variable(image.data.cuda())
        gt_thumbnail = Variable(gt_thumbnail.data.cuda())
        input_label = torch.cat((label, gt_thumbnail), dim=1)

        w = gt_thumbnail_original.shape[3]
        gt_thumbnail_original = gt_thumbnail_original[:,:,:,:w]
        gt_thumbnail_original = Variable(gt_thumbnail_original.data.cuda())
        return input_label, image, gt_thumbnail, gt_thumbnail_original

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)


    def forward(self, label, image, gt_thumbnail, gt_thumbnail_original, infer=False):

        # Encode Inputs
        input_label, image, gt_thumbnail, gt_thumbnail_original = self.encode_input(label, image, gt_thumbnail, gt_thumbnail_original)

        w = input_label.shape[3]

        real_image = image[..., :w]
        real_image_edge = self.edge(real_image)

        fake_image, fake_image_edge, enhance, enhance_edge, enhance_list, GT_list, tag = self.netG.forward(input_label, gt_thumbnail_original, real_image)  # 此时有三个数据回来
        input_label = input_label[..., :w]

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat

        loss_G_Edge1 = self.criterionEdge(real_image_edge, fake_image_edge) * 15
        loss_G_Edge2 = self.criterionEdge(real_image_edge, enhance_edge) * 15
        loss_G_VGG = self.criterionVGG(real_image, fake_image) * self.opt.lambda_feat

        loss_x1 = self.criterionL1_Charbonnier_loss(GT_list[0], enhance_list[0]) if GT_list[0] is not None else 0
        loss_x2 = self.criterionL1_Charbonnier_loss(GT_list[1], enhance_list[1]) if GT_list[1] is not None else 0
        loss_x4 = self.criterionL1_Charbonnier_loss(GT_list[2], enhance_list[2]) if GT_list[2] is not None else 0
        loss_x8 = self.criterionL1_Charbonnier_loss(GT_list[3], enhance_list[3]) if GT_list[3] is not None else 0

        no_edge_loss_cha_loss = False
        if no_edge_loss_cha_loss:
            loss_Char = loss_x2
            loss_G_L2 = loss_Char * 150

        else:
            loss_Char = loss_x1 + loss_x2 + loss_x4 + loss_x8
            loss_G_L2 = loss_Char * 150 / tag

        return [[loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_Edge1, loss_D_real, loss_D_fake, loss_G_L2, loss_G_Edge2], None if not infer else [fake_image, fake_image_edge, enhance, enhance_edge]]

    def inference(self, label, image, gt_thumbnail, gt_thumbnail_original):
        input_label, real_image, gt_thumbnail, gt_thumbnail_original = self.encode_input(label, image, gt_thumbnail, gt_thumbnail_original, infer=True)
        w = input_label.shape[3]
        real_image = image[..., :w]
        fake_image, fake_image_edge, enhance, enhance_edge,  _, _, _ = self.netG.forward(input_label, gt_thumbnail_original, real_image)  # 此时有三个数据回来
        return [fake_image, fake_image_edge, enhance, enhance_edge]


    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = torch.cuda.FloatTensor(1, self.opt.feat_num, inst.size()[2], inst.size()[3])                   
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == i).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k] 
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == i).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
