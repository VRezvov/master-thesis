#!/usr/bin/env python
# coding: utf-8

try:
    import os
    from os import listdir
    from os.path import join
    import pickle
    from queue import Empty, Queue
    from math import exp
    from math import log10
    import torch.nn.functional as F
    from matplotlib import pyplot as plt
    import numpy as np
    from PIL import Image
    import random
    import threading
    from threading import Thread
    import torch
    from torch import nn
    from torch.autograd import Variable
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.models.vgg import vgg19
    from tqdm import tqdm
    from libs.SGDR import CosineAnnealingWarmRestarts

except ImportError as e:
    print(e)
    raise ImportError

plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 300

nn.Sequential(*list(vgg19(pretrained=True).features)[:]).eval()

data_index_fname = 'data/data_index.p'
data_index_val_fname = 'data/data_index_val.p'

def find_files(dataset_dir):
    image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir)]
    return image_filenames


class thread_killer(object):    
    """Boolean object for signaling a worker thread to terminate"""
    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """
    Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    while tokill() == False:
        for _, (batch_images, batch_targets, batch_mask) in enumerate(dataset_generator):
            #We fill the queue with new fetched batch until we reach the max size.
            batches_queue.put(((batch_images, batch_targets, batch_mask)), block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill,cuda_batches_queue,batches_queue):
    
    """
    Thread worker for transferring pytorch tensors into GPU. 
    batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        (batch_images, batch_targets, batch_masks) = batches_queue.get(block=True)
        
        batch_images = torch.from_numpy(batch_images)
        batch_labels = torch.from_numpy(batch_targets)
        batch_masks = torch.from_numpy(batch_masks)
        
        #batch_images = Variable(batch_images).cuda()
        #batch_labels = Variable(batch_labels).cuda()
        #batch_masks = Variable(batch_masks).cuda()
        
        cuda_batches_queue.put(((batch_images, batch_labels, batch_masks)), block=True)
        if tokill() == True:
            return


class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def get_path_i(paths_count):
    """
    Cyclic generator of paths indices
    """   
    current_path_id = 0
    while True:
        yield current_path_id
        current_path_id  = (current_path_id + 1) % paths_count


class InputGenerator:
    def __init__(self, data_index_fname, batch_size, debug=False):
        print('loading files index...')
        
        with open(data_index_fname, 'rb') as f:
            data_dicts = pickle.load(f)
            
        lores_datafiles = [f for f in find_files('data/LoRes/Arrays/FIELD/')]
        lores_datafiles.sort()
        hires_datafiles = [f for f in find_files('data/HiRes/Arrays/FIELD/')]
        hires_datafiles.sort()
        
        data_dict_lores = dict(zip(lores_datafiles, [np.load(f, mmap_mode = 'r') for f in lores_datafiles]))
        data_dict_hires = dict(zip(hires_datafiles, [np.load(f, mmap_mode = 'r') for f in hires_datafiles]))
        
        self.data_index_fname = data_index_fname
        self.data_mask = np.load('data/HiRes/Arrays/Percentile/Mask_95.npy')
        self.batch_size = batch_size
        self.debug = debug
        self.index = 0
        self.init_count = 0
        self.dicts = data_dicts
        self.dict_lores = data_dict_lores
        self.dict_hires = data_dict_hires
        print('examples number: %d' % len(self.dicts))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.path_id_generator = threadsafe_iter(get_path_i(len(self.dicts)))
        self.imgs = []
        self.trgs = []
        self.msks = []

    def shuffle(self):
        random.shuffle(self.dicts)

    def __iter__(self):
        while True:
            # In the start of each epoch we shuffle the data paths
            with self.lock:
                if (self.init_count == 0):
                    self.shuffle()
                    self.imgs, self.trgs, self_msks = [], [], []
                    self.init_count = 1
            # Iterates through the input paths in a thread-safe manner
            for path_id in self.path_id_generator:
                
                img_fn = self.dicts[path_id]["data_fname_lores"]
                img_id = self.dicts[path_id]["data_lores_idx"]
                img = self.dict_lores[img_fn][img_id]
                
                trg_fn = self.dicts[path_id]["data_fname_hires"]
                trg_id = self.dicts[path_id]["data_hires_idx"]
                trg = self.dict_hires[trg_fn][trg_id]
                
                msk_id = self.dicts[path_id]['season_idx']
                msk = self.data_mask[msk_id]

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if (len(self.imgs)) < self.batch_size:
                        self.imgs.append(img)
                        self.trgs.append(trg)
                        self.msks.append(msk)
                    if len(self.imgs) % self.batch_size == 0:
                        imgs_f32 = np.float32(self.imgs)
                        trgs_f32 = np.float32(self.trgs)
                        msks_f32 = np.float32(self.msks)
                        yield (imgs_f32, trgs_f32, msks_f32)
                        self.imgs, self.trgs, self.msks = [], [], []
            # At the end of an epoch we re-init data-structures
            with self.lock:
                random.shuffle(self.dicts)
                self.init_count = 0

    def __call__(self):
        return self.__iter__()

# SSIM

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #.mm - Performs a matrix multiplication of the matrice
    
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=110, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

# LOSS

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        #vgg = vgg.cuda()
        loss_network = nn.Sequential(*list(vgg.features)[:20]).eval()
        # A sequential container. Modules will be added to it in the order they are passed in the constructor. 
        # Alternatively, an ordered dict of modules can also be passed in.
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        
        generator_loss = image_loss + 0.01 * perception_loss
        return generator_loss


g_loss = GeneratorLoss()

# MODEL

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=4), nn.Conv2d(3, 64, kernel_size=7, padding=3), nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), ResidualBlock(128))
        self.block6 = ResidualBlock(128)
        self.block7 = ResidualBlock(128)
        self.block8 = ResidualBlock(128)
        self.block9 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), ResidualBlock(256))
        self.block10 = ResidualBlock(256)
        self.block11 = ResidualBlock(256)
        self.block12 = ResidualBlock(256)
        self.block13 = ResidualBlock(256)
        self.block14 = ResidualBlock(256)
        self.block15 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), ResidualBlock(512))
        self.block16 = ResidualBlock(512)
        self.block18 = nn.Sequential(ResidualBlock(512), nn.PReLU())
        self.block19 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.PReLU())
        self.block20 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.PReLU())
        self.block21 = UNet_up_block(512, 512, 512)
        self.block22 = UNet_up_block(256, 512, 384)
        self.block23 = UNet_up_block(128, 384, 256)
        self.block24 = UNet_up_block(64, 256, 192)
        #self.block25 = UNet_up_block(64, 96, 99)
        self.block26 = nn.Sequential(
            nn.Upsample(scale_factor=0.625),
            nn.Conv2d(192, 96, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(96, 24, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(24, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(self.maxpool(block1))
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        block6 = self.block6(self.maxpool(block5))
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        block9 = self.block9(block8)

        block10 = self.block10(self.maxpool(block9))
        block11 = self.block11(block10)
        block12 = self.block12(block11)
        block13 = self.block13(block12)
        block14 = self.block14(block13)
        block15 = self.block15(block14)

        block16 = self.block16(block15)
        block18 = self.block18(block16)
        block19 = self.block19(block18)
        block20 = self.block20(block19)

        block21 = self.block21(block18, block20)
        block22 = self.block22(block9, block21)
        block23 = self.block23(block5, block22)
        block24 = self.block24(block1, block23)
        block26 = self.block26(block24)

        return (torch.tanh(block26) + 1) / 2


class UNet_up_block(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.PReLU()

    def forward(self, prev_feature_map, x):
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.up_sampling(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

# TRAIN

def train_epoch(netG: nn.Module, optimizerG: torch.optim.Optimizer, epoch: int,
                cuda_batches_queue: Queue, generator_criterion: callable, STEPS_PER_EPOCH: int):

    netG.train()

    term_columns = os.get_terminal_size().columns
    pbar = tqdm(total=STEPS_PER_EPOCH, ncols=min(term_columns,180))
    
    mse_metrics = 0.0
    rmse_wind_metrics = 0.0
    rmse_slp_metrics = 0.0
    rmse95_metrics = 0.0
    
    ssim_metrics = 0.0
    psnr_metrics = 0.0
    running_results_g_loss = 0.0
    ssims = 0

    for batch_idx in range(STEPS_PER_EPOCH):
        
        (img, trg, msk) = cuda_batches_queue.get(block=True)
        
        wind_real = torch.sqrt(torch.square(trg[:,0,:,:]) + torch.square(trg[:,1,:,:]))
        
        img_norm = torch.zeros_like(img)
        trg_norm = torch.zeros_like(trg)
        
        img_norm[:,0,:,:] = (img[:,0,:,:]+40)/80
        img_norm[:,1,:,:] = (img[:,1,:,:]+40)/80
        img_norm[:,2,:,:] = (img[:,2,:,:]-940)/120
                                
        trg_norm[:,0,:,:] = (trg[:,0,:,:]+40)/80
        trg_norm[:,1,:,:] = (trg[:,1,:,:]+40)/80      
        trg_norm[:,2,:,:] = (trg[:,2,:,:]-940)/120
                
        diff_msk = (wind_real - msk) > 0
        wind_real_msk = wind_real * diff_msk
        
        g_update_first = True

        fake_img = netG(img_norm)
        g_loss = generator_criterion(fake_img, trg_norm)
        
        netG.zero_grad()
        g_loss.backward(retain_graph = True)
        optimizerG.step()     

        running_results_g_loss += g_loss.item()

        fake_img_unnorm = torch.zeros_like(fake_img)
        
        fake_img_unnorm[:,0,:,:] = 80*fake_img[:,0,:,:] - 40
        fake_img_unnorm[:,1,:,:] = 80*fake_img[:,1,:,:] - 40
        fake_img_unnorm[:,2,:,:] = 120*fake_img[:,2,:,:] + 940
        
        wind_fake = torch.sqrt(torch.square(fake_img_unnorm[:,0,:,:]) + torch.square(fake_img_unnorm[:,1,:,:]))
        wind_fake_msk = wind_fake * diff_msk
        
        batch_mse = ((fake_img - trg_norm) ** 2).data.mean()        
        batch_rmse_wind = torch.sqrt(((wind_fake - wind_real) ** 2).data.mean())
        batch_rmse95 = torch.sqrt(((wind_fake_msk - wind_real_msk) ** 2).data.sum() / diff_msk.data.sum())
        batch_rmse_slp = torch.sqrt(((fake_img_unnorm[:,2,:,:] - trg[:,2,:,:]) ** 2).data.mean())
        
        mse_metrics += batch_mse        
        rmse_wind_metrics += batch_rmse_wind
        rmse_slp_metrics += batch_rmse_slp
        rmse95_metrics += batch_rmse95
        
        batch_ssim = ssim(fake_img, trg_norm).item()
        ssims += batch_ssim       
        psnr_metrics = 10 * log10((fake_img.max()**2) / (mse_metrics / (batch_idx+1)))
        ssim_metrics = ssims / (batch_idx + 1)
        
        pbar.update(1)
        pbar.set_postfix_str(
            'Train Epoch: %d [%d/%d (%.2f%%)] RMSE_Wind: %.3f; RMSE_SLP: %.3f; RMSE95: %.3f; SSIM: %.3f; PSNR: %.3f' % (epoch,
                                                                                                                        batch_idx+1,
                                                                                                                        STEPS_PER_EPOCH,
                                                                                                                        100. *(batch_idx+1) / STEPS_PER_EPOCH,
                                                                                                                        rmse_wind_metrics / (batch_idx + 1),
                                                                                                                        rmse_slp_metrics / (batch_idx + 1),
                                                                                                                        rmse95_metrics / (batch_idx + 1),
                                                                                                                        ssim_metrics,
                                                                                                                        psnr_metrics)
        )
        if batch_idx >= STEPS_PER_EPOCH - 1:
            break

    g_loss = running_results_g_loss / STEPS_PER_EPOCH
    rmse_wind_metrics = float((rmse_wind_metrics/STEPS_PER_EPOCH).cpu())
    rmse_slp_metrics = float((rmse_slp_metrics/STEPS_PER_EPOCH).cpu())
    rmse95_metrics = float((rmse95_metrics/STEPS_PER_EPOCH).cpu())
    
    pbar.set_postfix_str(
        'Train Epoch: %d; RMSE_Wind: %.3f; RMSE_SLP: %.3f; RMSE95: %.3f; SSIM: %.3f; PSNR: %.3f' % (epoch,                                                                                                                                rmse_wind_metrics,
                                                                                                    rmse_slp_metrics,
                                                                                                    rmse95_metrics,
                                                                                                    ssim_metrics,
                                                                                                    psnr_metrics)
    )
    pbar.close()

    losses_dict = {'train_g_loss': g_loss}
    metrics_values = {'RMSE_Wind': rmse_wind_metrics,
                      'RMSE_SLP': rmse_slp_metrics,
                      'RMSE95': rmse95_metrics,
                      'PSNR': psnr_metrics,
                      'SSIM': ssim_metrics}
    return dict(**losses_dict, **metrics_values)


def validation(netG: nn.Module, cuda_batches_queue: Queue, generator_criterion: callable, VAL_STEPS: int):
    
    netG.eval()

    mse_metrics = 0.0
    rmse_wind_metrics = 0.0
    rmse_slp_metrics = 0.0
    rmse95_metrics = 0.0
    ssim_metrics = 0.0
    psnr_metrics = 0.0
    running_results_g_loss = 0.0
    ssims = 0
    
    with torch.no_grad():
        term_columns = os.get_terminal_size().columns
        pbar = tqdm(total=VAL_STEPS, ncols=min(term_columns, 180))
        for batch_idx in range(VAL_STEPS):
            (img, trg, msk) = cuda_batches_queue.get(block=True)

            wind_real = torch.sqrt(torch.square(trg[:,0,:,:]) + torch.square(trg[:,1,:,:]))
            
            img_norm = torch.zeros_like(img)
            trg_norm = torch.zeros_like(trg)

            img_norm[:,0,:,:] = (img[:,0,:,:]+40)/80
            img_norm[:,1,:,:] = (img[:,1,:,:]+40)/80
            img_norm[:,2,:,:] = (img[:,2,:,:]-940)/120
                                
            trg_norm[:,0,:,:] = (trg[:,0,:,:]+40)/80
            trg_norm[:,1,:,:] = (trg[:,1,:,:]+40)/80      
            trg_norm[:,2,:,:] = (trg[:,2,:,:]-940)/120
                
            diff_msk = (wind_real - msk) > 0
            wind_real_msk = wind_real * diff_msk

            output = netG(img_norm)
            g_loss = generator_criterion(output, trg_norm)

            running_results_g_loss += g_loss.item()

            output_unnorm = torch.zeros_like(output)
        
            output_unnorm[:,0,:,:] = 80*output[:,0,:,:] - 40
            output_unnorm[:,1,:,:] = 80*output[:,1,:,:] - 40
            output_unnorm[:,2,:,:] = 120*output[:,2,:,:] + 940
            
            wind_fake = torch.sqrt(torch.square(output_unnorm[:,0,:,:]) + torch.square(output_unnorm[:,1,:,:]))
            wind_fake_msk = wind_fake * diff_msk
            
            batch_mse = ((output - trg_norm) ** 2).data.mean()        
            batch_rmse_wind = torch.sqrt(((wind_fake - wind_real) ** 2).data.mean())
            batch_rmse95 = torch.sqrt(((wind_fake_msk - wind_real_msk) ** 2).data.sum() / diff_msk.data.sum())
            batch_rmse_slp = torch.sqrt(((output_unnorm[:,2,:,:] - trg[:,2,:,:]) ** 2).data.mean())
            
            mse_metrics += batch_mse        
            rmse_wind_metrics += batch_rmse_wind
            rmse_slp_metrics += batch_rmse_slp
            rmse95_metrics += batch_rmse95
            
            batch_ssim = ssim(output, trg_norm).item()
            ssims += batch_ssim       
            psnr_metrics = 10 * log10((output.max()**2) / (mse_metrics / (batch_idx+1)))
            ssim_metrics = ssims / (batch_idx + 1)
        
            pbar.update(1)
            if batch_idx >= VAL_STEPS-1:
                break
        pbar.close()

    g_loss = running_results_g_loss/VAL_STEPS
    rmse_wind_metrics = float((rmse_wind_metrics/VAL_STEPS).cpu())
    rmse_slp_metrics = float((rmse_slp_metrics/VAL_STEPS).cpu())
    rmse95_metrics = float((rmse95_metrics/VAL_STEPS).cpu())

    losses_dict = {'val_g_loss': g_loss}
    metrics_values = {'RMSE_Wind': rmse_wind_metrics,
                      'RMSE_SLP': rmse_slp_metrics,
                      'RMSE95': rmse95_metrics,
                      'PSNR': psnr_metrics,
                      'SSIM': ssim_metrics}
    return dict(**losses_dict, **metrics_values)


def main(start_epoch: int, NUM_EPOCHS: int, STEPS_PER_EPOCH: int, batch_size: int, VAL_STEPS: int, val_batch_size: int):
    
    torch.autograd.set_detect_anomaly(True)

    tboard_dir_train = os.path.join(os.path.abspath('./'), 'logs', 'unet', 'train')
    tboard_dir_val = os.path.join(os.path.abspath('./'), 'logs', 'unet', 'val')


    #cuda = True if torch.cuda.is_available() else False
    #if cuda:
    #    torch.cuda.set_device(0)
    #cuda_dev = torch.device('cuda:0')
    cuda = False

    print('creating the model')

    TB_writer_train = SummaryWriter(log_dir = tboard_dir_train)
    TB_writer_val = SummaryWriter(log_dir = tboard_dir_val)

    netG = Generator()
    #netG = netG.cuda()
    
    if start_epoch != 1:
        netG.load_state_dict(torch.load('epochs/unet/netG_epoch_%d.pth' % (start_epoch - 1)))
    else:
        pass

    generator_criterion = GeneratorLoss()

    optimizerG = optim.Adam(netG.parameters(),lr=2e-4)
    schedulerG = CosineAnnealingWarmRestarts(optimizerG, T_0=50, T_mult=2, eta_min=1.0e-9, lr_decay=0.75)
    
    train_ds = InputGenerator(data_index_fname, batch_size, debug = False)
    
    batches_queue_length = min(STEPS_PER_EPOCH, 64)
    
    train_batches_queue = Queue(maxsize=batches_queue_length)
    train_cuda_batches_queue = Queue(maxsize=4)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)
    preprocess_workers = 4
    
    for _ in range(preprocess_workers):
        thr = Thread(target=threaded_batches_feeder, args=(train_thread_killer, train_batches_queue, train_ds))
        thr.start()
    
    train_cuda_transfers_thread_killer = thread_killer()
    train_cuda_transfers_thread_killer.set_tokill(False)
    train_cudathread = Thread(target=threaded_cuda_batches, args=(train_cuda_transfers_thread_killer, train_cuda_batches_queue, train_batches_queue))
    train_cudathread.start()
    #endregion train dataset

    # region test dataset
    val_ds = InputGenerator(data_index_val_fname, val_batch_size,debug=False)
    batches_queue_length = min(VAL_STEPS, 64)
    
    val_batches_queue = Queue(maxsize=batches_queue_length)
    val_cuda_batches_queue = Queue(maxsize=4)
    val_thread_killer = thread_killer()
    val_thread_killer.set_tokill(False)
    
    for _ in range(preprocess_workers):
        thr = Thread(target=threaded_batches_feeder, args=(val_thread_killer, val_batches_queue, val_ds))
        thr.start()
    val_cuda_transfers_thread_killer = thread_killer()
    val_cuda_transfers_thread_killer.set_tokill(False)
    val_cudathread = Thread(target=threaded_cuda_batches, args=(val_cuda_transfers_thread_killer, val_cuda_batches_queue, val_batches_queue))
    val_cudathread.start()
    # endregion train dataset

    print('\n\nstart training')
    for epoch in range(start_epoch, NUM_EPOCHS+1):
        print('Train epoch: %d of %d' % (epoch, NUM_EPOCHS))
        train_metrics = train_epoch(netG, optimizerG, epoch, train_cuda_batches_queue,
                                    generator_criterion, STEPS_PER_EPOCH)
        print(str(train_metrics))
        
        print('\nValidation:')
        val_metrics = validation(netG, val_cuda_batches_queue, generator_criterion, VAL_STEPS)
        print(str(val_metrics))

        # note: this re-shuffling will not make an immediate effect since the queues are already filled with the
        # examples from the previous shuffle-states of datasets
        train_ds.shuffle()
        val_ds.shuffle()
        
        torch.save(netG.state_dict(), 'epochs/unet/netG_epoch_%d.pth' % (epoch))

        # region write losses to tensorboard
        TB_writer_train.add_scalar('g_loss', train_metrics['train_g_loss'], epoch)
        TB_writer_train.add_scalar('LR', schedulerG.get_last_lr()[-1], epoch)
        TB_writer_train.add_scalar('RMSE_Wind', train_metrics['RMSE_Wind'], epoch)
        TB_writer_train.add_scalar('RMSE_SLP', train_metrics['RMSE_SLP'], epoch)
        TB_writer_train.add_scalar('RMSE95', train_metrics['RMSE95'], epoch)
        TB_writer_train.add_scalar('PSNR', train_metrics['PSNR'], epoch)
        TB_writer_train.add_scalar('SSIM', train_metrics['SSIM'], epoch)

        TB_writer_val.add_scalar('g_loss', val_metrics['val_g_loss'], epoch)
        TB_writer_val.add_scalar('RMSE_Wind', val_metrics['RMSE_Wind'], epoch)
        TB_writer_val.add_scalar('RMSE_SLP', val_metrics['RMSE_SLP'], epoch)
        TB_writer_val.add_scalar('RMSE95', val_metrics['RMSE95'], epoch)
        TB_writer_val.add_scalar('PSNR', val_metrics['PSNR'], epoch)
        TB_writer_val.add_scalar('SSIM', val_metrics['SSIM'], epoch)
        # endregion

        #text_file = open("train.txt", "a")
        #text_file.write(str(epoch) + " " +
        #                str(schedulerG.get_last_lr()[-1]) + " " +
        #                str(train_metrics['RMSE_Wind']) + " " +
        #                str(train_metrics['RMSE_SLP']) + " "+
        #                str(train_metrics['RMSE95'])+ " "+
        #                str(train_metrics['PSNR'])+"\n"
        #                )
        #text_file.close()

        #text_file = open("val.txt", "a")
        #text_file.write(str(epoch) + " " +
        #            str(val_metrics['RMSE_Wind']) + " " +
        #            str(val_metrics['RMSE_SLP']) + " " +
        #            str(val_metrics['RMSE95']) + " " +
        #            str(val_metrics['PSNR']) + "\n"
        #            )
        #text_file.close()

        schedulerG.step(epoch=epoch)

    train_thread_killer.set_tokill(True)
    train_cuda_transfers_thread_killer.set_tokill(True)
    val_thread_killer.set_tokill(True)
    val_cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            # Enforcing thread shutdown
            train_batches_queue.get(block=True, timeout=1)
            train_cuda_batches_queue.get(block=True, timeout=1)
            val_batches_queue.get(block=True, timeout=1)
            val_cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass

main(start_epoch = 1, NUM_EPOCHS = 750, STEPS_PER_EPOCH = 10, batch_size = 8, VAL_STEPS = 10, val_batch_size = 1)




