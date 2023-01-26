from src.dataset import CustomDataset

from src.models import resnet,de_resnet 
from src.test import evaluation
from src.augmentation import get_data_transforms
from src.log import setup_default_logging

from torch.utils.data import DataLoader,DataLoader2
import torch.nn as nn 
import torch.nn.functional as F 
import torch 
import torchvision.transforms as transforms 

import yaml 
import time 
import numpy as np 
import random 
import os 
import argparse 
import logging 

_logger = logging.getLogger('train') 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_function(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def run(cfg):
    
    # setting savedir and seed 
    setup_seed(cfg['TRAIN']['seed'])
    device = cfg['TRAIN']['device']
    
    # set logger 
    setup_default_logging(log_path=os.path.join(cfg['SAVE']['savedir'],'log.txt'))
    _logger.info('Device: {}'.format(device))
    
    
    #build augmentation 
    data_transform, gt_transform = get_data_transforms(cfg['DATA']['imgsize'],cfg['DATA']['imgsize'])
    
    #build datasets and dataloader 
    trainloader = DataLoader(
                    dataset = CustomDataset(
                                            root          = cfg['DATA']['datadir'],
                                            img_size      = cfg['DATA']['imgsize'],
                                            transform     = data_transform,
                                            img_cls       = cfg['DATA']['imgcls'],
                                            mode          = cfg['DATA']['mode'],
                                            train         = True 
                                            ),
                    batch_size = cfg['TRAIN']['batchsize'], 
                    shuffle    = True
                    )
                        
    testloader = DataLoader(
                    dataset = CustomDataset(
                                            root          = cfg['DATA']['datadir'],
                                            img_size      = cfg['DATA']['imgsize'],
                                            transform     = gt_transform,
                                            img_cls       = cfg['DATA']['imgcls'],
                                            mode          = cfg['DATA']['mode'],
                                            train         = False
                                            ),
                    #batch_size = cfg['TRAIN']['batchsize'], 
                    batch_size = 1,
                    shuffle    = False
                    )
    
    
    # build encoder,bn and decoder 
    #encoder,bn = resnet34(pretrained=True)
    encoder,bn = __import__('src').models.resnet.__dict__[f"{cfg['MODEL']['encoder']}"](pretrained=True)
    encoder,bn = encoder.to(device), bn.to(device)
    encoder.eval()
    #decoder = de_resnet34(pretrained=False)
    decoder = __import__('src').models.de_resnet.__dict__[f"{cfg['MODEL']['decoder']}"]()
    decoder = decoder.to(device)
    
    # set optimizer 
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()),lr = cfg['TRAIN']['lr'],betas=(0.5,0.999))
    _logger.info("All loaded, train start")
    train(trainloader,testloader,encoder,bn,decoder,optimizer,cfg)
    
def train(trainloader,testloader,encoder,bn,decoder,optimizer,cfg):
    # set AverageMeter 
    batch_time_m = AverageMeter() # 배치 1개 도는 시간  
    data_time_m = AverageMeter() #데이터 뽑는 시간 
    epoch_time_m = AverageMeter()
    
    batch_end = time.time() 
    epoch_end = time.time()
    
    best = 0 
    device = cfg['TRAIN']['device']
    for epoch in range(cfg['TRAIN']['epochs']):
        bn.train()
        decoder.train()
        loss_list = [] 
        for batch_imgs,_,batch_labels in trainloader:
            data_time_m.update(time.time()-batch_end)
            
            batch_imgs = batch_imgs.to(device)
            
            # predict 
            inputs = encoder(batch_imgs)
            outputs = decoder(bn(inputs))
            
            loss = loss_function(inputs,outputs)
            
            # loss update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
            batch_time_m.update(time.time()-batch_end)
            batch_end = time.time()
            
        epoch_time_m.update(time.time()-epoch_end) 
        epoch_end = time.time()  
        #logging 
        _logger.info(f"epoch: [{epoch+1}/{cfg['TRAIN']['epochs']}] "
                     f"loss: {np.mean(loss_list):.4f} "
                     f"lr = {optimizer.param_groups[0]['lr']:.4f} "
                     f"batch_time = {batch_time_m.val:.4f} "
                     f"epoch_time = {epoch_time_m.val:.4f} "
                     )
        
        if (epoch +1) % 2 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, testloader, device)
            _logger.info(f'Pixel Auroc:{auroc_px:.3f} | Sample Auroc:{auroc_sp:.3f} | '
                         'Pixel Aupro:{auroc_px:.3}')
            
            if auroc_px > best:
                torch.save(decoder,os.path.join(cfg['SAVE']['savedir'],'best_decoder.pt'))
                torch.save(bn,os.path.join(cfg['SAVE']['savedir'],'best_bn.pt'))
                _logger.info(f"New Best model saved - Epoch : {epoch+1}")
                best = auroc_px
            
    torch.save(decoder,os.path.join(cfg['SAVE']['savedir'],'last_decoder.pt'))
    torch.save(bn,os.path.join(cfg['SAVE']['savedir'],'last_bn.pt'))
    
def init():
    # configs 
    parser = argparse.ArgumentParser(description='RD')
    parser.add_argument('--yaml_config', type=str, default='./configs/mvtec.yaml', help='exp config file')    
    args = parser.parse_args()
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    
    # save point 
    savedir = os.path.join(cfg['SAVE']['savedir'],cfg['DATA']['dataset'],cfg['DATA']['imgcls'])
    try:
        os.mkdir(savedir)
    except:
        pass
    
    # save configs 
    cfg['SAVE']['savedir'] = savedir
    with open(f"{savedir}/config.yaml",'w') as f:
            yaml.dump(cfg,f)
            
    return cfg 

if __name__ == '__main__':
    cfg = init()
    
    run(cfg)
    