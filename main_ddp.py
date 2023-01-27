from src.dataset import CustomDataset
from src.models import resnet,de_resnet 
from src.test import evaluation
from src.log import setup_default_logging
from torch.utils.data import DataLoader,DataLoader2
import torch.nn as nn 
import torch.nn.functional as F 
import torch 
import torchvision.transforms as transforms 
import yaml 
from tqdm.auto import tqdm 
import numpy as np 
import random 
from tqdm import tqdm 
import os 
import argparse 
from accelerate import Accelerator,DistributedDataParallelKwargs,logging 
import warnings 
import time 
warnings.filterwarnings('ignore')

_logger = logging.get_logger('train')

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
    setup_seed(cfg['TRAIN']['seed'])
    
    # setting Accelerator 
    # ! ddp_scaler 반드시 추가 해주어야 함 
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])
    device = accelerator.device
    
    # set logger 
    setup_default_logging(log_path=os.path.join(cfg['SAVE']['savedir'],'log.txt'))
    _logger.info('Device: {}'.format(device))
    
    
    # build dataloader 
    trainloader = DataLoader(
                dataset = CustomDataset(
                                        root          = cfg['DATA']['datadir'],
                                        img_size      = cfg['DATA']['imgsize'],
                                        transform     = transforms.Compose([transforms.ToTensor()]),
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
                                        transform     = transforms.Compose([transforms.ToTensor()]),
                                        img_cls       = cfg['DATA']['imgcls'],
                                        mode          = cfg['DATA']['mode'],
                                        train         = False
                                        ),
                #batch_size = cfg['TRAIN']['batchsize'], 
                batch_size = 1,
                shuffle    = True
                )

    # build encoder, bn ,decoder 
    encoder,bn = __import__('src').models.resnet.__dict__[f"{cfg['MODEL']['encoder']}"](pretrained=True)
    #encoder,bn = encoder.to(device), bn.to(device)
    encoder,bn = encoder.to(accelerator.device),bn.to(accelerator.device)
    encoder.eval()
    
    decoder = __import__('src').models.de_resnet.__dict__[f"{cfg['MODEL']['decoder']}"]()
    #decoder = decoder.to(device)
    decoder.to(accelerator.device)
    
    # set optimizer 
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()),lr = cfg['TRAIN']['lr'],betas=(0.5,0.999))
    accelerator.print("All loaded, train start")
    run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers(run,cfg)
    
    # accelerator set and prepare 
    trainloader,testloader,encoder,bn,decoder,optimizer = accelerator.prepare(
        trainloader,testloader,encoder,bn,decoder,optimizer)
    
    # Training Start 
    _logger.info("All loaded, train start")
    train(trainloader,testloader,encoder,bn,decoder,optimizer,cfg,accelerator)
    
def train(trainloader,testloader,encoder,bn,decoder,optimizer,cfg,accelerator):
    
    
    batch_time_m = AverageMeter() # 배치 1개 도는 시간  
    data_time_m = AverageMeter() #데이터 뽑는 시간 
    epoch_time_m = AverageMeter()
    
    batch_end = time.time() 
    epoch_end = time.time()
    
    best = 0 
    for epoch in range(cfg['TRAIN']['epochs']):
        bn.train()
        decoder.train()
        loss_list = [] 
        for batch_imgs,_,batch_labels in trainloader:
            data_time_m.update(time.time()-batch_end)
            #batch_imgs = batch_imgs.to(device)
            
            # predict 
            inputs = encoder(batch_imgs)
            outputs = decoder(bn(inputs))
            
            loss = loss_function(inputs,outputs)
            
            # loss update 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
            # batch time update 
            batch_time_m.update(time.time()-batch_end)
            batch_end = time.time()
            
        # epoch time update 
        epoch_time_m.update(time.time()-epoch_end) 
        epoch_end = time.time()  
        
        # logging     
        #accelerator.print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1,cfg['TRAIN']['epochs'] , np.mean(loss_list)))
        _logger.info(f"epoch: [{epoch+1}/{cfg['TRAIN']['epochs']}]|"
                     f"loss: {np.mean(loss_list):.4f}|"
                     f"lr = {optimizer.param_groups[0]['lr']:.4f}|"
                     f"batch_time = {batch_time_m.val:.4f}|"
                     f"epoch_time = {epoch_time_m.val:.4f}|"
                     f"total time = {epoch_time_m.sum:.4f}|"
                     )
        
        # calculate metric 
        if (epoch +1) % 2 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, testloader, accelerator.device)
            #accelerator.print('Pixel Auroc:{:.3f} | Sample Auroc:{:.3f} | Pixel Aupro:{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            _logger.info(f' Pixel Auroc:{auroc_px:.3f} | Sample Auroc:{auroc_sp:.3f} | '
                         f' Pixel Aupro:{auroc_px:.3f} | ')
            
            # check point save 
            if auroc_px > best:
                torch.save(decoder,os.path.join(cfg['SAVE']['savedir'],'best_decoder.pt'))
                torch.save(bn,os.path.join(cfg['SAVE']['savedir'],'best_bn.pt'))
                #accelerator.save_state(os.path.join(cfg['SAVE']['savedir']))
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
    end = time.time() 
    cfg = init()
    
    run(cfg)
    print(time.time() -end)
    