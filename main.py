from src.dataset import CustomDataset
from src.models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from src.models.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from test import evaluation
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

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return 

def run(cfg):
    setup_seed(cfg['TRAIN']['seed'])
    device = cfg['TRAIN']['device']
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
                    shuffle    = False
                    )

    encoder,bn = resnet34(pretrained=True)
    encoder,bn = encoder.to(device), bn.to(device)
    encoder.eval()
    decoder = de_resnet34(pretrained=False)
    decoder = decoder.to(device)
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()),lr = cfg['TRAIN']['lr'],betas=(0.5,0.999))
    print("All loaded, train start")
    train(trainloader,testloader,encoder,bn,decoder,optimizer,cfg)
    
def train(trainloader,testloader,encoder,bn,decoder,optimizer,cfg):
    best = 0 
    device = cfg['TRAIN']['device']
    for epoch in tqdm(range(cfg['TRAIN']['epochs'])):
        bn.train()
        decoder.train()
        loss_list = [] 
        for batch_imgs,_,batch_labels in trainloader:
            batch_imgs = batch_imgs.to(device)
            inputs = encoder(batch_imgs)
            outputs = decoder(bn(inputs))
            loss = loss_function(inputs,outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1,cfg['TRAIN']['epochs'] , np.mean(loss_list)))
        
        if (epoch +1) % 2 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, testloader, device)
            print('Pixel Auroc:{:.3f} | Sample Auroc:{:.3f} | Pixel Aupro:{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            
            if auroc_px > best:
                torch.save(decoder,os.path.join(cfg['SAVE']['savedir'],'best_decoder.pt'))
                torch.save(bn,os.path.join(cfg['SAVE']['savedir'],'best_bn.pt'))
                
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
    