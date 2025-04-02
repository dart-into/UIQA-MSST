from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
import torchvision

import cv2 as cv
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, pearsonr
from UIQASFTNet import FCNet, FeatureNet
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

#torch.backends.cudnn.benchmark = True

##########################################


os.environ["CUDA_VISIBLE_DEVICES"]="0"
ModelLoad_path = 'pre_model/UID2021_SOTA_IQA_Meta_UIQASFTNet.pt'
ModelSave_path='finetune_model/UID2021_SOTA_IQA_Meta_UIQASFTNet_FineUIED.pt'
ResultSave_path='result/UID2021_SOTA_UIQASFTNet_FineUIED.txt'


##########################################





class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
        img_name = str(os.path.join(self.root_dir, str(self.images_frame.iloc[idx, 0])))
   
        
        # im
        im = Image.open(img_name).convert('RGB')
       
       
        rating = self.images_frame.iloc[idx, 1]

        im = self.transform(im)
      

        return im, rating



def computeSpearman(dataloader_valid, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        for batch_idx, (image, score) in enumerate(tqdm(dataloader_valid)):
            inputs_im = image
          
            batch_size = inputs_im.size()[0]
            labels = score.view(batch_size, -1)
            # labels = labels / 100.0
            if use_gpu:
                try:
                    inputs_im= inputs_im.float().cuda()
                    labels = labels.float().cuda()
                except:
                    print(inputs_im,  labels)
            else:
                inputs_im,labels = inputs_im.float(), labels.float()
            outputs_a = model(inputs_im)
            ratings.append(labels.float())
            predictions.append(outputs_a.float())

    # ratings_i = np.vstack(ratings)
    # predictions_i = np.vstack(predictions)

    ratings_i = np.vstack([r.cpu().numpy() for r in ratings])
    predictions_i = np.vstack([p.cpu().numpy() for p in predictions])
    a = ratings_i[:, 0]
    b = predictions_i[:, 0]
    
    sp = spearmanr(a, b)[0]
    pl = pearsonr(a, b)[0]
    
    return sp, pl


class Net(nn.Module):
    def __init__(self, headnet, net):
        super(Net, self).__init__()
        self.headnet = headnet
        self.net = net

    def forward(self, x1):
        f1 = self.headnet(x1)
        output = self.net(f1)
        return output
def normalization(data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range
def finetune_model():
    epochs =55
    
    srocc_l = []
    best_srocc = 0
    print('=============Saving Finetuned Prior Model===========')
    data_dir = os.path.join('database/UIED')
    images = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_score.csv'), sep=',')
    scores=images['mos']
    normalized_scores = normalization(scores)
    images['mos'] = normalized_scores
    images_fold = "database/UIED/"
  
    torch.manual_seed(20000615)
    if not os.path.exists(images_fold):
        os.makedirs(images_fold)
    for i in range(0,10):
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            
            print(i,file=f1)
        images_train, images_test = train_test_split(images, train_size = 0.8)
        train_path = images_fold + "train_imagenormal" +str(i+1) +".csv"
        test_path = images_fold + "test_imagenormal" +str(i+1) + ".csv"
        images_train.to_csv(train_path, sep=',', index=False)
        images_test.to_csv(test_path, sep=',', index=False)

        net1 = FeatureNet()
        net2 = FCNet()
        model = Net(headnet=net1, net=net2)
        model.load_state_dict(torch.load(ModelLoad_path))
        
        
  

       
        criterion = nn.L1Loss()
        #criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4,  weight_decay=0)
        model.cuda()

        spearman = 0
        plcc = 0
        for epoch in range(epochs):
            optimizer = exp_lr_scheduler(optimizer, epoch)

            if epoch == 0:
                dataloader_valid = load_data('train',i)
                model.eval()

                sp,pl = computeSpearman(dataloader_valid, model)
                if sp > spearman:
                    spearman = sp
                    plcc=pl
                print('no train srocc {:4f} plcc {:4f}'.format(sp,pl))


            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train = load_data('train',i)
            model.train()  # Set model to training mode
            for batch_idx, (image,  score) in enumerate(tqdm(dataloader_train)):

                inputs_im = image
              
                batch_size = inputs_im.size()[0]
                labels = score.view(batch_size, -1)
                # labels = labels / 100.0
                if use_gpu:
                    try:
                        inputs_im  = inputs_im.float().cuda()
                        labels = labels.float().cuda()
                    except:
                        print(inputs_im, labels)
                else:
                    inputs_im, labels = inputs_im.float(),  labels.float()

                optimizer.zero_grad()
                outputs = model(inputs_im)
                #print(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            #print('############# test phase epoch %2d ###############' % epoch)
            dataloader_valid = load_data('test',i)
            model.eval()

            sp, pl = computeSpearman(dataloader_valid, model)
            if sp > spearman:
                spearman = sp
                plcc=pl
            if sp > best_srocc:
                best_srocc = sp
                print('=====Prior model saved===Srocc:%f========'%best_srocc)
                best_model = copy.deepcopy(model)
                torch.save(best_model.cuda(),ModelSave_path)
          
            print('Validation Results - Epoch: {:2d}, PLCC: {:4f}, SROCC: {:4f}, '
                  'best SROCC: {:4f}'.format(epoch, pl, sp, spearman))

        srocc_l.append(spearman)
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print('{:4f},{:4f}'.format(plcc, spearman),file=f1)
   

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=13):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.9**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


output_size = (384, 384)
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((448, 448)),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.RandomCrop(size=output_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    #torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
])
test_transforms = torchvision.transforms.Compose([
    
    torchvision.transforms.Resize((384, 384)),
    
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    #torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])



def load_data(mod = 'train',i=0):

    bsize = 12
    data_dir = os.path.join('database/UIED/')
    traincsv_name="train_imagenormal" +str(i+1) +".csv"
    testcsv_name="test_imagenormal" +str(i+1) + ".csv"
    train_path = os.path.join(data_dir,  traincsv_name)
    test_path = os.path.join(data_dir,  testcsv_name)


    transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='database/UIED',
                                                    transform=train_transforms)
    transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='database/UIED',
                                                    transform=test_transforms)

    if mod == 'train':
        dataloader = DataLoader(transformed_dataset_train, batch_size=bsize,
                                  shuffle=True, num_workers=4, collate_fn=my_collate)
    else:
        dataloader = DataLoader(transformed_dataset_valid, batch_size= 12,
                                    shuffle=False, num_workers=4, collate_fn=my_collate)

    return dataloader

finetune_model()
