from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torchvision
#from resnetx import ModifiedResNet18
from MyNetfangbohu import FCNet, FeatureNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from scipy.stats import spearmanr

use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True



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
        #im 
        im = Image.open(img_name).convert('RGB')


        rating = self.images_frame.iloc[idx, 1]

        im = self.transform(im)

        return im,  rating

class Net(nn.Module):
    def __init__(self, headnet, net):
        super(Net, self).__init__()
        self.headnet = headnet
        self.net = net

    def forward(self, x1):
        f1 = self.headnet(x1)
        output = self.net(f1)
        return output


def computeSpearman(dataloader_valid, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        for batch_idx, (image,score) in enumerate(tqdm(dataloader_valid)):
            inputs_im = image
           
            batch_size = inputs_im.size()[0]
            labels = score.view(batch_size, -1)
            # labels = labels / 100.0
            if use_gpu:
                try:
                    inputs_im= inputs_im.float().cuda()
                    labels = labels.float().cuda()
                except:
                    print(inputs_im, labels)
            else:
                inputs_im, labels = inputs_im.float(), labels.float()
            outputs_a = model(inputs_im)
            ratings.append(labels.float())
            predictions.append(outputs_a.float())

   
    ratings_i = np.vstack([r.cpu().numpy() for r in ratings])
    predictions_i = np.vstack([p.cpu().numpy() for p in predictions])
    a = ratings_i[:, 0]
    b = predictions_i[:, 0]
    sp = spearmanr(a, b)
    return sp


def train_model():
    epochs = 20
    task_num = 7
    noise_num1 = 15
    noise_num2 = 10

    #model = ModifiedResNet18(1)
    net1 = FeatureNet()
    net2 = FCNet()
    model = Net(headnet=net1, net=net2)
    print("create successnewmsfb")
    criterion = nn.L1Loss()
    ignored_params = list(map(id, model.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.parameters(), 'lr': 1e-4}
    ], lr=1e-4)
    model.cuda()
    meta_model = copy.deepcopy(model)
    temp_model = copy.deepcopy(model)

    spearman = 0

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer = exp_lr_scheduler(optimizer, epoch)

        list_noise = list(range(noise_num1))
        np.random.shuffle(list_noise)
        print('############# UID2021 train phase epoch %2d ###############' % epoch)
        count = 0
        for index in list_noise:

            if count % task_num == 0:
                name_to_param = dict(temp_model.named_parameters())
                for name, param in meta_model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff)

            name_to_param = dict(model.named_parameters())
            for name, param in temp_model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            dataloader_train, dataloader_valid = load_data('train', 'UID2021', index)
            if dataloader_train == 0:
                continue
            dataiter = iter(enumerate(dataloader_valid))
            model.train()  # Set model to training mode
            # Iterate over data.

            for batch_idx, (image,  score) in enumerate(tqdm(dataloader_train)):

                inputs_im = image
                
                batch_size = inputs_im.size()[0]
                labels = score.view(batch_size, -1)
                # labels = labels / 100.0
                if use_gpu:
                    try:
                        inputs_im = inputs_im.float().cuda()
                        labels = labels.float().cuda()
                    except:
                        print(inputs_im,  labels)
                else:
                    inputs_im, labels = inputs_im.float(), labels.float()

                optimizer.zero_grad()
                outputs = model(inputs_im)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                idx, (image,  score) = next(dataiter)
                if idx >= len(dataloader_valid) - 1:
                    dataiter = iter(enumerate(dataloader_valid))
                inputs_im_val = image
               
                batch_size1 = inputs_im_val.size()[0]
                labels_val = score.view(batch_size1, -1)
                # labels_val = labels_val / 10.0
                if use_gpu:
                    try:
                        inputs_im_val = inputs_im_val.float().cuda()
                        labels_val = labels_val.float().cuda()

                    except:
                        print(inputs_im_val, labels_val)
                else:
                    inputs_im_val = inputs_im_val.float()
                    labels_val = labels_val.float()

                optimizer.zero_grad()
                outputs_val = model(inputs_im_val)
                loss_val = criterion(outputs_val, labels_val)
                loss_val.backward()
                optimizer.step()

                try:
                    running_loss += loss_val.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                name_to_param1 = dict(meta_model.named_parameters())
                name_to_param2 = dict(temp_model.named_parameters())
                for name, param in model.named_parameters():
                    diff = param.data - name_to_param2[name].data
                    name_to_param1[name].data.add_(diff / task_num)

                count += 1
        # print('trying epoch loss')
        epoch_loss = running_loss / count
        print('current loss = ', epoch_loss)

        running_loss = 0.0
        list_noise = list(range(noise_num2))
        np.random.shuffle(list_noise)
        # list_noise.remove(ii)
        print('############# SOTA train phase epoch %2d ###############' % epoch)
        count = 0
        for index in list_noise:
            if count % task_num == 0:
                name_to_param = dict(temp_model.named_parameters())
                for name, param in meta_model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff)

            name_to_param = dict(model.named_parameters())
            for name, param in temp_model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            dataloader_train, dataloader_valid = load_data('train', 'SOTA', index)
            if dataloader_train == 0:
                continue
            dataiter = iter(enumerate(dataloader_valid))
            model.train()  # Set model to training mode

            # Iterate over data.
            for batch_idx, (image,  score) in enumerate(tqdm(dataloader_train)):

                inputs_im = image
               
                batch_size = inputs_im.size()[0]
                labels = score.view(batch_size, -1)
                #labels = (labels - 0.5) / 5.0
                if use_gpu:
                    try:
                        inputs_im = inputs_im.float().cuda()
                        labels = labels.float().cuda()
                    except:
                        print(inputs_im,  labels)
                else:
                    inputs_im, labels = inputs_im.float(), labels.float()

                optimizer.zero_grad()
                outputs = model(inputs_im)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                idx, (image,  score) = next(dataiter)
                if idx >= len(dataloader_valid) - 1:
                    dataiter = iter(enumerate(dataloader_valid))
                inputs_im_val = image
             
                batch_size1 = inputs_im_val.size()[0]
                labels_val = score.view(batch_size1, -1)
                #labels_val = (labels_val - 0.5) / 5.0
                if use_gpu:
                    try:
                        inputs_im_val= inputs_im_val.float().cuda()
                        labels_val = labels_val.float().cuda()

                    except:
                        print(inputs_im_val, labels_val)
                else:
                    inputs_im_val = inputs_im_val.float()
                    labels_val = labels_val.float()

                optimizer.zero_grad()
                outputs_val = model(inputs_im_val)
                loss_val = criterion(outputs_val, labels_val)
                loss_val.backward()
                optimizer.step()

                try:
                    running_loss += loss_val.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                name_to_param = dict(meta_model.named_parameters())
                for name, param in model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff / task_num)

                count += 1
        # print('trying epoch loss')
        epoch_loss = running_loss / count
        print('current loss = ', epoch_loss)

        print('############# test phase epoch %2d ###############' % epoch)
        dataloader_train, dataloader_valid = load_data('test', 0)
        model.eval()
        model.cuda()
        sp = computeSpearman(dataloader_valid, model)[0]
        if sp > spearman:
            spearman = sp
            best_model = copy.deepcopy(model)
            # best_model = copy.deepcopy(meta_model)
            # torch.save(best_model.cuda(),
            #        'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18.pt')
        print('new srocc {:4f}, best srocc {:4f}'.format(sp, spearman))

    # torch.save(model.cuda(),
    #       'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18.pt')
    torch.save(model.cuda().state_dict(),
               '/home/user/Vit_Add_Eff/opensource/pre_model/UID2021_SOTA_IQA_Meta_UIQASFTNet.pt')


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=2):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate = 0.9 ** (epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer


def my_collate(batch):  # 过滤样本的 去除None
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


output_size = (384, 384)
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((448, 448)),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.RandomCrop(size=output_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

])
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((384, 384)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def normalization(data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range
def load_data(mod='train', dataset='UID2021', worker_idx=0):
    if dataset == 'UID2021':
        data_dir = os.path.join('/home/user/Database/UID2021')
        worker_original = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_per_noise.csv'), sep=',')
        scores=worker_original['mos']
        normalized_scores = normalization(scores)
        worker_original['mos'] = normalized_scores
        image_path = '/home/user/Database/UID2021/'
    else:
        data_dir = os.path.join('/home/user/Database/SOTA_Dataset')
        worker_original = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_per_noise.csv'), sep=',')
        scores=worker_original['mos']
        normalized_scores = normalization(scores)
        worker_original['mos'] = normalized_scores
        image_path = '/home/user/Database/SOTA_Dataset/'
    workers_fold = "noise_UID2021_SOTA/"
    if not os.path.exists(workers_fold):
        os.makedirs(workers_fold)

    worker = worker_original['noise'].unique()[worker_idx]
    print("----worker number: %2d---- %s" % (worker_idx, worker))
    if mod == 'train':
        percent = 0.8
        images = worker_original[worker_original['noise'].isin([worker])][['image', 'mos']]

        train_dataframe, valid_dataframe = train_test_split(images, train_size=percent)
        train_path = workers_fold + "train_scores_" + str(worker) + ".csv"
        test_path = workers_fold + "test_scores_" + str(worker) + ".csv"
        train_dataframe.to_csv(train_path, sep=',', index=False)
        valid_dataframe.to_csv(test_path, sep=',', index=False)

        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir=image_path,
                                                        transform=train_transforms)
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir=image_path,
                                                        transform=test_transforms)
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=20,
                                      shuffle=True, num_workers=4, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=20,
                                      shuffle=False, num_workers=4, collate_fn=my_collate)
    else:
        cross_data_path = '/home/user/Database/UIED/image_labeled_by_score.csv'
        transformed_dataset_valid_1 = ImageRatingsDataset(csv_file=cross_data_path,
                                                          root_dir='/home/user/Database/UIED/',
                                                          transform=test_transforms)
        dataloader_train = 0
        dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size=20,
                                      shuffle=False, num_workers=4, collate_fn=my_collate)

    return dataloader_train, dataloader_valid


train_model()
