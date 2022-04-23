from datetime import datetime
import sys
import argparse
import random
import time
from pathlib import Path
import os
from types import SimpleNamespace
from urllib.error import HTTPError
import pandas as pd
import numpy as np
from skimage import io
import pytorch_lightning as pl
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.loops import Loop
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Callable, Union, Optional

# from IPython.display import HTML, display, set_matplotlib_formats
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torchvision import transforms
import urllib.request
# import ssl

best_acc = 0

class Args:
    root_dir = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/Dataset'
    lib_dir = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/CRC_DX_Lib'
    output_path = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/Output'
    batch_size = 128
    nepochs = 10
    num_workers = 4
    test_every = 10
    weights = 0.5
    k = 1

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--root_dir",
    type=Path,
    required=True,
    help="root directory of dataset",
)
parser.add_argument(
    "--lib_dir",
    type=Path,
    required=True,
    help="root directory of libraryfile",
)
parser.add_argument(
    "--output_path",
    type=Path,
    required=True,
    help="output directory",
)
parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
    help="batch size",
)
parser.add_argument(
    "--num_workers",
    default=0,
    type=int,
    required=True,
    help="number of workers",
)
parser.add_argument(
    "--nepochs",
    default=50,
    type=int,
    help="training epoch",
)
parser.add_argument(
    '--test_every',
    default=10,
    type=int,
    help='test on val every (default: 10)')

parser.add_argument(
    "--weights",
    default=0.5,
    type=float,
    help="unbalanced positive class weight (default: 0.5, balanced classes)",
)

parser.add_argument(
    "--k",
    default=1,
    type=int,
    help="top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)",
)

def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            # print(
            #     'Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size +
                input.size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)


def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    pos = np.equal(pred, real)
    neq = np.not_equal(pred, real)
    acc = float(pos.sum())/pred.shape[0]
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred == 1, neq).sum())/(real == 0).sum()
    fnr = float(np.logical_and(pred == 0, neq).sum())/(real == 1).sum()
    return acc, err, fpr, fnr


def group_argtopk(groups, data, k=1):
    # groups in slide, data is prob of each tile
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])  # output top prob tile index in each slide


def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out


class MILdataset(Dataset):
    def __init__(self, libraryfile_dir='', root_dir='', dataset_mode='Train', transform=None, subset_rate=None):
        libraryfile_path = os.path.join(
            libraryfile_dir, f'CRC_DX_{dataset_mode}_ALL.csv')
        lib = pd.read_csv(libraryfile_path)
        lib = lib if subset_rate is None else lib.sample(
            frac=subset_rate, random_state=2022)
        lib = lib.sort_values(['subject_id'], ignore_index=True)
        lib.to_csv(os.path.join(libraryfile_dir,
                   f'{dataset_mode}_temporary.csv'))
        slides = []
        for i, name in enumerate(lib['subject_id'].unique()):
            sys.stdout.write(
                'Slides: [{}/{}]\r'.format(i+1, len(lib['subject_id'].unique())))
            sys.stdout.flush()
            slides.append(name)

        # Flatten grid
        grid = []
        slideIDX = []
        for i, g in enumerate(lib['subject_id'].unique()):
            tiles = lib[lib['subject_id'] == g]['slice_id']
            grid.extend(tiles)
            slideIDX.extend([i]*len(tiles))

        # print('Number of tiles: {}'.format(len(grid)))
        self.dataframe = self.load_data_and_get_class(lib)
        self.slidenames = list(lib['subject_id'].values)
        self.slides = slides
        self.targets = self.dataframe['Class']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.root_dir = root_dir
        self.dset = f"CRC_DX_{dataset_mode}"

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x], self.grid[x],
                        self.targets[x]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def load_data_and_get_class(self, df):
        df.loc[df['label']=='MSI', 'Class'] = 1
        df.loc[df['label']=='MSS', 'Class'] = 0
        return df

    def __getitem__(self, index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            tile_id = self.grid[index]
            slide_id = self.slides[slideIDX]
            img_name = "blk-{}-{}.png".format(tile_id, slide_id)
            target = self.targets[index]
            label = 'CRC_DX_MSIMUT' if target == 1 else 'CRC_DX_MSS'
            img_path = os.path.join(self.root_dir, self.dset, label, img_name)
            img = io.imread(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, tile_id, target = self.t_data[index]
            slide_id = self.slides[slideIDX]
            label = 'CRC_DX_MSIMUT' if target == 1 else 'CRC_DX_MSS'
            img_name = "blk-{}-{}.png".format(tile_id, slide_id)
            img_path = os.path.join(self.root_dir, self.dset, label, img_name)
            img = io.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

class Lite(LightningLite):

    def run(self, learning_rate):
        global args, best_acc
        # args = parser.parse_args()
        args = Args()
        self.seed_everything(2022)
        model_name = "resnet18"
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=1e-4)
        if args.weights == 0.5:
            criterion = nn.CrossEntropyLoss()
        else:
            w = torch.Tensor([1-args.weights, args.weights])
            criterion = nn.CrossEntropyLoss(w)
        # Scale model and optimizers
        model, optimizer = self.setup(model, optimizer, move_to_device=True)

        DATA_MEANS = [0.485, 0.456, 0.406]
        DATA_STD = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(DATA_MEANS, DATA_STD)])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEANS, DATA_STD)])

        train_dataset = MILdataset(
            args.lib_dir, args.root_dir, 'Train', transform=train_transform, subset_rate=0.001)
        val_dataset = MILdataset(
            args.lib_dir, args.root_dir, 'Val', transform=test_transform, subset_rate=0.001)
        test_dataset = MILdataset(
            args.lib_dir, args.root_dir, 'Test', transform=test_transform, subset_rate=0.001)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, pin_memory=True)
        train_dataloader, val_dataloader, test_dataloader = self.setup_dataloaders(
            train_dataloader, val_dataloader, test_dataloader, move_to_device=True)

        # open output file
        fconv=open(os.path.join(args.output_path, f'{random.getrandbits(8)}_prob_output_{model_name}.csv'), 'w')
        fconv.write('epoch,metric,value\n')
        fconv.close()

        for epoch in tqdm(range(args.nepochs)):
            train_dataset.setmode(1)
            print("train_set_len:", len(train_dataloader.dataset))
            probs = inference(epoch, train_dataloader, model)
            # return the indices of topk tile(s) in each slides
            topk = group_argtopk(np.array(train_dataset.slideIDX), probs, args.k)
            train_dataset.maketraindata(topk)
            train_dataset.shuffletraindata()
            train_dataset.setmode(2)

            model.train()
            running_loss = 0.
            for i, (input, target) in enumerate(train_dataloader):
                output = model(input)
                loss = criterion(output, target.long())
                optimizer.zero_grad()
                self.backward(loss)
                optimizer.step()
                running_loss += loss.item()*input.size(0)
            batch_loss = running_loss/len(train_dataloader.dataset)
            print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, batch_loss))
            fconv = open(os.path.join(
                args.output_path, 'convergence.csv'), 'a')
            fconv.write('{},loss,{}\n'.format(epoch+1, loss))
            fconv.close()

            # Validation
            if (epoch+1) % args.test_every == 0:
                val_dataset.setmode(1)
                probs = inference(epoch, val_dataloader, model)
                # maxs = group_max(np.array(val_dataset.slideIDX),probs, len(val_dataset.targets))
                pred = [1 if x >= 0.5 else 0 for x in probs]
                print(f"pred in epoch{epoch}:{pred}")
                print(f"target in epoch{epoch}:{val_dataset.targets}")
                acc, err, fpr, fnr = calc_err(pred, val_dataset.targets)

                print('Validation\tEpoch: [{}/{}]\t ACC: {}\tError: {}\tFPR: {}\tFNR: {}'.format(
                    epoch+1, args.nepochs, acc, err, fpr, fnr))
                fconv = open(os.path.join(
                    args.output_path, 'convergence.csv'), 'a')

                fconv.write('{},acc,{}\n'.format(epoch+1, acc))    
                fconv.write('{},error,{}\n'.format(epoch+1, err))
                fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
                fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
                fconv.write('')
                fconv.close()

                # Save best model
                err = (fpr+fnr)/2.
                if 1-err >= best_acc:
                    best_acc = 1-err
                    obj = {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(obj, os.path.join(
                        args.output_path, 'checkpoint_best.pth')) 

        

        
def main():
    Lite(devices="auto", accelerator="auto").run(1e-4)


if __name__ == "__main__":
    main()
