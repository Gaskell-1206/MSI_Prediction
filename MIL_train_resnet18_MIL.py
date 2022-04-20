import sys
import argparse
import random
from pathlib import Path
import os
import urllib.request
from types import SimpleNamespace
from urllib.error import HTTPError
# from matplotlib.style import library
import pandas as pd
import numpy as np
from skimage import io
import pytorch_lightning as pl
from pytorch_lightning.lite import LightningLite
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

# from IPython.display import HTML, display, set_matplotlib_formats
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torchvision import transforms
from typing import Callable, Union, Optional
import urllib.request
# import ssl

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
    default=512,
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
            print(
                'Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
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
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred == 1, neq).sum())/(real == 0).sum()
    fnr = float(np.logical_and(pred == 0, neq).sum())/(real == 1).sum()
    return err, fpr, fnr


def group_argtopk(groups, data, k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])


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
    def __init__(self, libraryfile_dir='', root_dir='', dataset_mode='Train', transform=None):
        libraryfile_path = os.path.join(
            libraryfile_dir, f'CRC_DX_{dataset_mode}_ALL.csv')
        lib = pd.read_csv(libraryfile_path)
        lib = lib.sort_values(['subject_id'], ignore_index=True)
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
        self.targets = list(lib['label'].values)
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.root_dir = root_dir
        self.dset = f"CRC_DX_{dataset_mode}"

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x], self.grid[x],
                        self.targets[self.slideIDX[x]]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def load_data_and_get_class(self, df):
        encoder = LabelEncoder()
        encoder.fit(["MSI", "MSS"])
        df['Class'] = encoder.transform(df['label'])
        return df

    def __getitem__(self, index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            tile_id = self.grid[index]
            slide_id = self.slides[slideIDX]
            img_name = "blk-{}-{}.png".format(tile_id, slide_id)
            target = self.dataframe.loc[index, 'Class']
            label = 'CRC_DX_MSIMUT' if target == 0 else 'CRC_DX_MSS'
            img_path = os.path.join(self.root_dir, self.dset, label, img_name)
            img = io.imread(img_path).astype('float')
            img = img.transpose(2, 0, 1)
            return img
        elif self.mode == 2:
            slideIDX, tile_id, target = self.t_data[index]
            slide_id = self.slides[slideIDX]
            label = 'CRC_DX_MSIMUT' if target == 0 else 'CRC_DX_MSS'
            img_name = "blk-{}-{}.png".format(tile_id, slide_id)
            img_path = os.path.join(self.root_dir, self.dset, label, img_name)
            img = io.imread(img_path).astype('float')
            img = img.transpose(2, 0, 1)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataframe)


class Args:
    root_dir = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/Dataset'
    lib_dir = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/CRC_DX_Lib'
    output_path = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/Output'
    batch_size = 128
    nepochs = 1
    num_workers = 4
    test_every = 10
    weights = 0.5
    k = 1


class Lite(LightningLite):

    def run(self, learning_rate):
        global args
        # args = parser.parse_args()
        args = Args()
        self.seed_everything(2022)

        model = models.resnet18(pretrained=True)
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

        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(
        ), transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)])
        train_dataset = MILdataset(
            args.lib_dir, args.root_dir, 'Train', transform=train_transform)
        val_dataset = MILdataset(
            args.lib_dir, args.root_dir, 'Val', transform=test_transform)
        test_dataset = MILdataset(
            args.lib_dir, args.root_dir, 'Test', transform=test_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, pin_memory=True)
        train_dataloader, val_dataloader, test_dataloader = self.setup_dataloaders(
            train_dataloader, val_dataloader, test_dataloader, move_to_device=True)

        #open output file
        fconv = open(os.path.join(args.output_path,'convergence.csv'), 'w')
        fconv.write('epoch,metric,value\n')
        fconv.close()

        for epoch in range(args.nepochs):
            train_dataset.setmode(1)
            probs = inference(epoch, train_dataloader, model)
            topk = group_argtopk(
                np.array(train_dataset.slideIDX), probs, args.k)
            train_dataset.maketraindata(topk)
            train_dataset.shuffletraindata()
            train_dataset.setmode(2)

            model.train()
            running_loss = 0.
            for i, (input, target) in enumerate(train_dataloader):
                output = model(input)
                loss = criterion(output, target)
                optimizer.zero_grad()
                self.backward(loss)
                optimizer.step()
                running_loss += loss.item()*input.size(0)
            batch_loss = running_loss/len(train_dataloader.dataset)
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},loss,{}\n'.format(epoch+1,loss))
            fconv.close()
            print(
                'Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, batch_loss))

            #Validation
            if args.val_lib and (epoch+1) % args.test_every == 0:
                val_dataset.setmode(1)
                probs = inference(epoch, val_dataloader, model)
                maxs = group_max(np.array(val_dataset.slideIDX), probs, len(val_dataset.targets))
                pred = [1 if x >= 0.5 else 0 for x in maxs]
                err,fpr,fnr = calc_err(pred, val_dataset.targets)
                print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
                fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
                fconv.write('{},error,{}\n'.format(epoch+1, err))
                fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
                fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
                fconv.close()
                #Save best model
                err = (fpr+fnr)/2.
                if 1-err >= best_acc:
                    best_acc = 1-err
                    obj = {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict()
                    }
                    torch.save(obj, os.path.join(args.output,'checkpoint_best.pth'))


def main():
    lite = Lite(accelerator="cpu", devices=2)
    lite.run(learning_rate=1e-4)


if __name__ == "__main__":
    main()