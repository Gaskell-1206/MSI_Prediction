import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Union
from urllib.error import HTTPError
import glob
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.loops import Loop
from skimage import io
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
sys.path.append('/gpfs/scratch/sc9295/digPath/MSI_vs_MSS_Classification/Step1_Training_MSI_MSS')
from train_tile_level_classification import MSI_MSS_Module
from sklearn.metrics import (auc, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)

best_acc = 0

def inference(loader, model):
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
    k = min(k,len(data))
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
            # sys.stdout.write(
            #     'Slides: [{}/{}]\r'.format(i+1, len(lib['subject_id'].unique())))
            # sys.stdout.flush()
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
        df.loc[df['label'] == 'MSI', 'Class'] = 1
        df.loc[df['label'] == 'MSS', 'Class'] = 0
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

    def run(self, args):
        global best_acc
        print(args)

        self.seed_everything(2022)
        model_name = args.model_name
        sample_rate = args.sample_rate
        ckpt_path = os.path.join(args.model_path, f'{args.model_name}_bs{args.batch_size}_lr{args.learning_rate}')
        ckpt_file_path = glob.glob(os.path.join(ckpt_path,'*.ckpt'))[0]
        model = MSI_MSS_Module.load_from_checkpoint(ckpt_file_path)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
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
            args.lib_dir, args.root_dir, 'Train', transform=train_transform, subset_rate=sample_rate)
        val_dataset = MILdataset(
            args.lib_dir, args.root_dir, 'Val', transform=test_transform, subset_rate=sample_rate)
        test_dataset = MILdataset(
            args.lib_dir, args.root_dir, 'Test', transform=test_transform, subset_rate=sample_rate)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, pin_memory=True)
        train_dataloader, val_dataloader, test_dataloader = self.setup_dataloaders(
            train_dataloader, val_dataloader, test_dataloader, move_to_device=True)

        # open output file
        version_name = f'MIL_{model_name}_bs{args.batch_size}_lr{args.learning_rate}_w{args.weights}_k{args.k}_output'
        # logger
        output_path = os.path.join(args.output_path,version_name)
        writer = SummaryWriter(output_path)

        for epoch in tqdm(range(args.nepochs)):
            train_dataset.setmode(1)
            # print("train_set_len:", len(train_dataloader.dataset))
            probs = inference(train_dataloader, model)
            # return the indices of topk tile(s) in each slides
            topk = group_argtopk(
                np.array(train_dataset.slideIDX), probs, args.k)
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

            train_loss = running_loss/len(train_dataloader.dataset)
            print(
                'Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, train_loss))
            writer.add_scalar('train_loss', train_loss, epoch+1)


            # Validation
            if (epoch+1) % args.test_every == 0:
                val_dataset.setmode(1)
                probs = inference(val_dataloader, model)
                maxs = group_max(np.array(val_dataset.slideIDX),
                                 probs, len(val_dataset.targets))
                pred = [1 if x >= 0.5 else 0 for x in probs]
                val_acc, err, fpr, fnr = calc_err(pred, val_dataset.targets)

                print('Validation\tEpoch: [{}/{}]\t ACC: {}\tError: {}\tFPR: {}\tFNR: {}'.format(
                    epoch+1, args.nepochs, val_acc, err, fpr, fnr))

                writer.add_scalar('val_acc', val_acc, epoch+1)
                writer.add_scalar('fpr', fpr, epoch+1)
                writer.add_scalar('fnr', fnr, epoch+1)

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
                    torch.save(obj, os.path.join(output_path, 'checkpoint_best.pth'))

        # test
        ch = torch.load(os.path.join(output_path,'checkpoint_best.pth'))
        # load params
        model.load_state_dict(ch['state_dict'])
        model = model.cuda()
        cudnn.benchmark = True
        train_dataset.setmode(1)
        val_dataset.setmode(1)
        test_dataset.setmode(1)

        # Train
        probs = inference(train_dataloader, model)
        maxs = group_max(np.array(train_dataset.slideIDX), probs, len(train_dataset.targets))
        fp = open(os.path.join(output_path, f'Train_{version_name}.csv'), 'w')
        fp.write('slides,tiles,target,prediction,probability\n')
        for slides, tiles, target, prob in zip(train_dataset.slidenames, train_dataset.grid, train_dataset.targets, probs):
            fp.write('{},{},{},{},{}\n'.format(slides, tiles, target, int(prob>=0.5), prob))
        fp.close()

        # Val
        probs = inference(val_dataloader, model)
        maxs = group_max(np.array(val_dataset.slideIDX), probs, len(val_dataset.targets))
        fp = open(os.path.join(output_path, f'Val_{version_name}.csv'), 'w')
        fp.write('slides,tiles,target,prediction,probability\n')
        for slides, tiles, target, prob in zip(val_dataset.slidenames, val_dataset.grid, val_dataset.targets, probs):
            fp.write('{},{},{},{},{}\n'.format(slides, tiles, target, int(prob>=0.5), prob))
        fp.close()

        # Test
        probs = inference(test_dataloader, model)
        maxs = group_max(np.array(test_dataset.slideIDX), probs, len(test_dataset.targets))
        fp = open(os.path.join(output_path, f'Test_{version_name}.csv'), 'w')
        fp.write('slides,tiles,target,prediction,probability\n')
        for slides, tiles, target, prob in zip(test_dataset.slidenames, test_dataset.grid, test_dataset.targets, probs):
            fp.write('{},{},{},{},{}\n'.format(slides, tiles, target, int(prob>=0.5), prob))
        fp.close()   

        pred = [1 if x >= 0.5 else 0 for x in probs]
        test_acc, err, fnr, fpr = calc_err(pred, test_dataset.targets)
        test_f1_score = f1_score(test_dataset.targets, pred, average='binary')

        try:
            test_auroc_score = roc_auc_score(test_dataset.targets, probs)
            writer.add_scalar("test_auroc_score", test_auroc_score)
        except ValueError:
            writer.add_scalar('test_auroc_score', .0)

        writer.add_scalar('test_f1_score', test_f1_score)    
        writer.add_scalar('test_acc', test_acc)



def main(args):
    Lite(devices="auto", accelerator="auto").run(args)

if __name__ == "__main__":
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
        "--model_path",
        type=Path,
        required=True,
        help="root directory of pretrained models",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--model_name",
        default='alexnet',
        choices=('resnet18', 'resnet34', 'alexnet', 'vgg',
                 'squeezenet', 'densenet', 'inception'),
        type=str,
        help="model use for train",
    )
    parser.add_argument(
        "--sample_rate",
        default=1,
        type=float,
        help="undersample rate",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="learning rate",
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
        default=1,
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
    
    args = parser.parse_args()
    main(args)
