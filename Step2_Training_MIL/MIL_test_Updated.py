import sys
import os
import numpy as np
import pandas as pd
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
from skimage import io
from collections import OrderedDict

parser = argparse.ArgumentParser(description='')
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
parser.add_argument('--lib', type=str, default='filelist', help='path to data file')
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--model', type=str, default='', help='path to pretrained model')
parser.add_argument('--batch_size', type=int, default=100, help='how many images to sample per slide (default: 100)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')

class Args:
    root_dir = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/Dataset'
    lib_dir = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/CRC_DX_Lib'
    output_path = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/Output'
    model = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/Output/checkpoint_best.pth'
    batch_size = 100
    # nepochs = 2
    num_workers = 1
    # test_every = 1
    # weights = 0.5
    # k = 1

def main():
    global args
    # args = parser.parse_args()
    args = Args()

    #load model
    model = models.resnet18(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ch = torch.load(args.model)
    new_state_dict = OrderedDict()
    for k, v in ch['state_dict'].items():
        name = k[8:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(ch['state_dict'])
    model = model.cpu()
    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(),normalize])

    #load data
    dset = MILdataset(args.lib_dir, args.root_dir, 'Test', transform=trans, subset_rate=0.01)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)

    dset.setmode(1)
    probs = inference(loader, model)
    maxs = group_max(np.array(dset.slideIDX), probs, len(dset.targets))

    fp = open(os.path.join(args.output_path, 'predictions.csv'), 'w')
    fp.write('slides,tiles,target,prediction,probability\n')
    for slides, tiles, target, prob in zip(dset.slidenames, dset.grid, dset.targets, probs):
        fp.write('{},{},{},{},{}\n'.format(slides, tiles, target, int(prob>=0.5), prob))
    fp.close()

def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Batch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cpu()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

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
    return list(out)

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

if __name__ == '__main__':
    main()
