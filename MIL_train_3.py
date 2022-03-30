import sys
import os
import numpy as np
import pandas as pd
from skimage import io
import argparse
import random
import PIL.Image as Image
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models


class Args:
    root_dir = './ex_data/CRC_DX_data_set/'
    train_lib = './ex_data/CRC_DX_data_set/CRC_DX_Lib/CRC_DX_TRAIN_ALL_Clear.csv'
    val_lib = './ex_data/CRC_DX_data_set/CRC_DX_Lib/CRC_DX_VAL_ALL.csv'
    output = './ex_data/CRC_DX_data_set/Output'
    batch_size = 512
    nepochs = 10
    workers = 1
    test_every = 10
    weights = 0.5
    k = 1


def main():
    global args, best_acc
    # args = parser.parse_args()
    args = Args()

    # define model cnn
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(num_ftrs, 2)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    cudnn.benchmark = True

    # normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load data
    train_dset = MILdataset(args.train_lib, args.root_dir, trans)
    # print(next(iter(train_dset)))
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, args.root_dir, trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    # open output file
    fconv = open(os.path.join(args.output, 'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    # loop through epochs
    for epoch in range(args.nepochs):
        # train
        loss = train(epoch, train_loader, model, criterion, optimizer, device)
        print(
            'Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1, loss))
        fconv.close()

        # validation
        if args.val_lib:
            pred = inference(epoch, val_loader, model, device)
            err, fpr, fnr = calc_err(pred, val_dset.targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(
                epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
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
                    args.output, 'checkpoint_best.pth'))


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', root_dir='', transform=None):

        lib = pd.read_csv(libraryfile)
        lib = lib.sort_values(['subject_id'], ignore_index=True)
        # slides = []

        # for i, name in enumerate(lib['subject_id'].unique()):
        #     sys.stdout.write(
        #         'Slides: [{}/{}]\r'.format(i+1, len(lib['subject_id'].unique())))
        #     sys.stdout.flush()
        #     slides.append(name)

        # grid = []
        # slideIDX = []
        # for i, g in enumerate(lib['subject_id'].unique()):
        #     tiles = lib[lib['subject_id'] == g]['slice_id']
        #     grid.extend(tiles)
        #     slideIDX.extend([i]*len(tiles))

        # print('Number of tiles: {}'.format(len(grid)))
        self.dataframe = load_data_and_get_class(libraryfile)
        self.slidenames = list(lib['subject_id'].values)
        # self.slides = slides
        self.targets = list(lib['label'].values)
        # self.grid = grid
        # self.slideIDX = slideIDX
        self.transform = transform
        self.root_dir = root_dir
        if "TRAIN" in libraryfile:
            self.mode = "CRC_DX_Train"
        elif "VAL" in libraryfile:
            self.mode = "CRC_DX_Val"
        else:
            self.mode = "CRC_DX_Test"

    def __getitem__(self, index):

        # slideIDX = self.slideIDX[index]
        # tile_id = self.grid[index]
        # slide_id = self.slides[slideIDX]
        # img_name = "blk-{}-{}.png".format(tile_id, slide_id)
        img_name = self.dataframe.iloc[index, 3]
        target = self.dataframe.iloc[index, -1]

        label = 'CRC_DX_MSIMUT' if target == 0 else 'CRC_DX_MSS'

        img_path = os.path.join(self.root_dir, self.mode, label, img_name)
        img = io.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return (img, target)

    def __len__(self):
        return len(self.dataframe)


def inference(run, loader, model, device):
    model.eval()
    model = model.to(device)
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print(
                'Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.to(device)
            output = F.softmax(model(input), dim=1)
            # probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return output.cpu().numpy()


def train(run, loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    model = model.to(device)
    for i, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)
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


def load_data_and_get_class(path_to_data):
    data = pd.read_csv(path_to_data)
    encoder = LabelEncoder()
    encoder.fit(["MSI","MSS"])
    data['Class'] = encoder.transform(data['label'])
    return data


if __name__ == '__main__':
    main()
