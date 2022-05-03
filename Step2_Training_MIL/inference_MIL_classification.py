import sys
import os
import numpy as np
import pandas as pd
import argparse
import random
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


def main(args):

    #load model
    model_name = args.model_name
    model,_ = initialize_model(model_name=model_name, num_classes=2, feature_extract=False, use_pretrained=False)

    ckpt_path = os.path.join(args.model_path, model_name)
    version_name = f'MIL_{model_name}_bs{args.batch_size}_w{args.weights}_k{args.k}_output'
    ch = torch.load(os.path.join(ckpt_path, version_name, 'checkpoint_best.pth'))
    
    new_state_dict = OrderedDict()
    for k, v in ch['state_dict'].items():
        name = k[14:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(ch['state_dict'])
    model = model.cuda()
    cudnn.benchmark = True

    #normalization
    DATA_MEANS = [0.485, 0.456, 0.406]
    DATA_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=DATA_MEANS,std=DATA_STD)
    trans = transforms.Compose([transforms.ToTensor(),normalize])

    #load data
    train_dataset = MILdataset(args.lib_dir, args.root_dir, 'Train', transform=trans, subset_rate=args.sample_rate)
    val_dset = MILdataset(args.lib_dir, args.root_dir, 'Val', transform=trans, subset_rate=args.sample_rate)
    test_dset = MILdataset(args.lib_dir, args.root_dir, 'Test', transform=trans, subset_rate=args.sample_rate)
    train_DataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers, pin_memory=False)
    val_DataLoader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=False)
    test_DataLoader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=False)

    train_dataset.setmode(1)
    val_dset.setmode(1)
    test_dset.setmode(1)
        
    # Train
    probs = inference(train_DataLoader, model)
    maxs = group_max(np.array(train_dataset.slideIDX), probs, len(train_dataset.targets))

    output_version_name = f'MIL_Train_{args.model_name}_bs{args.batch_size}_lr{args.learning_rate}_output'
    fp = open(os.path.join(args.model_path,args.model_name,version_name, f'{output_version_name}.csv'), 'w')
    fp.write('slides,tiles,target,prediction,probability\n')
    for slides, tiles, target, prob in zip(train_dataset.slidenames, train_dataset.grid, train_dataset.targets, probs):
        fp.write('{},{},{},{},{}\n'.format(slides, tiles, target, int(prob>=0.5), prob))
    fp.close()
    
    # Val
    probs = inference(val_DataLoader, model)
    maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
    output_version_name = f'MIL_Val_{args.model_name}_bs{args.batch_size}_lr{args.learning_rate}_output'
    fp = open(os.path.join(args.model_path,args.model_name,version_name, f'{output_version_name}.csv'), 'w')
    fp.write('slides,tiles,target,prediction,probability\n')
    for slides, tiles, target, prob in zip(val_dset.slidenames, val_dset.grid, val_dset.targets, probs):
        fp.write('{},{},{},{},{}\n'.format(slides, tiles, target, int(prob>=0.5), prob))
    fp.close()
    
    # Test
    probs = inference(test_DataLoader, model)
    maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))

    output_version_name = f'MIL_Test_{args.model_name}_bs{args.batch_size}_lr{args.learning_rate}_output'
    fp = open(os.path.join(args.model_path,args.model_name,version_name, f'{output_version_name}.csv'), 'w')
    fp.write('slides,tiles,target,prediction,probability\n')
    for slides, tiles, target, prob in zip(test_dset.slidenames, test_dset.grid, test_dset.targets, probs):
        fp.write('{},{},{},{},{}\n'.format(slides, tiles, target, int(prob>=0.5), prob))
    fp.close()    

def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Batch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda()
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

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained, aux_logits=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        # num_ftrs = model_ft.AuxLogits.fc.in_features
        # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--root_dir",
        default='',
        type=Path,
        required=True,
        help="root directory of dataset",
    )
    parser.add_argument(
        "--lib_dir",
        default='',
        type=Path,
        required=True,
        help="root directory of libraryfile",
    )
    parser.add_argument(
        "--output_path",
        default='',
        type=Path,
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--model_path",
        default='',
        type=Path,
        required=True,
        help="model directory",
    )
    
    parser.add_argument(
        "--model_name",
        default='alexnet',
        choices=('resnet18', 'resnet34', 'alexnet', 'vgg',
                 'squeezenet', 'densenet', 'inception','vit'),
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
        default=64,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="batch size",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
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
