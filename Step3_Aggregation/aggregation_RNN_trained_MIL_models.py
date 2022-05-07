# Run RNN aggregation use MIL models
# Reference: 1.Campanella, G. et al. Clinical-grade computational pathology using weakly supervised
#            deep learning on whole slide images. Nat Med 25, 1301–1309 (2019).
#            doi:10.1038/s41591-019-0508-1. Available from http://www.nature.com/articles/s41591-019-0508-1
#            The source codes of the referenced paper available at https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019
# This code was modified by Shengjia Chen for our work.

import sys
import os
from importlib_metadata import version
import numpy as np
import pandas as pd
import argparse
import random
#import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
# from skimage import io
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (auc, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)


def genPatientIdxDict(patient_ID):
    ''' generate patient->patches index dict
    '''
    patient_idx_dict = {}
    unique_patient, unique_patient_idx = np.unique(patient_ID, return_index=True)
    for p in unique_patient:
        patient_idx_dict[p] = np.where(patient_ID == p)[0]

    return patient_idx_dict, unique_patient_idx

def genkID(df,k):
  patient_idx_dict, unique_patient_idx = genPatientIdxDict(df['slides'])
  slide_id=[None]*len(unique_patient_idx)
  target=[None]*len(unique_patient_idx)
  tile_id=[None]*len(unique_patient_idx)
  for i in range(len(unique_patient_idx)):
        idx = patient_idx_dict[df['slides'][unique_patient_idx[i]]]
        probs = df['probability'][idx]
        idx = idx[np.argsort(probs)[-k:]]
        slide_id[i] = df['slides'][idx[0]]
        target[i] = df['target'][idx[0]]
        id = []
        for kk in range(k):
          id.append(df['tiles'][idx[kk]])
        tile_id[i] = id
  return slide_id, tile_id, target

best_acc = 0
def main(args):
    global best_acc
    model_name = args.model_name
    version_name = f'MIL_{model_name}_bs{args.batch_size}_lr0.001_w0.5_k1_output'
    file_path = Path(os.path.join(args.model_path,version_name))
    
    # load model
    ckpt_path = file_path / 'checkpoint_best.pth'
    ch = torch.load(ckpt_path, map_location=torch.device('cuda'))
    model,_ = initialize_model(model_name=model_name, num_classes=2, feature_extract=True, use_pretrained=False)
    new_state_dict = OrderedDict()
    for k, v in ch['state_dict'].items():
        name = k[14:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    
        
    # prepare k tile_ids
    df_train = pd.read_csv(file_path / f'Train_{version_name}.csv')
    key_list = list(df_train['slides'].value_counts()[df_train['slides'].value_counts() < args.k].keys())
    df_train = df_train.drop(df_train[df_train['slides'].isin(key_list)].index).reset_index()
    slide_id, tile_id, target = genkID(df_train,args.k)
    lib_train = pd.DataFrame({'slides':slide_id, 'targets':target, 'grid':tile_id})
    
    df_val = pd.read_csv(file_path / f'Val_{version_name}.csv')
    key_list = list(df_val['slides'].value_counts()[df_val['slides'].value_counts() < args.k].keys())
    df_val = df_val.drop(df_val[df_val['slides'].isin(key_list)].index).reset_index()
    slide_id, tile_id, target = genkID(df_val,args.k)
    lib_val = pd.DataFrame({'slides':slide_id, 'targets':target, 'grid':tile_id})
    
    df_test = pd.read_csv(file_path / f'Test_{version_name}.csv')
    key_list = list(df_test['slides'].value_counts()[df_test['slides'].value_counts() < args.k].keys())
    df_test = df_test.drop(df_test[df_test['slides'].isin(key_list)].index).reset_index()
    slide_id, tile_id, target = genkID(df_test,args.k)
    lib_test = pd.DataFrame({'slides':slide_id, 'targets':target, 'grid':tile_id})
    
    #load libraries
    DATA_MEANS = [0.485, 0.456, 0.406]
    DATA_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=DATA_MEANS,std=DATA_STD)
    trans = transforms.Compose([transforms.ToTensor(),normalize])
    train_dset = rnndata(lib_train, args.k, args.root_dir, 'Train', transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)
    val_dset = rnndata(lib_val, args.k, args.root_dir, 'Val', transform=trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)
    test_dset = rnndata(lib_test, args.k, args.root_dir, 'Test', transform=trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)
    

    #make model
    embedder = ResNetEncoder(model)
    for param in embedder.parameters():
        param.requires_grad = False
    embedder = embedder.cuda()
    embedder.eval()

    rnn = rnn_single(256)
    rnn = rnn.cuda()
    
    #optimization
    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.SGD(rnn.parameters(), args.learning_rate, momentum=0.9, dampening=0, weight_decay=1e-5, nesterov=True)
    cudnn.benchmark = True
    
    # open output file
    version_name = f'RNN_{model_name}_bs{args.batch_size}_lr{args.learning_rate}_w{args.weights}_k{args.k}_output_newk1'
    # logger
    output_path = os.path.join(args.output_path,version_name)
    writer = SummaryWriter(output_path)

    #
    for epoch in range(args.nepochs):

        train_loss, train_fpr, train_fnr = train_single(epoch, embedder, rnn, train_loader, criterion, optimizer)
        val_acc, val_loss, val_fpr, val_fnr = val_single(epoch, embedder, rnn, val_loader, criterion)
        writer.add_scalar('train_loss', train_loss, epoch+1)
        writer.add_scalar('train_fpr', train_fpr, epoch+1)
        writer.add_scalar('train_fnr', train_fnr, epoch+1)
        writer.add_scalar('val_acc', val_acc, epoch+1)
        writer.add_scalar('val_loss', val_loss, epoch+1)
        writer.add_scalar('val_fpr', val_fpr, epoch+1)
        writer.add_scalar('val_fnr', val_fnr, epoch+1)

        val_err = (val_fpr + val_fnr)/2
        if 1-val_err >= best_acc:
            best_acc = 1-val_err
            obj = {
                'epoch': epoch+1,
                'state_dict': rnn.state_dict()
            }
            torch.save(obj, os.path.join(output_path,'rnn_checkpoint_best.pth'))
            
    # Test
    probs = test_single(embedder, rnn, test_loader)

    fp = open(os.path.join(output_path, f'{version_name}.csv'), 'w')
    fp.write('file,target,prediction,probability\n')
    for name, target, prob in zip(test_dset.slidenames, test_dset.targets, probs):
        fp.write('{},{},{},{}\n'.format(name, target, int(prob>=0.5), prob))
    fp.close()
    
    pred = [1 if x >= 0.5 else 0 for x in probs]
    test_acc, err, fnr, fpr = calc_err(pred, test_dset.targets)
    test_f1_score = f1_score(test_dset.targets, pred, average='binary')

    try:
        test_auroc_score = roc_auc_score(test_dset.targets, probs)
        writer.add_scalar("test_auroc_score", test_auroc_score)
    except ValueError:
        writer.add_scalar('test_auroc_score', .0)

    writer.add_scalar('test_f1_score', test_f1_score)    
    writer.add_scalar('test_acc', test_acc)

    print('Test - \tAcc: {} \tfnr: {}\tfpr: {}\tF1: {}\tAUROC: {}'.format(test_acc, fnr, fpr, test_f1_score, test_acc))

class rnndata(Dataset):

    def __init__(self, lib, k, root_dir, dataset_mode='Train', transform=None):

        self.s = k
        self.transform = transform
        self.slidenames = lib['slides']
        self.targets = lib['targets']
        self.grid = lib['grid']
        #self.level = lib['level']
        #self.mult = lib['mult']
        self.mult = 1
        self.size = int(224*self.mult)
        #self.shuffle = shuffle
        self.root_dir = root_dir
        self.dset = f"CRC_DX_{dataset_mode}"

        slides = []
        for i, name in enumerate(lib['slides']):
            # sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            # sys.stdout.flush()
            slides.append(name)
        print('')
        self.slides = slides

    def __getitem__(self,index):

        slide = self.slidenames[index]
        grid = self.grid[index]
        #if self.shuffle:
        #    grid = random.sample(grid,len(grid))
        out = []
        label = 'CRC_DX_MSIMUT' if self.targets[index] == 1 else 'CRC_DX_MSS'
        s = min(self.s, len(grid))
        for kk in range(s):
            img_name = "blk-{}-{}.png".format(grid[kk], slide)
            img_path = os.path.join(self.root_dir, self.dset, label, img_name)
            img = Image.open(img_path)

            if self.transform is not None:
                img = self.transform(img)
                
            out.append(img)
            
        out = np.stack(out)
        
        return out, self.targets[index]

    def __len__(self):
        
        return len(self.targets)

class ResNetEncoder(nn.Module):

    def __init__(self, model):
        super(ResNetEncoder, self).__init__()
        
        # model = MSI_MSS_Module.load_from_checkpoint(path)
        self.features = nn.Sequential(*list(model.children())[:-1])
        if args.model_name == 'densenet':
            self.fc = model.classifier
        elif args.model_name == 'resnet18':
            self.fc = model.fc
        else:
            self.fc = nn.Linear(512,2)

    def forward(self,x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0),-1)
        return self.fc(x), x

class rnn_single(nn.Module):

    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims

        self.fc1 = nn.Linear(1024, ndims)
        self.fc2 = nn.Linear(ndims, ndims)

        self.fc3 = nn.Linear(ndims, 2)

        self.activation = nn.ReLU()

    def forward(self, input, state):
        input = self.fc1(input)
        state = self.fc2(state)
        state = self.activation(state+input)
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)

def train_single(epoch, embedder, rnn, loader, criterion, optimizer):
    rnn.train()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    for i,(inputs,target) in enumerate(loader):
        print('Training - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1, args.nepochs, i+1, len(loader)))

        batch_size = inputs.size(0)
        rnn.zero_grad()

        state = rnn.init_hidden(batch_size).cuda()
        for s in range(args.k):
            input = inputs[:,s,:,:,:].cuda()
            _, input = embedder(input)
            output, state = rnn(input, state)

        target = target.cuda()
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*target.size(0)
        acc, fps, fns = errors(output.detach(), target.cpu())
        running_fps += fps
        running_fns += fns

    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    print('Training - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def val_single(epoch, embedder, rnn, loader, criterion):
    rnn.eval()
    running_loss = 0.
    running_acc = 0.
    running_fps = 0.
    running_fns = 0.

    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            print('Validating - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1,args.nepochs,i+1,len(loader)))
            
            batch_size = inputs.size(0)
            
            state = rnn.init_hidden(batch_size).cuda()
            for s in range(args.k):
                input = inputs[:,s,:,:,:].cuda()
                _, input = embedder(input)
                output, state = rnn(input, state)
            
            target = target.cuda()
            loss = criterion(output,target.long())
            
            running_loss += loss.item()*target.size(0)
            acc, fps, fns = errors(output.detach(), target.cpu())
            running_acc += acc
            running_fps += fps
            running_fns += fns
            
    running_loss = running_loss/len(loader.dataset)
    running_acc = running_acc / len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    print('Validating - Epoch: [{}/{}]\tAcc: {}\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, running_acc, running_loss, running_fps, running_fns))
    return running_acc, running_loss, running_fps, running_fns

def test_single(embedder, rnn, loader):
    rnn.eval()

    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, (inputs, target) in enumerate(loader):
            print('Testing - Batch: [{}/{}]'.format(i+1,len(loader)))
            
            batch_size = inputs[0].size(0)
            
            state = rnn.init_hidden(batch_size).cuda()
            for s in range(len(inputs)):
                input = inputs[s].cuda()
                _, input = embedder(input)
                output, state = rnn(input, state)

            output = F.softmax(output, dim=1)
            probs[i*args.batch_size:i*args.batch_size+batch_size] = output.detach()[:,1].clone()

    return probs.cpu().numpy()

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
            
def errors(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    real = target.numpy()
    pos = np.equal(pred, real)
    neq = pred!=real
    acc = pos.sum()
    fps = float(np.logical_and(pred==1,neq).sum())
    fns = float(np.logical_and(pred==0,neq).sum())
    return acc,fps,fns

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