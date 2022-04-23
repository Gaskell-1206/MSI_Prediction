import sys
import argparse
import random
from pathlib import Path
import os
from types import SimpleNamespace
from urllib.error import HTTPError
import pandas as pd
import numpy as np
from skimage import io
import pytorch_lightning as pl
from pytorch_lightning.loops import Loop
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import Callable, Union, Optional
from pytorch_lightning.loops.fit_loop import FitLoop

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
    num_workers = 2
    test_every = 1
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
        self.targets = list(lib['Class'].values)
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
            img = io.imread(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, tile_id, target = self.t_data[index]
            slide_id = self.slides[slideIDX]
            label = 'CRC_DX_MSIMUT' if target == 0 else 'CRC_DX_MSS'
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

class MIL_Module(pl.LightningModule):
    def __init__(
        self, model_name, model_hparams, optimizer_name, optimizer_hparams,
        train_dataset, val_dataset, args):

        super().__init__()
        self.save_hyperparameters()
        # self.model = create_model(model_name, model_hparams)
        self.model=models.resnet18(pretrained=True)
        self.loss_module =  nn.CrossEntropyLoss(torch.Tensor([1-args.weights, args.weights]))
        self.train_dataset = train_dataset
        self.args = args

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name[0] == "Adam":
            optimizer = torch.optim.AdamW(self.parameters(),**self.hparams.optimizer_hparams[0])
        elif self.hparams.optimizer_name[0] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams[0])
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
        # Reduce the learning rate by 0.1 after 50 and 100 epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100], gamma=0.1)
        return [optimizer], [scheduler]
    
    def on_train_epoch_start(self):
        self.train_dataset.setmode(1)
        probs = self.inference()
        # return the indices of topk tile(s) in each slides
        topk = group_argtopk(np.array(self.train_dataset.slideIDX), probs, self.args.k)
        self.train_dataset.maketraindata(topk)
        self.train_dataset.shuffletraindata()
        self.train_dataset.setmode(2)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        probs = torch.FloatTensor(len())
        return super().on_validation_epoch_start()
        
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)

    def inference(self,loader, model):
        model.eval()
        probs = torch.FloatTensor(len(loader.dataset))
        with torch.no_grad():
            for i, input in enumerate(loader):
                output = F.softmax(model(input), dim=1)
                probs[i*self.args.batch_size:i*self.args.batch_size +
                    input.size(0)] = output.detach()[:, 1].clone()
        return probs.cpu().numpy()

class TrainingEpochLoop(Loop):
    
    def __init__(self, model, optimizer, dataloader):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.batch_idx = 0
    
    @property
    def done(self):
        return self.batch_idx >= len(self.dataloader)
    
    def reset(self) -> None:
        self.dataloader_iter = iter(self.dataloder)

    def advance(self, *args, **kwargs) -> None:
        batch = next(self.dataloader_iter)
        self.optimizer.zero_grad()
        loss = self.model.training_step(
            batch, self.batch_idx
        )
        loss.backward()
        self.optimizer.step()

class MultipleInstanceLearningLoop(Loop):
    max_epochs: int
    
    def __init__(self, num_epochs, model, optimizer, train_dataloader, val_dataloader):
        super().__init__()
        self.args = parser.parse_args()
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.model = model
        self.train_dataloder = train_dataloader
        self.val_dataloader = val_dataloader
        self.epoch_loop_train = TrainingEpochLoop(
            model, optimizer, train_dataloader
        )
        self.fit_loop: Optional[FitLoop] = None
    
    @property
    def done(self):
        return self.progress.current.completed >= self.max_epochs

    def connect(self, fit_loop: FitLoop): # read setting from trainer.fit_loop
        self.fit_loop = fit_loop
        self.max_epochs = self.fit_loop.max_epochs
    
    def reset(self) -> None:
        pass

    def advance(self, train_dataset, val_dataset) -> None:
        # Inference and find topk
        train_dataset.setmode(1)
        probs = self.model.inference(self.train_dataloder, self.model)
        topk = group_argtopk(np.array(train_dataset.slideIDX), probs, self.args.k)
        train_dataset.maketraindata(topk)
        train_dataset.shuffletraindata()
        train_dataset.setmode(2)

        # Run Train
        self.epoch_loop_train.run()
        self.current_epoch += 1

        # Validation
        if (self.num_epochs+1) % self.args.test_every == 0:
            val_dataset.setmode(1)
            probs = self.model.inference(self.num_epochs, self.val_dataloader, self.model)
            maxs = group_max(np.array(val_dataset.slideIDX),
                                probs, len(val_dataset.targets))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            auroc_score = roc_auc_score(val_dataset.targets, probs)
            acc, err, fpr, fnr = calc_err(pred, val_dataset.targets)
            self.log("val_acc", acc)
            self.log("val_err", err)
            self.log("val_fpr", fpr)
            self.log("val_fnr", fnr)
            self.log("auroc_score", auroc_score)

        
def main(args):
    args = Args()
    # ssl._create_default_https_context = ssl._create_unverified_context
    #Environment
    DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/ConvNets")
    pl.seed_everything(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # data
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
        args.lib_dir, args.root_dir, 'Train', transform=train_transform, subset_rate=0.01)
    val_dataset = MILdataset(
        args.lib_dir, args.root_dir, 'Val', transform=test_transform, subset_rate=0.01)
    test_dataset = MILdataset(
        args.lib_dir, args.root_dir, 'Test', transform=test_transform, subset_rate=0.01)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    model_name = "resnet18"
    model_hparams={"num_classes": 2, "act_fn_name": "relu"}
    optimizer_name="Adam",
    optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
    model = MIL_Module(model_name, model_hparams, optimizer_name, optimizer_hparams)

    # training
    save_name = model_name

    # Create a PyTorch Lightning trainer with the generation callback
    # early_stop_callback = EarlyStopping(
    #     monitor="val_acc", min_delta=0.00, patience=10,verbose=False, mode="max")

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        gpus=1 if str(device) == "cuda:0" else 0,
        min_epochs=10,
        max_epochs=args.nepochs,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"), 
        LearningRateMonitor("epoch")],
        # auto_scale_batch_size='binsearch', # [Optional] auto_scale_batch_size
        auto_lr_find=True
    )

    # # customize loop
    # trainer.fit_loop = MILLoop(args.nepochs,model,)

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # [Optional] lr_finder
    lr_finder = trainer.tuner.lr_find(model,train_dataloader=train_dataloader,val_dataloaders=val_dataloader)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()

    model.hparams.learning_rate = lr_finder.suggestion()

    # fit
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    # Test best model on test set
    model = MSI_MSSModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) 
    test_result = trainer.test(model, datamodule=data_module, verbose=True)
    result = {"test": test_result[0]["test_acc"]}
    print(result)


if __name__ == "__main__":
    main()
