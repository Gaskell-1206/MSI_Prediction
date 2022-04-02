import os
import pathlib
from skimage import io
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
# import tabulate
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.models as models

# from IPython.display import HTML, display, set_matplotlib_formats
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from typing import Callable, Union, Optional


class Args:
    root_dir = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/CRC_DX_data_set/Dataset'
    output = '/Users/gaskell/Dropbox/Mac/Desktop/CBH/ex_data/output'
    batch_size = 512
    num_workers = 6
    nepochs = 50


class MSS_MSI_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        batch_size: int = 1,
        num_workers: int = 1,
    ):

        super().__init__()
        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        train_path = os.path.join(self.data_path, "CRC_DX_Train")
        train_dataset = torchvision.datasets.ImageFolder(train_path, self.train_transform)
        train_DataLoader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers, pin_memory=True)
        return train_DataLoader

    def val_dataloader(self):
        train_path = os.path.join(self.data_path, "CRC_DX_Val")
        val_dataset = torchvision.datasets.ImageFolder(train_path, self.val_transform)
        val_DataLoader = data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers, pin_memory=True)
        return val_DataLoader

    def test_dataloader(self):
        train_path = os.path.join(self.data_path, "CRC_DX_Test")
        test_dataset = torchvision.datasets.ImageFolder(train_path, self.test_transform)
        test_DataLoader = data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers, pin_memory=True)
        return test_DataLoader


class MSI_MSSModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # self.model = create_model(model_name, model_hparams)
        self.model=models.resnet18(pretrained=True)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name[0] == "Adam":
            optimizer = optim.AdamW(self.parameters(),**self.hparams.optimizer_hparams[0])
        elif self.hparams.optimizer_name[0] == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams[0])
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
        # Reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)

        return loss

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


def create_model(model_name, model_hparams):
    model_dict = {}
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'

def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std
  

def main():
    args = Args()

    #Environment
    DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/ConvNets")
    pl.seed_everything(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # data
    # train_transform = transforms.Compose([transforms.ToTensor()])
    # train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.root_dir,"CRC_DX_Train"),train_transform)
    # train_DataLoader = data.DataLoader(train_dataset, batch_size=512, shuffle=False,num_workers=0)
    # DATA_MEANS, DATA_STD = batch_mean_and_sd(train_DataLoader)
    DATA_MEANS = [0.7263, 0.5129, 0.6925]
    DATA_STD = [0.1564, 0.2031, 0.1460]
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEANS, DATA_STD)
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)])

    data_module = MSS_MSI_DataModule(
        data_path=args.root_dir,
        train_transform=train_transform,
        val_transform=test_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # model
    model_name = "resnet34"
    model_hparams={"num_classes": 2, "act_fn_name": "relu"}
    optimizer_name="Adam",
    optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
    model = MSI_MSSModule(model_name, model_hparams, optimizer_name, optimizer_hparams)

    # training
    
    save_name = model_name

  
    # Create a PyTorch Lightning trainer with the generation callback
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
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # [Optional] lr_finder
    lr_finder = trainer.tuner.lr_find(model,datamodule=data_module)
    fig = lr_finder.plot(suggest=True)
    fig.show()

    model.hparams.learning_rate = lr_finder.suggestion()

    # fit
    trainer.fit(model, datamodule=data_module)

    # Test best model on test set
    model = MSI_MSSModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) 
    test_result = trainer.test(model, datamodule=data_module, verbose=True)
    result = {"test": test_result[0]["test_acc"]}
    print(result)

def build_args():
    parser = ArgumentParser()

    parser = ArgumentParser(description='MIL_train')
    parser.add_argument('--root_path', type=str, default='./', help='path to root data folder')
    parser.add_argument('--output', type=str, default='', help='name of output file')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
    parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
    parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

    # basic args

    # # data config
    # parser = MSS_MSI_DataModule.add_data_specific_args(parser)
    # parser.set_defaults(
    #     data_path=,
    #     train_transform: Callable,
    #     val_transform: Callable,
    #     test_transform: Callable,
    #     batch_size: int = 1,
    #     num_workers: int = 1,
    #     data_path=data_path,  # path to fastMRI data
    #     mask_type="equispaced_fraction",  # VarNet uses equispaced mask
    #     challenge="multicoil",  # only multicoil implemented for VarNet
    #     batch_size=batch_size,  # number of samples per batch
    #     test_path=None,  # path for test split, overwrites data_path
    # )

    # # module config
    # parser = VarNetModule.add_model_specific_args(parser)
    # parser.set_defaults(
    #     num_cascades=8,  # number of unrolled iterations
    #     pools=4,  # number of pooling layers for U-Net
    #     chans=18,  # number of top-level channels for U-Net
    #     sens_pools=4,  # number of pooling layers for sense est. U-Net
    #     sens_chans=8,  # number of top-level channels for sense est. U-Net
    #     lr=0.001,  # Adam learning rate
    #     lr_step_size=40,  # epoch at which to decrease learning rate
    #     lr_gamma=0.1,  # extent to which to decrease learning rate
    #     weight_decay=0.0,  # weight regularization strength
    # )

    # # trainer config
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser.set_defaults(
    #     gpus=num_gpus,  # number of gpus to use
    #     replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
    #     accelerator=backend,  # what distributed version to use
    #     seed=42,  # random seed
    #     deterministic=True,  # makes things slower, but deterministic
    #     default_root_dir=default_root_dir,  # directory for logs and checkpoints
    #     max_epochs=50,  # max number of epochs
    # )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    root_path = Path(args.root_path)
    checkpoint_dir=root_path / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=root_path / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args



if __name__ == "__main__":
    # args = build_args()
    main()
