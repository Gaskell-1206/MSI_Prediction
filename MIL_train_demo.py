import os
import urllib.request
from types import SimpleNamespace
from urllib.error import HTTPError
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import tabulate
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
    root_dir = './ex_data/CRC_DX_data_set/Dataset'
    output = './ex_data/CRC_DX_data_set/Output'
    batch_size = 512
    num_workers = 1
    nepochs = 10



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


def main():
    args = Args()

    #Environment
    DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/ConvNets")
    pl.seed_everything(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # data
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.root_dir,"CRC_DX_Train"))
    # DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    # DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    DATA_MEANS = [0.5,0.5,0.5]
    DATA_STD = [0.1,0.1,0.1]
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
    model_name = "resnet18"
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
        max_epochs=100,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"), 
        LearningRateMonitor("epoch")],
        auto_scale_batch_size='binsearch', # [Optional] auto_scale_batch_size
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


if __name__ == "__main__":
    main()
