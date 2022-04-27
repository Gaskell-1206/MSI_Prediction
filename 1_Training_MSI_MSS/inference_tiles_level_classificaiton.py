import argparse
import os
import random
from pathlib import Path
from typing import Callable, Optional, Union
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from matplotlib import image
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from skimage import io
from sklearn.metrics import (auc, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MSI_MSS_dataset(Dataset):
    def __init__(self, libraryfile_dir='', root_dir='', dataset_mode='Train', transform=None, subset_rate=1.0):
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
        self.mode = 1

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
            return img, target
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


class MSI_MSS_DataModule(pl.LightningDataModule):
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
        train_dataset = MSI_MSS_dataset(
            args.lib_dir, args.root_dir, 'Train', transform=self.train_transform, subset_rate=args.sample_rate)
        train_DataLoader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                      num_workers=self.num_workers, pin_memory=True)
        return train_DataLoader

    def val_dataloader(self):
        val_dataset = MSI_MSS_dataset(
            args.lib_dir, args.root_dir, 'Val', transform=self.test_transform, subset_rate=args.sample_rate)
        val_DataLoader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                    num_workers=self.num_workers, pin_memory=True)
        return val_DataLoader

    def test_dataloader(self):
        test_dataset = MSI_MSS_dataset(
            args.lib_dir, args.root_dir, 'Test', transform=self.test_transform, subset_rate=args.sample_rate)
        test_DataLoader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                     num_workers=self.num_workers, pin_memory=True)
        return test_DataLoader


class MSI_MSS_Module(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, loader='Test'):
        super().__init__()
        self.save_hyperparameters()
        self.model, self.input_size = self.initialize_model(
            model_name, self.hparams.model_hparams['num_classes'], feature_extract=False, use_pretrained=True)
        # print(self.model)
        self.loss_module = nn.CrossEntropyLoss()
        self.loader = loader

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name[0] == "Adam":
            optimizer = optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams[0])
        elif self.hparams.optimizer_name[0] == "SGD":
            optimizer = optim.SGD(self.parameters(), **
                                  self.hparams.optimizer_hparams[0])
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name[0]}"'
        # Reduce the learning rate by 0.0001 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 200], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels.long())
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        probs = F.softmax(self.model(imgs), dim=1)
        preds = probs.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return probs.cpu()[:, 1], preds.cpu(), labels.cpu()

    def validation_epoch_end(self, outputs):
        y_true = torch.stack([output[2] for output in outputs]).reshape(-1,)
        y_pred = torch.stack([output[1] for output in outputs]).reshape(-1,)
        y_prob = torch.stack([output[0] for output in outputs]).reshape(-1,)
        val_f1_score = f1_score(y_true, y_pred, average='binary')
        try:
            val_auroc_score = roc_auc_score(y_true, y_prob)
            self.log("val_auroc_score", val_auroc_score,
                     on_step=False, on_epoch=True)
        except ValueError:
            self.log("val_auroc_score", .0, on_step=False, on_epoch=True)
        self.log("val_f1_score", val_f1_score, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        probs = F.softmax(self.model(imgs), dim=1)
        preds = probs.argmax(dim=-1)
        # self.test_acc(preds, labels)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        return probs.cpu()[:, 1], preds.cpu(), labels.cpu()

    def test_epoch_end(self, outputs):
        y_true = torch.stack([output[2] for output in outputs]).reshape(-1,)
        y_pred = torch.stack([output[1] for output in outputs]).reshape(-1,)
        y_prob = torch.stack([output[0] for output in outputs]).reshape(-1,)
        test_f1_score = f1_score(y_true, y_pred, average='binary')
        try:
            test_auroc_score = roc_auc_score(y_true, y_prob)
            self.log("test_auroc_score", test_auroc_score,
                     on_step=False, on_epoch=True)
        except ValueError:
            pass
        self.log("test_f1_score", test_f1_score, on_step=False, on_epoch=True)
        # image_name = os.path.join(
        #     args.output_path, 'Image', f'{self.hparams.model_name}_Test_{self.global_step}_Confusion_Matrix.jpg')
        # self.createConfusionMatrix(y_true, y_pred).savefig(image_name)
        # load_image = np.array(image.imread(image_name)).transpose(2, 0, 1)
        # self.logger.experiment.add_image(
        #     "Test Confusion matrix", load_image, global_step=self.global_step)

        # save y_true, y_pred for use
        version_name = f'Tiles_level_{self.loader}_{self.hparams.model_name}_bs{args.batch_size}_lr{args.learning_rate}_output'
        df_temp = pd.read_csv(f'/gpfs/scratch/sc9295/digPath/CRC_DX_data_set/CRC_DX_Lib/{self.loader}_temporary.csv')
        fp = open(os.path.join(
            f'saved_models/ConvNets/{self.hparams.model_name}', f'{version_name}.csv'), 'w')
        fp.write('slides,tiles,target,prediction,probability\n')
        for slides, tiles, target, pred, prob in zip(list(df_temp['subject_id'][:len(y_true)]),list(df_temp['slice_id'][:len(y_true)]),y_true, y_pred, y_prob):
            fp.write('{},{},{},{},{}\n'.format(slides, tiles, int(target), pred, prob))
        fp.close()

    # def createConfusionMatrix(self, y_true, y_pred):
    #     # constant for classes
    #     classes = ('MSS', 'MSI')

    #     # Build confusion matrix
    #     cf_matrix = confusion_matrix(y_true, y_pred)
    #     df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=[i for i in classes],
    #                          columns=[i for i in classes])
    #     plt.figure(figsize=(16, 16))
    #     s = sns.heatmap(df_cm, annot=True)
    #     s.set(xlabel='y_pred', ylabel='y_true')
    #     return s.get_figure()

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "resnet34":
            """ Resnet34
            """
            model_ft = models.resnet34(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(
                512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(
                pretrained=use_pretrained, aux_logits=False)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            # num_ftrs = model_ft.AuxLogits.fc.in_features
            # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        elif model_name == "vit":
            """ Vit_b_16
            Be careful, expects (224,224) sized images
            """
            model_ft = models.vit_b_16(
                pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size


def main(args):
    # Environment
    DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
    CHECKPOINT_PATH = os.environ.get(
        "PATH_CHECKPOINT", "saved_models/ConvNets")
    pl.seed_everything(2022)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    print(args)

    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # data
    DATA_MEANS = [0.485, 0.456, 0.406]
    DATA_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(196),
        # transforms.RandomInvert(p=0.5),
        # transforms.RandomRotation((-90, 90), fill=(0,)),
        transforms.Normalize(DATA_MEANS, DATA_STD)])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD)])

    if args.model_name == 'inception':
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(196),
            transforms.Resize(299),
            transforms.Normalize(DATA_MEANS, DATA_STD)])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(299),
            transforms.Normalize(DATA_MEANS, DATA_STD)])

    data_module = MSI_MSS_DataModule(
        data_path=args.root_dir,
        train_transform=train_transform,
        val_transform=test_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,)

    # model
    model_name = args.model_name
    model_hparams = {"num_classes": 2, "act_fn_name": "relu"}
    optimizer_name = "Adam",
    optimizer_hparams = {"lr": args.learning_rate, "weight_decay": 1e-4},
    loader = "Train"
    model = MSI_MSS_Module(model_name, model_hparams,
                           optimizer_name, optimizer_hparams,loader)

    # training

    save_name = model_name

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        gpus=1 if str(device) == "cuda:0" else 0,
        min_epochs=10,
        max_epochs=args.nepochs,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc",),
                   LearningRateMonitor("epoch")],
    )
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = False
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # fit
    # trainer.fit(model, datamodule=data_module)

    train_dataset = MSI_MSS_dataset(
        args.lib_dir, args.root_dir, 'Train', transform=train_transform, subset_rate=args.sample_rate)
    train_DataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                                    num_workers=args.num_workers, pin_memory=True)

    val_dataset = MSI_MSS_dataset(
        args.lib_dir, args.root_dir, 'Val', transform=test_transform, subset_rate=args.sample_rate)
    val_DataLoader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                                num_workers=args.num_workers, pin_memory=True)

    test_dataset = MSI_MSS_dataset(
        args.lib_dir, args.root_dir, 'Test', transform=test_transform, subset_rate=args.sample_rate)
    test_DataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                                    num_workers=args.num_workers, pin_memory=True)


    # Test best model on test set
    ckpt_path = f'/gpfs/scratch/sc9295/digPath/1_Training_MSI_MSS_sbatch/saved_models/ConvNets/{args.model_name}/lightning_logs/version_0/checkpoints/'
    ckpt_file_path = glob.glob(f'{ckpt_path}*.ckpt')[0]
    model = MSI_MSS_Module.load_from_checkpoint(ckpt_file_path)
    model.loader = 'Train'
    test_result = trainer.test(model, dataloaders=train_DataLoader, verbose=True)
    model.loader = 'Val'
    test_result = trainer.test(model, dataloaders=val_DataLoader, verbose=True)
    model.loader = 'Test'
    test_result = trainer.test(model, dataloaders=test_DataLoader, verbose=True)


if __name__ == "__main__":
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

    args = parser.parse_args()

    main(args)
