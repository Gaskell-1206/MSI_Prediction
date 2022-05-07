# MSI_vs_MSS_Classification

## Installation

To use this package, install the required python packages (tested with python 3.8 on macOS Monterey):

```bash
pip install -r requirements.txt
```

## Dataset

Training images for MSI detection are obtained from the published paper: Kather, Jakob Nikolas, et al. "Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer." Nature medicine 25.7 (2019): 1054-1056. They are available at https://doi.org/10.5281/zenodo.2530835.

## Step0 Data_Preprocess

Data Preprocess, Statistical Summary, Dataset Split are implemented in jupyter notebook. `Data_Preprocess.ipynb` and `Data_Exploratory.ipynb`

## Step1 Traning CNN for tile_level classification

`train_tile_level_classification.py` is used to train tile_level classification with pretrained models in ImageNet.

Submit jobs in HPC:

```bash
python ./MSI_vs_MSS_Classification/1_Training_MSI_MSS/train_tile_level_classification.py \
    --root_dir ./CRC_DX_data_set/Dataset/ \ 
    --lib_dir ./CRC_DX_data_set/CRC_DX_Lib/ \
    --output_path ./output \
    --model_name resnet18 \
    --sample_rate 1.0 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_workers 4 \
    --nepochs 50 \
```

### class `DataLoader`

+ DataLoader is defined by `MSI_MSS_dataset`, which read information from library files in `CRC_DX_Lib` folder. 
+ If `subset_rate` is specified, subsets can be extracted from the wholde dataset for quick debugging and saved in `<dataset_mode>_temporary.csv`
+ For labels, we used `load_data_and_get_class` to encode `MSI` as 1 and `MSS` as 0.
+ Dataloader will return `image, target`

### class `DataModule`

+ Subclass of `LightningDataModule`, which is used to define train,val,test dataloader for use.
+ We shuffle data for train and drop last for all dataset.
+ Data tranform is applied for data augmentation.

### class `MSI_MSS_Module`

+ Subclass of `LightningModule`, which is used to define train, val, test procedures.
+ `initialize_model` is used to load pre-trained models and modify number of class in fully connected layer. We keep all weights in both CNN and FC updated during training using `set_parameter_requires_grad`.
+ Available pretained models: 'resnet18', 'resnet34', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'.
+ Optimizer is defined in `configure_optimizers`. Options are `Adam` and `SGD`. Schduler is set to reduce learning rate by 0.0001 after 100 and 150 epochs.
+ `Modelcheckpoint` is saved using `val_acc` by max value.

More information to use pytorch-lightning framework can be found in (https://www.pytorchlightning.ai/)

### Inference tile_level classification

`inference_tile_level_classificaiton.py` is used to test models and get ouptus of probabilities of each tile.

Submit jobs in HPC:

```bash
python ./MSI_vs_MSS_Classification/Step1_Training_MSI_MSS/inference_tiles_level_classificaiton.py \
    --root_dir ./CRC_DX_data_set/Dataset/ \
    --lib_dir ./CRC_DX_data_set/CRC_DX_Lib/ \
    --output_path ./output/ \
    --model_name resnet18 \
    --sample_rate 1.0 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_workers 4 \
    --nepochs 50
```

Classes are same to `train_tile_level_classifciaton.py`.

`Trainer` is defined and loaded for testing.

## Step2 Train MIL for tile_level classificaiton

`train_MIL_classification_trained_cnn_models.py` is used to train MIL models using pre-trained CNN models and get ouptus of probabilities of each tile.

Submit jobs in HPC

```bash
python ./MSI_vs_MSS_Classification/Step2_Training_MIL/train_MIL_classification_Lite.py \
    --root_dir ./CRC_DX_data_set/Dataset/ \
    --lib_dir ./CRC_DX_data_set/CRC_DX_Lib \
    --model_path ./saved_models/ConvNets \
    --output_path ./saved_models/ConvNets/ \
    --model_name resnet18 \
    --sample_rate 1.0 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_workers 4 \
    --nepochs 50 \
    --test_every 1 \
    --weights 0.5 \
    --k 1 \
```

class `DataLoder` is defined the same as CNN. But we change the mode as `1` for the whole dataset and `2` for topk tiles in subset to train MIL.

### Training process

+ Training is defined by PyTorch-lightning `LightningLite` module. `Device` and `accelerator` will be automatically defined based on environments.
+ Hyperparameters are `k` for topk probability tiles and `weight` for weight of CrossEntropy.
+ Best model is save by error (`err=(fpr+fnr)/2`) to `checkpoint_best.pth`.

### Inference MIL classification

`inference_MIL_classificaiton.py` is used to test MIL models and get ouptus of probabilities of each tile.

Submit jobs in HPC:

```bash
python ./MSI_vs_MSS_Classification/Step2_Training_MIL/inference_MIL_classification.py \
    --root_dir ./CRC_DX_data_set/Dataset/ \
    --lib_dir ./CRC_DX_data_set/CRC_DX_Lib/ \
    --output_path ./1_Training_MSI_MSS_sbatch/output/ \
    --model_path ./2_Training_MIL_skynet/saved_models/ConvNets \
    --model_name resnet18 \
    --sample_rate 1.0 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --num_workers 4 \
    --nepochs 50 \
    --test_every 5 \
    --weights 0.5 \
    --k 1 \
```

## Step 3 Aggregation

### MajorityVote

`aggregation_MajorityVote.py` can be run in notebook or any cpus.

+ Input is the probabilities of each tile saved in `prediciton.csv`.


### Machine Learning

`aggregation_MachineLearning.py` can be run in notebook or any cpus.

+ Input is the probabilities of each tile saved in `prediciton.csv`.

### RNN

`aggregation_RNN_trained_MIL_models.py` is used to train RNN models aggregating probabilities of each tile for slide_level classification.

Submit jobs in HPC:

```bash
python ./MSI_vs_MSS_Classification/Step3_Aggregation/aggregation_RNN_trained_MIL_models.py \
    --root_dir ./CRC_DX_data_set/Dataset/ \
    --lib_dir ./CRC_DX_data_set/CRC_DX_Lib \
    --model_path ./saved_models/ConvNets \
    --output_path ./saved_models/ConvNets \
    --model_name resnet18 \
    --sample_rate 1.0 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --num_workers 4 \
    --nepochs 50 \
    --test_every 1 \
    --weights 0.5 \
    --k 10 \
```

+ RNN load models from raw or MIL CNN models. Default version is model trained with `lr1e-3, w0.5, k1`.
+ `genPatientIdxDict` is used to get `patined_idx` and `unique_patient_idx`
+ `getkId` is used to get `slice_id`, `tile_id`, `target` for topk tiles.
+ If number of tiles in a slide is less than `k`, this slide will be removed.
+ False negative rate, false positive rate, F1-score, and auroc score are calculated and saved during test.

## Data Visualization

`Data_Visualization` is used to plot confusion martrix and auroc. It can run in jupyter notebook.