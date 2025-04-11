import time

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

from data.datamodule import DataModule
from data.transforms.transform import Identity
from models.networks.encoder.utils.pos_embed import (
    get_2d_sincos_pos_embed_with_resolution,
    get_2d_sincos_pos_embed_with_scale)

NUM_WORKERS = 2
BATCH_SIZE = 4
MAX_ITER_TRAIN = 50
MAX_ITER_TEST = 50
SEMSEG_DROP_RATE=0.99
SOLVER = 'lbfgs'

@torch.no_grad()
def train_LP(dataloader_train, dataloader_val, dataset_name, scale, type, model, device, verbose=False):
    """
    Evaluate the model on the validation set using linear probing(sklearn).
    """
    if type == 'semseg':
        output_type= 'dense'
    elif type == 'classif':
        output_type= 'tile'

    model.eval()
    features_train = []
    labels_train = []
    features_val = []
    labels_val = []

    if verbose:
        print("Start evaluating on", dataset_name)
    # Enable torch mixed precision (FP16)
    with torch.amp.autocast("cuda",enabled=True, dtype=torch.float16):
        if verbose:
            print("Sampling train set")
        for i, batch in enumerate(dataloader_train):
            if verbose:
                print(f"{i}/{MAX_ITER_TRAIN}", end='\r')
            label = batch.pop('label')
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            out = model.forward_release(batch, scale, output=output_type, output_modality='')
            if i == 0 and verbose:
                print("First batch output shape:")
                print(out.shape)
                print(label.shape)
            features_train.append(out.cpu())
            labels_train.append(label.cpu())
            if i>= MAX_ITER_TRAIN:
                break
        if verbose:
            print()

        if verbose:
            print("Sampling val set")
        for i, batch in enumerate(dataloader_val):
            if verbose:
                print(f"{i}/{MAX_ITER_TEST}", end='\r')
            label = batch.pop('label')
            # for k, v in batch.items():
            #     try:
            #         print(f"{k}: {v.shape}")
            #     except AttributeError as e:
            #         print(f"{k}: not a tensor")

            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            out = model.forward_release(batch, scale, output=output_type, output_modality='')
            if i == 0 and verbose:
                print("First batch output shape:")
                print(out.shape)
                print(label.shape)
            features_val.append(out.cpu())
            labels_val.append(label.cpu())
            if i>= MAX_ITER_TEST:
                break
        if verbose:
            print()
    if verbose:
        print("End sampling, Training Linear regression")
    features_train = torch.cat(features_train, dim=0).numpy()
    labels_train = torch.cat(labels_train, dim=0).numpy()
    features_val = torch.cat(features_val, dim=0).numpy()
    labels_val = torch.cat(labels_val, dim=0).numpy()

    if type == 'classif':
        while len(labels_train.shape)>1:
            labels_train = np.argmax(labels_train, axis=1)
            labels_val = np.argmax(labels_val, axis=1)

        clf = LogisticRegression(max_iter=1000, tol=3e-3, n_jobs=-1, solver=SOLVER)
        clf.fit(features_train, labels_train)

        pred_val = clf.predict(features_val)
        acc = accuracy_score(labels_val, pred_val)
        f1 = f1_score(labels_val, pred_val, average='macro')

        if verbose:
            print("End of evaluation on", dataset_name)
        return {'accuracy': acc, 'f1_score': f1}

    elif type == 'semseg':
        # Flatten the features and labels
        features_train = features_train.reshape(-1, features_train.shape[-1])
        features_val = features_val.reshape(-1, features_val.shape[-1])
        labels_train = labels_train.reshape(-1)
        labels_val = labels_val.reshape(-1)

        # Select SEMSEG_DROP_RATE of the training pixel to drop
        keep_mask = np.random.rand(*labels_train.shape) > SEMSEG_DROP_RATE
        labels_train = labels_train[keep_mask]
        features_train = features_train[keep_mask]

        t_start = time.time()
        clf = LogisticRegression(max_iter=1000, tol=3e-3, n_jobs=-1, solver=SOLVER)
        clf.fit(features_train, labels_train)
        t_end = time.time()
        if verbose:
            print(f"train time: {t_end - t_start:.2f} seconds")
        t_start = time.time()
        pred_val = clf.predict(features_val)
        t_end = time.time()
        if verbose:
            print(f"test time: {t_end - t_start:.2f} seconds")

        acc = accuracy_score(labels_val, pred_val)
        jaccard_score_val = jaccard_score(labels_val, pred_val, average='macro')

        if verbose:
            print("End of evaluation on", dataset_name)
        return {
            'accuracy': acc,
            'jaccard_score': jaccard_score_val
        }
def eval_model_FT(datasets_config, model, device, verbose=False):
    """
    Evaluate the model on the validation set using linear probing(sklearn).
    """
    # L.seed_everything(42, workers=True)
    metrics= {}
    for dataset in datasets_config:
        if verbose:
            print(f"Evaluating on dataset: {dataset['name']}")

        dataloader_train = dataset['train_dataloader']
        dataloader_val = dataset['val_dataloader']
        result = train_LP(dataloader_train,
                                 dataloader_val,
                                 dataset_name=dataset['name'],
                                 scale=dataset['scale'],
                                 model=model,
                                 type=dataset['task_type'],
                                 device=device,
                                 verbose=verbose)
        metrics.update({f"{dataset['name']}_{k}": v for k, v in result.items()})
        if verbose:
            print(f"Results for {dataset['name']}: {result}")
    return metrics

def eval_model_from_path(model_path, model_config, device = 'cuda', overwrite_data_dir=None, verbose=False):
    """
    Evaluate the model on the validation set using linear probing(sklearn).
    """
    OmegaConf.register_new_resolver("eval", eval)

    config = OmegaConf.load(model_config)
    model = hydra.utils.instantiate(config['model']['network']['encoder'])

    model_param = torch.load(model_path, map_location=device, weights_only=False)['state_dict']
    model_param = {k.replace('target_encoder.', ''): v for k, v in model_param.items() if 'target_encoder' in k}

    model.load_state_dict(model_param, strict=True)
    model.to(device)

    data_dir = config['data']['data_dir'] if overwrite_data_dir is None else overwrite_data_dir
    list_dataconfig = get_dataset_config(data_dir=data_dir, verbose=verbose)
    metrics = eval_model_FT(list_dataconfig, model, device, verbose=verbose)
    return metrics

# Replace multiple lists with a nested dictionary for better organization
EVAL_DATASETS = {
    "So2Sat": {
        "train_augmentation": Identity(),
        "test_augmentation": Identity(),
        "modalities": ["s2", "s1"],
        "scale": 8,
        "task_type": "classif",
        'overrides': {}
    },
    # "BurnScars": {
    #     "train_augmentation": Identity(),
    #     "test_augmentation": Identity(),
    #     "modalities": ["hls"],
    #     "scale": 24,
    #     "task_type": "semseg",
    #     'overrides': {}
    # },
    "Pastis": {
        "train_augmentation": Identity(),
        "test_augmentation": Identity(),
        "modalities": ["spot", "s2", "s1"],
        "scale": 4,
        "task_type": "semseg",
        'overrides': {'classif':False}
    }
}

def get_dataset_config(data_dir, verbose=False):
    list_dataconfig = []
    for dataset_name, dataset_config in EVAL_DATASETS.items():
        dataset_config_path = f"configs/dataset/{dataset_name}.yaml"
        dataconfig = OmegaConf.load(dataset_config_path)
        dataconfig['modalities'] = dataset_config['modalities']
        dataconfig['scale'] = dataset_config['scale']
        dataconfig['data_dir'] = data_dir+"/${dataset.name}/"

        for k, v in dataset_config['overrides'].items():
            ks = k.split('.')
            target=dataconfig
            for k in ks[:-1]:
                target = target[k]
            target[ks[-1]] = v

        # Handle transforms
        if verbose:
            print(f"Configuring dataset: {dataset_name}")
        dataconfig['train_dataset']['transform'] = None
        dataconfig['val_dataset']['transform'] = None
        dataconfig['test_dataset']['transform'] = None
        dataconfig ['dataset'] = dataconfig
        dataconfig = OmegaConf.to_container(dataconfig, resolve=True)
        dataconfig['train_dataset']['transform'] = dataset_config['train_augmentation']
        dataconfig['val_dataset']['transform'] = dataset_config['test_augmentation']
        dataconfig['test_dataset']['transform'] = dataset_config['test_augmentation']
        dataconfig['task_type'] = dataset_config['task_type']

        datamodule = DataModule(train_dataset=hydra.utils.instantiate(dataconfig['train_dataset']),
                                val_dataset=hydra.utils.instantiate(dataconfig['val_dataset']),
                                test_dataset=hydra.utils.instantiate(dataconfig['test_dataset']),
                                global_batch_size=BATCH_SIZE,
                                num_nodes=1,
                                num_devices=1,
                                num_workers=NUM_WORKERS,
                                verbose=verbose)
        datamodule.setup()
        dataconfig['train_dataloader'] = datamodule.train_dataloader()
        dataconfig['val_dataloader'] = datamodule.val_dataloader()

        list_dataconfig.append(dataconfig)
    return list_dataconfig

if __name__ == "__main__":
    model_config = "logs/JZ/train/runs/20250410-11:22-Geom-ASt-loss/0_/.hydra/config.yaml"
    model_path = "logs/JZ/train/runs/20250410-11:22-Geom-ASt-loss/0_/checkpoints/epoch_002.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = '/home/yperron/code/AnySat/data/'
    metrics = eval_model_from_path(model_path, model_config, device, overwrite_data_dir=data_dir, verbose=True)
    print(metrics)
