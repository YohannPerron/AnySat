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

SOLVER = 'lbfgs'
MAX_ITER = 1000
TOL=3e-3

@torch.no_grad()
def train_LP(dataloader_train, dataloader_val, dataset_name, scale, type, model, device, max_iter_train, max_iter_test, semseg_drop_rate, verbose=False):
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
    t_sample_start = time.time()
    with torch.amp.autocast("cuda",enabled=True, dtype=torch.float16):
        if verbose:
            print("Sampling train set")
        for i, batch in enumerate(dataloader_train):
            if verbose:
                print(f"{i}/{max_iter_train}", end='\r')
            label = batch.pop('label')
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            out = model.forward_release(batch, scale, output=output_type, output_modality='')
            if i == 0 and verbose:
                print("First batch output shape:")
                print(out.shape)
                print(label.shape)
            features_train.append(out.cpu())
            labels_train.append(label.cpu())
            if i>= max_iter_train:
                break
        if verbose:
            print()

        if verbose:
            print("Sampling val set")
        for i, batch in enumerate(dataloader_val):
            if verbose:
                print(f"{i}/{max_iter_test}", end='\r')
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
            if i>= max_iter_test:
                break
        if verbose:
            print()
    t_sample_end = time.time()
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
    elif type == 'semseg':
        # Flatten the features and labels
        features_train = features_train.reshape(-1, features_train.shape[-1])
        features_val = features_val.reshape(-1, features_val.shape[-1])
        labels_train = labels_train.reshape(-1)
        labels_val = labels_val.reshape(-1)

        # Select semseg_drop_rate of the training pixel to drop
        keep_mask = np.random.rand(*labels_train.shape) > semseg_drop_rate
        labels_train = labels_train[keep_mask]
        features_train = features_train[keep_mask]

    #normalize features
    mean = np.mean(features_train, axis=0)
    std = np.std(features_train, axis=0)
    features_train = (features_train - mean) / std
    features_val = (features_val - mean) / std


    t_LP_start = time.time()
    clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, n_jobs=None, solver=SOLVER)
    clf.fit(features_train, labels_train)
    t_LP_end = time.time()

    if type == 'classif':
        pred_val = clf.predict(features_val)
        acc = accuracy_score(labels_val, pred_val)
        f1 = f1_score(labels_val, pred_val, average='macro')

        if verbose:
            print("End of evaluation on", dataset_name)
        return {'accuracy': acc, 'f1_score': f1, 'time_LP': t_LP_end - t_LP_start, 'time_sample': t_sample_end - t_sample_start}

    elif type == 'semseg':
        t_predict_start = time.time()
        pred_val = clf.predict(features_val)
        acc = accuracy_score(labels_val, pred_val)
        jaccard_score_val = jaccard_score(labels_val, pred_val, average='macro')
        t_predict_end = time.time()
        print("time predict", t_predict_end - t_predict_start)

        if verbose:
            print("End of evaluation on", dataset_name)
        return {
            'accuracy': acc,
            'jaccard_score': jaccard_score_val,
            'time_LP': t_LP_end - t_LP_start,
            'time_sample': t_sample_end - t_sample_start,
        }
def eval_model_FT(datasets_config, model, device, verbose=False):
    """
    Evaluate the model on the validation set using linear probing(sklearn).
    """
    # L.seed_everything(42, workers=True)
    metrics= {}
    for dataset in datasets_config:
        if verbose:
            print(f"Evaluating on dataset: {dataset['out_name']}")

        dataloader_train = dataset['train_dataloader']
        dataloader_val = dataset['val_dataloader']
        result = train_LP(dataloader_train,
                          dataloader_val,
                          dataset_name=dataset['name'],
                          scale=dataset['scale'],
                          model=model,
                          type=dataset['task_type'],
                          device=device,
                          max_iter_train=dataset['max_iter_train'],
                          max_iter_test=dataset['max_iter_test'],
                          semseg_drop_rate=dataset['semseg_drop_rate'],
                          verbose=verbose)
        metrics.update({f"{dataset['out_name']}_{k}": v for k, v in result.items()})
        if verbose:
            print(f"Results for {dataset['out_name']}: {result}")
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
        "config": "So2Sat",
        "train_augmentation": Identity(),
        "test_augmentation": Identity(),
        "modalities": ["s2", "s1"],
        "scale": 8,
        "task_type": "classif",
        "num_workers": 0,
        "batch_size": 16,
        "max_iter_train": 100,
        "max_iter_test": 100,
        "semseg_drop_rate": 0,
        'overrides': {'train_dataset.max_samples': 2000,}
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
        "config": "Pastis",
        "train_augmentation": Identity(),
        "test_augmentation": Identity(),
        "modalities": ["spot", "s2", "s1"],
        "scale": 4,
        "task_type": "semseg",
        "num_workers": 2,
        "batch_size": 4,
        "max_iter_train": 50,
        "max_iter_test": 50,
        "semseg_drop_rate": 0.99,
        'overrides': {'classif':False}
    },
    "PastisS1": {
        "config": "Pastis",
        "train_augmentation": Identity(),
        "test_augmentation": Identity(),
        "modalities": ["s1"],
        "scale": 4,
        "task_type": "semseg",
        "num_workers": 2,
        "batch_size": 4,
        "max_iter_train": 50,
        "max_iter_test": 50,
        "semseg_drop_rate": 0.99,
        'overrides': {'classif':False}
    }
}

def get_dataset_config(data_dir, verbose=False):
    list_dataconfig = []
    for dataset_name, dataset_config in EVAL_DATASETS.items():
        dataset_config_path = f"configs/dataset/{dataset_config['config']}.yaml"
        dataconfig = OmegaConf.load(dataset_config_path)
        dataconfig['modalities'] = dataset_config['modalities']
        dataconfig['scale'] = dataset_config['scale']
        dataconfig['data_dir'] = data_dir+"/${dataset.name}/"
        dataconfig['out_name'] = dataset_name

        # Add training parameters to dataconfig
        dataconfig['num_workers'] = dataset_config['num_workers']
        dataconfig['batch_size'] = dataset_config['batch_size']
        dataconfig['max_iter_train'] = dataset_config['max_iter_train']
        dataconfig['max_iter_test'] = dataset_config['max_iter_test']
        dataconfig['semseg_drop_rate'] = dataset_config['semseg_drop_rate']

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
                                global_batch_size=dataconfig['batch_size'],
                                num_nodes=1,
                                num_devices=1,
                                num_workers=dataconfig['num_workers'],
                                verbose=verbose)
        datamodule.setup()
        dataconfig['train_dataloader'] = datamodule.train_dataloader()
        dataconfig['val_dataloader'] = datamodule.val_dataloader()

        list_dataconfig.append(dataconfig)
    return list_dataconfig

if __name__ == "__main__":
    model_config = "logs/JZ/train/runs/20250411-16:31-GEOm-ASt-Original/0_/.hydra/config.yaml"
    model_path = "logs/JZ/train/runs/20250411-16:31-GEOm-ASt-Original/0_/checkpoints/last.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = '/home/yperron/code/AnySat/data/'
    metrics = eval_model_from_path(model_path, model_config, device, overwrite_data_dir=data_dir, verbose=True)
    print(metrics)
