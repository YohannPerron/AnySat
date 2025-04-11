import copy
import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from lightning import LightningModule

from models.metrics.LP_SSL import eval_model_FT, get_dataset_config
from models.networks.encoder.Transformer import (apply_masks,
                                                 repeat_interleave_batch)
from src import utils
from utils.mask_collator import MaskCollator, MaskCollatorNaive

log = utils.get_pylogger(__name__)

class Module(LightningModule):
    def __init__(self,
                 network,
                 loss,
                 train_metrics,
                 val_metrics,
                 test_metrics,
                 scheduler,
                 optimizer,
                 ema,
                 ipe_scale,
                 len_data,
                 batch_size,
                 num_epochs,
                 scale,
                 shape,
                 ):
        super().__init__()
        self.model = network.instance
        self.target_encoder = copy.deepcopy(self.model.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.mask_collator = MaskCollator(input_size=(shape // scale, shape // scale),
                                            patch_size=1,
                                            enc_mask_scale=(0.85, 1.0),
                                            pred_mask_scale=(0.2, 0.8),
                                            aspect_ratio=(0.75, 1.5),
                                            nenc=1,
                                            npred=4,
                                            min_keep=0,
                                            allow_overlap=False)
        ipe = len_data // batch_size

        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))


    def forward(self, x):
        mask_enc, mask_pred = self.mask_collator(x)
        with torch.no_grad():
            h = self.target_encoder(x)[:, 1:, :]
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)
            # -- create targets (masked regions of h)
            h = apply_masks(h, mask_pred)
            h = repeat_interleave_batch(h, B, repeat=len(mask_enc))
        return self.model(x, mask_enc, mask_pred), h

    def training_step(self, batch, batch_idx):
        pred, target = self.forward(batch)
        batch['target'] = target
        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            loss.pop("logits")
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    def on_after_backward(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.model.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred, target = self.forward(batch)
        batch['target'] = target
        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            self.val_metrics.update(loss["logits"])
            loss.pop("logits")
        else:
            self.val_metrics.update(pred, batch)
        for metric_name, metric_value in loss.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred, target = self.forward(batch)
        batch['target'] = target
        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            self.test_metrics.update(loss["logits"])
            loss.pop("logits")
        else:
            self.test_metrics.update(pred, batch)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

class ModuleMulti(LightningModule):
    def __init__(self,
                 network,
                 target_head,
                 loss,
                 train_metrics,
                 val_metrics,
                 test_metrics,
                 scheduler,
                 optimizer,
                 ema,
                 ipe_scale,
                 ipe,
                 batch_size,
                 num_epochs,
                 scales,
                 shapes,
                 devices,
                 path_to_data,
                 eval_every,
                 ):
        super().__init__()
        self.model = network.instance
        self.target_encoder = copy.deepcopy(self.model.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_head = target_head
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_every = eval_every
        self.path_to_data = path_to_data

        self.mask_collator = {}
        datasets = list(scales.keys())
        for dataset in datasets:
            for scale in scales[dataset]:
                shape = shapes[dataset] // scale
                self.mask_collator['_'.join([dataset, str(scale)])] = MaskCollator(input_size=(shape, shape),
                                                                                patch_size=1,
                                                                                enc_mask_scale=(0.85, 1.0),
                                                                                pred_mask_scale=(0.2, 0.8),
                                                                                aspect_ratio=(0.75, 1.5),
                                                                                nenc=1,
                                                                                npred=4,
                                                                                min_keep=0,
                                                                                allow_overlap=False)

        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

        self.config_FT = get_dataset_config(self.path_to_data)

    def forward(self, x):
        mask_enc, mask_pred = self.mask_collator['_'.join([x['dataset'], str(x['scale'])])](x)
        with torch.no_grad():
            h = self.target_encoder(x)[:, 1:, :]
            h,target_loss = self.target_head(h)  # apply target head
            B = len(h)
            # -- create targets (masked regions of h)
            h = apply_masks(h, mask_pred)
            h = repeat_interleave_batch(h, B, repeat=len(mask_enc))
        return self.model(x, mask_enc, mask_pred), h, target_loss

    def training_step(self, batch, batch_idx):
        pred, target, target_loss = self.forward(batch)
        batch['target'] = target
        loss = self.loss(pred, batch, average=True, target_loss=target_loss)
        if "logits" in loss.keys():
            loss.pop("logits")
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    def on_after_backward(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.model.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred, target, target_loss = self.forward(batch)
        batch['target'] = target
        loss = self.loss(pred, batch, average=True, target_loss=target_loss)
        if "logits" in loss.keys():
            # self.val_metrics.update(loss["logits"], dataset=batch['dataset'])
            loss.pop("logits")

        self.val_metrics.update(pred, batch)
        for metric_name, metric_value in loss.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()

        if self.current_epoch % self.eval_every == 0:
            log.info("Evaluating FT metrics")
            #spread dataset_config (List) on all the ranks
            rank_dataset_config = [config for i, config in enumerate(self.config_FT) if i%self.trainer.world_size == self.global_rank]
            FT_metrics = eval_model_FT(rank_dataset_config, self.model.encoder, device=self.device, verbose=False)

            #gather FT_metrics on all the ranks
            FT_metrics_all = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(object_list=FT_metrics_all, obj=FT_metrics)
            # FT_metrics_all = self.all_gather(FT_metrics)
            log.info(f"FT_metrics_all: {FT_metrics_all}")
            for FT_metrics in FT_metrics_all:
                metrics.update(FT_metrics)

        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred, target, target_loss = self.forward(batch)
        batch['target'] = target
        loss = self.loss(pred, batch, average=True, target_loss=target_loss)
        if "logits" in loss.keys():
            self.test_metrics.update(loss["logits"], dataset=batch['dataset'])
            loss.pop("logits")
        else:
            self.test_metrics.update(pred, batch)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
    def on_train_epoch_start(self):
        log.info(f"Epoch: {self.current_epoch}")
        return super().on_train_start()

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

class ModuleMultiNaive(LightningModule):
    def __init__(self,
                 network,
                 loss,
                 train_metrics,
                 val_metrics,
                 test_metrics,
                 scheduler,
                 optimizer,
                 ema,
                 ipe_scale,
                 ipe,
                 batch_size,
                 num_epochs,
                 scales,
                 shapes,
                 devices
                 ):
        super().__init__()
        self.model = network.instance
        self.target_encoder = copy.deepcopy(self.model.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.mask_collator = {}
        datasets = list(scales.keys())
        for dataset in datasets:
            for scale in scales[dataset]:
                shape = shapes[dataset] // scale
                self.mask_collator['_'.join([dataset, str(scale)])] = MaskCollatorNaive(input_size=(shape, shape),
                                                                                patch_size=1)

        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    def forward(self, x):
        mask_enc, mask_pred = self.mask_collator['_'.join([x['dataset'], str(x['scale'])])](x)
        with torch.no_grad():
            h = self.target_encoder(x)[:, 1:, :]
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)
            # -- create targets (masked regions of h)
            h = apply_masks(h, mask_pred)
            h = repeat_interleave_batch(h, B, repeat=len(mask_enc))
        return self.model(x, mask_enc, mask_pred), h

    def training_step(self, batch, batch_idx):
        pred, target = self.forward(batch)
        batch['target'] = target
        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            loss.pop("logits")
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    def on_after_backward(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.model.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred, target = self.forward(batch)
        batch['target'] = target
        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            self.val_metrics.update(loss["logits"], dataset=batch['dataset'])
            loss.pop("logits")
        else:
            self.val_metrics.update(pred, batch)
        for metric_name, metric_value in loss.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred, target = self.forward(batch)
        batch['target'] = target
        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            self.test_metrics.update(loss["logits"], dataset=batch['dataset'])
            loss.pop("logits")
        else:
            self.test_metrics.update(pred, batch)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

