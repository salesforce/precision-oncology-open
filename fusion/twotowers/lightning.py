"""
TODO: maybe create a SupervisedLightningModule class to inherit from
"""

import os
import torch
import pandas as pd
import numpy as np

from . import builder 
from .constants import *
from .utils import get_auroc, get_auprc
from pytorch_lightning.core import LightningModule


class EMRLightningModule(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg, dm):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.dm = dm
        self.model = builder.build_model(cfg, self.dm)
        self.loss_fn = builder.build_loss(cfg)

        #self.y_list = []
        #self.prob_list = []
        
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def training_epoch_end(self, training_step_outputs): 
        return self.shared_epoch_end(training_step_outputs, 'train')

    def validation_epoch_end(self, validation_step_outputs): 
        return self.shared_epoch_end(validation_step_outputs, 'val')

    def test_epoch_end(self, test_step_outputs): 
        return self.shared_epoch_end(test_step_outputs, 'test')

    def shared_step(self, batch, split):
        """Similar to traning step"""

        x, y, idx = batch
        logit, M_loss = self.model(x)   # M_loss: overall sparcity loss

        logit = logit.reshape(-1)
        y = y.float()
        loss = self.loss_fn(logit, y) 

        # sparcity loss coefficient (bigger the coefficient, sparser your model will be)
        if self.cfg.train.lambda_sparse is not None:
            loss -= self.cfg.train.lambda_sparse * M_loss

        if split != 'test':
            self.log(f'{split}_loss', loss, on_epoch=True, on_step=False, logger=True, prog_bar=True)

        #self.y_list += y.detach().cpu().tolist()
        #self.prob_list += prob.detach().cpu().tolist()
        #return_dict = {'loss': loss}

        return_dict = {
            'y': y,
            'idx': idx,
            'loss': loss,
            'logit': logit}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):
        
        #y = self.y_list
        #prob = self.prob_list
        #self.y_list = []
        #self.prob_list = []
        logit = torch.cat([x['logit'] for x in step_outputs])
        y = torch.cat([x['y'] for x in step_outputs])
        #idx = [v for v in [x['idx'] for x in step_outputs]]
        # flattern idx list
        idx = [v for w in [x['idx'] for x in step_outputs] for v in w]
        prob = torch.sigmoid(logit)

        # get metrics
        y = y.detach().cpu().tolist()
        prob = prob.detach().cpu().tolist()
        auroc = get_auroc(y, prob)
        auprc = get_auprc(y, prob)
        if split != 'test':
            self.log(f'{split}/auroc', auroc, on_epoch=True, logger=True, prog_bar=True)
            self.log(f'{split}/auprc', auprc, on_epoch=True, logger=True, prog_bar=True) 
        else: 
            print(f'AUROC: {auroc}') 
            print(f'AUPRC: {auprc}')

        if split == 'test': 
            results = {'y': y, 'prob': prob, 'id':idx} 
            results_df = pd.DataFrame.from_dict(results)
            csv_path = os.path.join(self.cfg.save_dir, 'results.csv')
            results_df.to_csv(csv_path, index=False)
            print(f'RESULTS_PATH {csv_path}')


class EMRPretrainLightningModule(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg, dm):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.dm = dm
        self.model = builder.build_model(cfg, self.dm)
        self.loss_fn = builder.build_loss(cfg)
        
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def training_epoch_end(self, training_step_outputs): 
        return self.shared_epoch_end(training_step_outputs, 'train')

    def validation_epoch_end(self, validation_step_outputs): 
        return self.shared_epoch_end(validation_step_outputs, 'val')

    def test_epoch_end(self, test_step_outputs): 
        return self.shared_epoch_end(test_step_outputs, 'test')

    def shared_step(self, batch, split):
        """Similar to traning step"""
        x, _, _ = batch
        output, embedded_x, obf_vars = self.model(x)
        loss = self.loss_fn(output, embedded_x, obf_vars)
        return {'loss': loss}

    def shared_epoch_end(self, step_outputs, split):
        
        loss = np.mean([x['loss'].detach().cpu().item() for x in step_outputs])
        if np.isnan(loss):
            loss = 0
        self.log(f'{split}/pretrain_loss', loss, on_epoch=True, logger=True, prog_bar=True) 


class ImageLightningModule(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg, dm):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.dm = dm
        self.model = builder.build_model(cfg, self.dm)
        self.loss_fn = builder.build_loss(cfg)
        
        self.y_list = []
        self.prob_list = []

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def training_epoch_end(self, training_step_outputs): 
        return self.shared_epoch_end(training_step_outputs, 'train')

    def validation_epoch_end(self, validation_step_outputs): 
        return self.shared_epoch_end(validation_step_outputs, 'val')

    def test_epoch_end(self, test_step_outputs): 
        return self.shared_epoch_end(test_step_outputs, 'test')

    def shared_step(self, batch, split):
        """Similar to traning step"""

        x, y, idx = batch
        logit = self.model(x)   # M_loss: overall sparcity loss

        logit = logit.reshape(-1)
        prob = torch.sigmoid(logit)
        y = y.float()
        loss = self.loss_fn(logit, y)

        if split != 'test':
            self.log(f'{split}_loss', loss, on_epoch=True, on_step=False, logger=True, prog_bar=True)

        #self.y_list += y.detach().cpu().tolist()
        #self.prob_list += prob.detach().cpu().tolist()
        #return_dict = {'loss': loss}

        return_dict = {
            'y': y,
            'idx': idx, 
            'loss': loss,
            'logit': logit}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):
        
        #y = self.y_list
        #prob = self.prob_list
        #self.y_list = []
        #self.prob_list = []

        logit = torch.cat([x['logit'] for x in step_outputs])
        y = torch.cat([x['y'] for x in step_outputs])
        #idx = list([x['idx'] for x in step_outputs][0])
        idx = [v for w in [x['idx'] for x in step_outputs] for v in w]
        prob = torch.sigmoid(logit)

        # get metrics
        y= y.detach().cpu().tolist()
        prob = prob.detach().cpu().tolist()
        auroc = get_auroc(y, prob)
        auprc = get_auprc(y, prob)
        if split != 'test':
            self.log(f'{split}/auroc', auroc, on_epoch=True, logger=True, prog_bar=True)
            self.log(f'{split}/auprc', auprc, on_epoch=True, logger=True, prog_bar=True) 
        else: 
            print(f'AUROC: {auroc}')
            print(f'AUPRC: {auprc}')

        if split == 'test': 
            results = {'y': y, 'prob': prob, 'id': idx}
            results_df = pd.DataFrame.from_dict(results)
            csv_path = os.path.join(self.cfg.save_dir, 'results.csv')
            results_df.to_csv(csv_path, index=False)
            print(f'RESULTS_PATH {csv_path}')


class FusionLightningModule(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg, dm):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.dm = dm
        self.model = builder.build_model(cfg, self.dm)
        self.loss_fn = builder.build_loss(cfg)

        self.y_list = []
        self.prob_list = []
        
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def training_epoch_end(self, training_step_outputs): 
        return self.shared_epoch_end(training_step_outputs, 'train')

    def validation_epoch_end(self, validation_step_outputs): 
        return self.shared_epoch_end(validation_step_outputs, 'val')

    def test_epoch_end(self, test_step_outputs): 
        return self.shared_epoch_end(test_step_outputs, 'test')

    def shared_step(self, batch, split):
        """Similar to traning step"""

        x, quilt, y, idx = batch
        # TODO: make this into another class
        if self.cfg.model.fusion.type == 'EarlyFusionModel':
            logit, M_loss = self.model(x, quilt)   # M_loss: overall sparcity loss
        else: 
            logit = self.model(x, quilt)

        logit = logit.reshape(-1)
        y = y.float()
        loss = self.loss_fn(logit, y)

        # make this into another class
        if self.cfg.model.fusion.type == 'EarlyFusionModel':
            if self.cfg.train.lambda_sparse is not None:
                loss -= self.cfg.train.lambda_sparse * M_loss

        if split != 'test':
            self.log(f'{split}_loss', loss, on_epoch=True, on_step=False, logger=True, prog_bar=True)

        #self.y_list += y.detach().cpu().tolist()
        #self.prob_list += prob.detach().cpu().tolist()
        #return_dict = {'loss': loss}

        # TODO: make this into another class
        if self.cfg.model.fusion.type == 'EarlyFusionModel':
            if self.cfg.train.lambda_sparse is not None:
                loss -= self.cfg.train.lambda_sparse * M_loss

        return_dict = {
            'y': y,
            'idx': idx, 
            'loss': loss,
            'logit': logit}

        return return_dict

    def shared_epoch_end(self, step_outputs, split):

        logit = torch.cat([x['logit'] for x in step_outputs])
        y = torch.cat([x['y'] for x in step_outputs])
        #idx = list([x['idx'] for x in step_outputs][0])
        idx = [v for w in [x['idx'] for x in step_outputs] for v in w]

        prob = torch.sigmoid(logit)

        #y = self.y_list
        #prob = self.prob_list
        #self.y_list = []
        #self.prob_list = []

        # get metrics
        y= y.detach().cpu().tolist()
        prob = prob.detach().cpu().tolist()
        auroc = get_auroc(y, prob)
        auprc = get_auprc(y, prob)
        if split != 'test':
            self.log(f'{split}/auroc', auroc, on_epoch=True, logger=True, prog_bar=True)
            self.log(f'{split}/auprc', auprc, on_epoch=True, logger=True, prog_bar=True) 
        else: 
            print(f'AUROC: {auroc}')
            print(f'AUPRC: {auprc}')

        if split == 'test': 
            results = {'y': y, 'prob': prob, 'id': idx}
            results_df = pd.DataFrame.from_dict(results)
            csv_path = os.path.join(self.cfg.save_dir, 'results.csv')
            results_df.to_csv(csv_path, index=False)
            print(f'RESULTS_PATH {csv_path}') 