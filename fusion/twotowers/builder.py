import torch
import torch.nn as nn
import copy

from omegaconf import OmegaConf
from pytorch_tabnet import tab_network, metrics
from . import models
from . import lightning
from . import datasets
from .constants import *


def build_data_module(cfg):
    if cfg.data.dataset == 'emr':
        return datasets.data_module.EMRDataModule(cfg)
    elif cfg.data.dataset == 'image': 
        return datasets.data_module.ImageDataModule(cfg)
    elif cfg.data.dataset == 'multimodal':
        return datasets.data_module.MultimodalDataModule(cfg)
    else:
        raise NotImplementedError(f'Data module not implemented for {cfg.data.dataset}')


def build_lightning_model(cfg, dm):
    if cfg.data.dataset == 'emr':
        if cfg.phase == 'pretrain': 
            model = lightning.EMRPretrainLightningModule
        else:
            model = lightning.EMRLightningModule
    elif cfg.data.dataset == 'image': 
            model = lightning.ImageLightningModule
    elif cfg.data.dataset == 'multimodal': 
            model = lightning.FusionLightningModule
    else: 
        raise NotImplementedError(f'Model not implemented for {cfg.data.dataset}')

    if OmegaConf.is_none(cfg, "checkpoint"):
        return model(cfg, dm)
    else:
        ckpt = cfg.pop('checkpoint')
        print(f'Using checkpoint from {ckpt}')
        return model.load_from_checkpoint(ckpt, cfg=cfg, dm=dm)


def build_model(cfg, dm):

    # model specific weights (i.e from pretrained)
    checkpoint_path = None
    if not OmegaConf.is_none(cfg.model, "checkpoint"):
        checkpoint_path = cfg.model.pop('checkpoint')

    if cfg.data.dataset == 'emr':
        # get feature dimentions
        cat_idxs, cat_dims, cat_emb_dims = dm.train_dataloader().dataset.get_category_info()
        input_dim = dm.train_dataloader().dataset.get_input_dim()

        # pretrain emr model 
        if cfg.phase == 'pretrain': 
            model = tab_network.TabNetPretraining(
                input_dim=input_dim,
                cat_idxs=cat_idxs,
                cat_dims=cat_dims, 
                cat_emb_dim=cat_emb_dims, 
                **cfg.model)  
        else:
            # supervised model doesn't take pretraining ratio
            if not OmegaConf.is_none(cfg.model, "pretraining_ratio"):
                cfg.model.pop('pretraining_ratio')

            model = tab_network.TabNet(
                input_dim=input_dim,
                output_dim=len(TARGETS[cfg.data.target]),  
                cat_idxs=cat_idxs,
                cat_dims=cat_dims, 
                cat_emb_dim=cat_emb_dims, 
                **cfg.model)  

            # load checkpoint
            # TODO: make function (repreated)
            if checkpoint_path is not None: 
                update_state_dict = copy.deepcopy(model.state_dict())
                ckpt = torch.load(checkpoint_path)
                for param, weights in ckpt['state_dict'].items():
                    if param.startswith("encoder"):
                        # Convert encoder's layers name to match
                        new_param = "tabnet." + param
                    else:
                        new_param = param
                    if model.state_dict().get(new_param) is not None:
                        # update only common layers
                        update_state_dict[new_param] = weights
                model.load_state_dict(update_state_dict)
                print(f'\n\nLoaded ckpt from {checkpoint_path}')
        return model

    elif cfg.data.dataset == 'image': 
        model_type = cfg.model.pop('type')
        model = getattr(models.image_models, model_type)
        return model(num_classes=len(TARGETS[cfg.data.target]), **cfg.model)

    elif cfg.data.dataset == 'multimodal':

        # check for checkpoints
        img_checkpoint_path = None
        emr_checkpoint_path = None
        if not OmegaConf.select(cfg, "model.emr.checkpoint") is None:
            emr_checkpoint_path = cfg.model.emr.pop('checkpoint')
            emr_model_config_path = '/'.join(emr_checkpoint_path.split('/')[:-1]) + '/config.yaml'
            emr_model_config = OmegaConf.load(emr_model_config_path)
            cfg.model.emr = emr_model_config.model
        
        if not OmegaConf.select(cfg, "model.vision.checkpoint") is None:
            img_checkpoint_path = cfg.model.vision.pop('checkpoint')
            img_model_config_path = '/'.join(img_checkpoint_path.split('/')[:-1]) + '/config.yaml'
            img_model_config = OmegaConf.load(img_model_config_path)
            cfg.model.vision = img_model_config.model

        # EMR model
        cat_idxs, cat_dims, cat_emb_dims = dm.train_dataloader().dataset.get_category_info()
        input_dim = dm.train_dataloader().dataset.get_input_dim()
        emr_model = tab_network.TabNet(
            input_dim=input_dim,
            output_dim=len(TARGETS[cfg.data.target]),  
            cat_idxs=cat_idxs,
            cat_dims=cat_dims, 
            cat_emb_dim=cat_emb_dims, 
            **cfg.model.emr) 
        if emr_checkpoint_path is not None: 
            update_state_dict = copy.deepcopy(emr_model.state_dict())
            ckpt = torch.load(emr_checkpoint_path, map_location='cuda:0')
            for param, weights in ckpt['state_dict'].items():
                if param.startswith("encoder"):
                    # Convert encoder's layers name to match
                    new_param = "tabnet." + param
                else:
                    new_param = param
                if emr_model.state_dict().get(new_param) is not None:
                    # update only common layers
                    update_state_dict[new_param] = weights
            emr_model.load_state_dict(update_state_dict)
            print('\nLOADED EMR MODEL STATE DICT\n')

        # Vision model 
        if not OmegaConf.select(cfg, "model.vision") is None:  # early fusion
            vision_type = cfg.model.vision.pop('type')
            image_model = getattr(models.image_models, vision_type)(
                num_classes=len(TARGETS[cfg.data.target]), **cfg.model.vision)
            if img_checkpoint_path is not None: 
                update_state_dict = copy.deepcopy(image_model.state_dict())
                ckpt = torch.load(img_checkpoint_path, map_location='cuda:0')
                update_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in ckpt['state_dict'].items()}
                image_model.load_state_dict(update_state_dict)
                print('LOADED IMAGE MODEL STATE DICT')
        else: 
            image_model = None

        # Fusion model
        fusion_type = cfg.model.fusion.type
        fusion_configs = {k:v for k,v in cfg.model.fusion.items() if k != 'type'}
        fusion_model = getattr(models.fusion_models, fusion_type)(
            cfg, emr_model, image_model=image_model, num_classes=len(TARGETS[cfg.data.target]), **fusion_configs)
        return fusion_model
    else: 
        raise NotImplementedError(f'Model not implemented for {cfg.data.dataset}')


def build_optimizer(cfg, model):

    params = list(model.named_parameters())
    optimizer_name = cfg.train.optimizer.pop('name')
    optimizer = getattr(torch.optim, optimizer_name) 

    if cfg.data.dataset == 'multimodal':

        def is_img_model(n): return 'image_model' in n
        def is_emr_model(n): return 'emr_model' in n

        img_lr = cfg.train.optimizer.pop('img_lr')
        emr_lr = cfg.train.optimizer.pop('emr_lr')

        params = [
            {"params": [p for n,p in params if is_img_model(n)], 'lr': img_lr},
            {"params": [p for n,p in params if is_emr_model(n)], 'lr': emr_lr},
            {"params": [p for n,p in params if (not is_emr_model(n) and not is_img_model(n))], 'lr': cfg.lightning.trainer.lr}, ]

        return optimizer(params, **cfg.train.optimizer)
    else: 
        params = [p for _,p in params if p.requires_grad]
        return optimizer(params, lr=cfg.lightning.trainer.lr, **cfg.train.optimizer)




def build_scheduler(cfg, optimizer):
    if cfg.train.scheduler.name is not None: 
        scheduler_name = cfg.train.scheduler.pop('name')
        monitor = cfg.train.scheduler.pop('monitor')
        interval = cfg.train.scheduler.pop('interval')
        frequency = cfg.train.scheduler.pop('frequency')
        
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_class(optimizer, **cfg.train.scheduler) 
    else: 
        scheduler = None
        monitor = None
        interval = None
        frequency = None

    scheduler = {
        'scheduler': scheduler,
        'monitor': monitor,
        'interval': interval,
        'frequency': frequency}

    return scheduler


def build_loss(cfg):
    if cfg.phase == 'pretrain': 
        loss_fn = metrics.UnsupervisedLoss
        return loss_fn
    else: 
        loss_fn_name = cfg.train.loss_fn.pop('name')
        loss_fn = getattr(nn, loss_fn_name) 
        return loss_fn(**cfg.train.loss_fn)
