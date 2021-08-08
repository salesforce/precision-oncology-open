"""
TODO: consider merging the data modules. Only difference is the dataset class used
"""

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from .. import builder
from .emr_dataset import EMRDataset 
from .image_dataset import ImageDataset 
from .multimodal_dataset import MultimodalDataset 


class EMRDataModule(pl.LightningDataModule):

    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        dataset = EMRDataset(self.cfg, split='train')
        
        if self.cfg.data.weighted_sample:
            sampler = dataset.get_sampler()
            return DataLoader(
                dataset,
                pin_memory=False,
                drop_last=True, 
                shuffle=False,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers)
        else: 
            return DataLoader(
                dataset,
                pin_memory=False,
                drop_last=True, 
                shuffle=True,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers)

    def val_dataloader(self):
        dataset = EMRDataset(self.cfg, split='valid')
        return DataLoader(
            dataset,
            pin_memory=False,
            drop_last=False, 
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers)

    def test_dataloader(self):
        dataset = EMRDataset(self.cfg, split='test')
        return DataLoader(
            dataset,
            pin_memory=False,
            drop_last=False, 
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers)


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        dataset = ImageDataset(self.cfg, split='train')
        
        if self.cfg.data.weighted_sample:
            sampler = dataset.get_sampler()
            return DataLoader(
                dataset,
                pin_memory=False,
                drop_last=True, 
                shuffle=False,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers)
        else: 
            return DataLoader(
                dataset,
                pin_memory=False,
                drop_last=True, 
                shuffle=True,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers)

    def val_dataloader(self):
        dataset = ImageDataset(self.cfg, split='valid')
        return DataLoader(
            dataset,
            pin_memory=False,
            drop_last=False, 
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers)

    def test_dataloader(self):
        dataset = ImageDataset(self.cfg, split='test')
        return DataLoader(
            dataset,
            pin_memory=False,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers)


class MultimodalDataModule(pl.LightningDataModule):

    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        dataset = MultimodalDataset(self.cfg, split='train')
        
        if self.cfg.data.weighted_sample:
            sampler = dataset.get_sampler()
            return DataLoader(
                dataset,
                pin_memory=False,
                drop_last=True, 
                shuffle=False,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers)
        else: 
            return DataLoader(
                dataset,
                pin_memory=False,
                drop_last=True, 
                shuffle=True,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers)

    def val_dataloader(self):
        dataset = MultimodalDataset(self.cfg, split='valid')
        return DataLoader(
            dataset,
            pin_memory=False,
            drop_last=False, 
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers)

    def test_dataloader(self):
        dataset = MultimodalDataset(self.cfg, split='test')
        return DataLoader(
            dataset,
            pin_memory=False,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers)

