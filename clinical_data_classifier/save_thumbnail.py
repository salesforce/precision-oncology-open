#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:38:16 2019

@author: nnaik
"""

#! /home/ubuntu/anaconda3/bin/python

import openslide
import pickle
import numpy as np
from glob import glob
import os, sys
import pandas as pd
from PIL import Image
import csv
from math import gcd
import random
#from utils import save_as_hdf5
#%%
res = 4000
wsi_dir = '/export/medical_ai/ucsf/RTOG-9413/svs'
thumbnail_dir = os.path.join(wsi_dir, 'thumbnail_' + str(res))

thumbnail_dir = '/export/home/datasets/RTOG-9413_thumbnail_4000'
if not os.path.exists(thumbnail_dir):
    os.makedirs(thumbnail_dir)
#df_labels = pd.read_csv( os.path.join('/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9202', 'val.csv'), delimiter=',')
#df = pd.read_excel( os.path.join('/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9202', 'Baseline_and_results_9202.xlsx'))
#dir_names = df_labels['image'].tolist()
#labels = df_labels['label'].tolist()

dir_names = os.listdir('/export/medical_ai/ucsf/RTOG-9413/svs')
labels = [0] * len(dir_names)

for ix, slide in enumerate(dir_names):
    th_name = 'val_' + str(labels[ix]) + '_' + slide + '.png'
    print(th_name)
    if not os.path.isfile(os.path.join(thumbnail_dir, th_name)):
        try:
            sample = openslide.open_slide(os.path.join(wsi_dir, slide))
            thumbnail = sample.get_thumbnail((res, res))
            thumbnail.save(os.path.join(thumbnail_dir,th_name))
        except:
            print('NO IMAGE: '+ slide)
    else:
        print("DONE " + slide)
