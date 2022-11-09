

import numpy as np
import os
#import sys

#abs_barc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',))

# stanext dataset
# (1) path to stanext dataset
# STAN_V12_ROOT_DIR = abs_barc_dir + '/datasets/StanfordExtra_V12/'
STAN_V12_ROOT_DIR = '/vol/research/animal_motion_data/datasets_3rd_party/Stanford_Dogs/'
# IMG_V12_DIR = os.path.join(STAN_V12_ROOT_DIR, 'StanExtV12_Images')		      
IMG_V12_DIR = os.path.join(STAN_V12_ROOT_DIR, 'data')
JSON_V12_DIR = os.path.join(STAN_V12_ROOT_DIR, 'annotations', "StanfordExtra_v12.json")    
STAN_V12_TRAIN_LIST_DIR = os.path.join(STAN_V12_ROOT_DIR, 'annotations', 'train_stanford_StanfordExtra_v12.npy')  
STAN_V12_VAL_LIST_DIR = os.path.join(STAN_V12_ROOT_DIR, 'annotations', 'val_stanford_StanfordExtra_v12.npy')  
STAN_V12_TEST_LIST_DIR = os.path.join(STAN_V12_ROOT_DIR, 'annotations', 'test_stanford_StanfordExtra_v12.npy')  
# (2) path to related data such as breed indices and prepared predictions for withers, throat and eye keypoints 
STANEXT_RELATED_DATA_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stanext_related_data')

# image crop dataset (for demo, visualization)
TEST_IMAGE_CROP_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'test_image_crops') 
