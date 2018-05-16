gpu_num = '0'

import os
os.environ['CUDA_VISIBLE_DEVICES']=gpu_num

import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import json as js
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds
from utils.training import train_mod, infer_coeffs

## Set base directory
base_dir = '/media/tbell/sanborn/rd_analysis/'

# params = {
#   "batch_size": 100,
#   "center_data": False,
#   "contrast_normalize": False,
#   "cp_int": 10000,
#   "cp_load": False,
#   "cp_load_name": "pretrain_mnist",
#   "cp_load_step": 150000,
#   "cp_load_var": [
#     "phi"
#   ],
#   "cp_load_ver": "0.0",
#   "data_file": "test_ft",
#   "data_shape": [
#     256
#   ],
#   "data_type": "vanhateren",
#   "device": "/gpu:0",
#   "dt": 0.001,
#   "eps": 1e-12,
#   "extract_patches": True,
#   "gen_plot_int": 10000,
#   "log_int": 100,
#   "log_to_file": True,
#   "max_cp_to_keep": 1,
#   "model_name": "lca_vh_ft_1c",
#   "model_type": "lca",
#   "norm_data": False,
#   "norm_weights": True,
#   "num_images": 100,
#   "num_neurons": 256,
#   "num_patches": 1000000.0,
#   "num_steps": 1000,
#   "optimizer": "annealed_sgd",
#   "overlapping_patches": True,
#   "patch_edge_size": 16,
#   "patch_variance_threshold": 1e-06,
#   "rand_seed": 1234567890,
#   "randomize_patches": True,
#   "rectify_a": True,
#   "save_plots": True,
#   "standardize_data": True,
#   "tau": 0.03,
#   "thresh_type": "soft",
#   "vectorize_data": True,
#   "version": "0.0",
#   "whiten_data": True,
#   "whiten_method": "FT"
# }

# schedule= [
#   {
#     "decay_rate": [
#       0.8
#     ],
#     "decay_steps": [
#       25000
#     ],
#     "num_batches": 1024,
#     "sparse_mult": 0.6,
#     "staircase": [
#       True
#     ],
#     "weight_lr": [
#       0.1
#     ],
#     "weights": [
#       "phi"
#     ]
#   }
# ]

params = {
  "model_type": "ica",
  "model_name": "ica_raw_patch_centered",
  "version": "0.0",
  "num_images": 50,
  "vectorize_data": True,
  "norm_data": False,
  "whiten_data": False,
  "extract_patches": True,
  "num_patches": 1e6,
  "patch_edge_size": 16,
  "overlapping_patches": True,
  "randomize_patches": True,
  "patch_variance_threshold": 1e-6,
  "batch_size": 100,
  #"prior": "cauchy",
  "prior": "laplacian",
  "optimizer": "annealed_sgd",
  "cp_int": 10000,
  "max_cp_to_keep": 5,
  "cp_load": False,
  "cp_load_name": "pretrain",
  "cp_load_step": 150000,
  "cp_load_ver": "0.0",
  "cp_load_var": ["a"],
  "log_int": 100,
  "log_to_file": True,
  "gen_plot_int": 500,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/cpu:0",
  "rand_seed": 1234567890,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}

schedule = [
  {"weights": ["a"],
  "weight_lr": [0.01],
  "decay_steps": [30000],
  "decay_rate": [0.6],
  "staircase": [True],
  "num_batches": 1000000}]


params['base_dir'] = base_dir
# params['model_dir'] = "/home/dpaiton/Work/Projects/" + params["model_name"]
params['out_dir'] = params['base_dir'] + 'outputs/'
params['input_dir'] = '/media/tbell/sanborn/rd_analysis/inputs/test_raw_patch_centered.npz'

# params['out_dir'] = params['out_dir'] + 'coeffs/'
    
## Import data
with np.load(params['input_dir']) as d:
    data = d['arr_0'].item()  
params["num_pixels"] = params["patch_edge_size"] ** 2
params["input_shape"] = [params["patch_edge_size"] ** 2]

params['cp_dir'] = params['out_dir']+ '/checkpoints/'
print('Beginning ' + params['model_name'])
train_mod(data, params, schedule)
print('Training ' + params['model_name'] + ' Complete\n') 
