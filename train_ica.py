gpu_num = '0'

import os
os.environ['CUDA_VISIBLE_DEVICES']=gpu_num

import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
import json as js
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds
from utils.training import train_mod, infer_coeffs

## Specify model type and data type
model_type = "ica"
data_type = "vanhateren"
#data_type = "field"

## Import params
params, schedule = pp.get_params(model_type)
if "rand_seed" in params.keys():
    params["rand_state"] = np.random.RandomState(params["rand_seed"])
params["data_type"] = data_type
params["num_neurons"] = 256


""""""""""""
params['model_type'] =  "ica"

batch_params = {
    'data_file': 'vh_train_ftwhite2',
    "process": train_mod,
#     "process": infer_coeffs,
    'input_type': 'overlapping',
    'input_type': 'tiled',   
    'base_dir': '/media/tbell/sanborn/rd_analysis/',
    'out_dir': "/media/tbell/sanborn/rd_analysis/outputs/ica_ftwhite/"
}

params["batch_size"] = 100
params["version"] = '1.0'
params["lpf_data"] = False
params["lpf_cutoff"] = "0.7"
schedule[0]["num_batches"] = 1000000

schedule[0]["weight_lr"] = [0.01]
schedule[0]["decay_steps"] = [25000]
schedule[0]["decay_rate"] = [0.8]
schedule[0]["staircase"] = [True]

params["model_name"] = params["model_type"] + "_v" + params["version"]

""""""""""""

if batch_params['process'] == infer_coeffs:
#     if batch_params['input_type'] == 'tiled':
#         params['out_dir'] = batch_params['base_dir'] + 'outputs/' + params['model_name'] + '/coeffs/tiled/'+batch_params['data_file']+'/'  
#     else: 
    params['out_dir'] = batch_params['base_dir'] + 'outputs/' + params['model_type'] + '/coeffs/'   
else:
    params['out_dir'] = batch_params['base_dir'] + 'outputs/' + params['model_type'] +'/'
params['input_dir'] = batch_params['base_dir'] + 'inputs/' + batch_params['data_file']+'.npz'


# if batch_params['process'] == ica_infer_coeffs:
#     if batch_params['input_type'] == 'tiled':
#         params['out_dir'] = batch_params['base_dir'] + 'outputs/' + params['model_name'] + '/coeffs/tiled/'+params['model_name']+'/'    
#     else: 
#         params['out_dir'] = batch_params['base_dir'] + 'outputs/' + params['model_name'] + '/coeffs/'+params['model_name']+'/'  
# else:
#     params['out_dir'] = batch_params['base_dir'] + 'outputs/' + params['model_name'] +'/'
    
# params['input_dir'] = batch_params['base_dir'] + 'inputs/' + batch_params['data_file']
    
## Import data
with np.load(params['input_dir']) as d:
    data = d['arr_0'].item()  
params["num_pixels"] = params["patch_edge_size"] ** 2
params["input_shape"] = [params["patch_edge_size"] ** 2]

params['cp_dir'] = batch_params['base_dir'] + 'outputs/' + params['model_type'] +'/'+ params['model_name'] +'/checkpoints/'

batch_params["process"](data, params, schedule)