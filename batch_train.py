gpu_num = '1'

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

''''''''''''
## Set base directory
base_dir = '/media/tbell/sanborn/rd_analysis/'

## Set batch parameters
batch_params = {
    ## Specify model type and data type
    'model_type': 'lca',
    'data_type': 'vanhateren', 
    'data_file': 'vh_train_lpf2', # 'test_100000','test_nowht_tiled.npz', 'test_100000.npz', 'baboon_zca_wht',
    'model_class': 'lca_lpf',

    ## Set training type
    'process': train_mod, # train_mod or infer_coeffs
    'input_type': 'overlapping', # 'tiled' or 'overlapping'
    
    ## Set model parameters
    'num_neurons': [256],
    'thresh_types': ['l1'],
    'lambdas': [1],
    'version': '9.0',
    
    ## Set training parameters   
    "dt": .001,
    'tau': .5,
    'lrs': [[.05], [.05]],  # set lrs per lambda
    "batch_size": 100,
    'nbs': [1000000], # set nbs per lambda
    'num_steps': 40,
    'decay_steps': [[2500]], # set decay_steps per lambda
    'decay_rate': [[.8]], # set decay_rate per lambda
    'staircase': [True],
    'rectify_a': False,
    
    ## Set plotting parameters
    "gen_plot_int": 100,
    "log_int": 1000,
    
    ## Load checkpoint?
    "cp_load": False,
    "cp_load_name": None
}

## Import model params
params, schedule = pp.get_params(batch_params['model_type'])
if 'rand_seed' in params.keys():
    params['rand_state'] = np.random.RandomState(params['rand_seed'])
params['data_type'] = batch_params['data_type']

## Set device
params['device'] = '/gpu:'+gpu_num

## Set base directory
params['base_dir'] = base_dir

params['out_dir'] = params['base_dir'] + 'outputs/' + batch_params['model_class'] +'/'
params['input_dir'] = params['base_dir'] + 'inputs/' + batch_params['data_file']+'.npz'

if batch_params['process'] == infer_coeffs:
    params['out_dir'] = params['out_dir'] + 'coeffs/'   
#     if batch_params['input_type'] == 'tiled':
#         params['out_dir'] = params['out_dir'] + 'tiled/'+batch_params['data_file']+'/'         
    
## Save batch params
np.savez(params['out_dir'] + 'batch_params.npz', batch_params)
    
''''''''''''
## Import data
with np.load(params['input_dir']) as d:
    data = d['arr_0'].item()  
params["num_pixels"] = params["patch_edge_size"] ** 2
params["input_shape"] = [params["patch_edge_size"] ** 2]

for n in batch_params['num_neurons']:
    for thresh in batch_params['thresh_types']:
        for idx, lam in enumerate(batch_params['lambdas']):
            schedule[0]['num_steps'] = batch_params['num_steps']
            params['batch_size'] = batch_params['batch_size']
            params['num_steps'] = batch_params['num_steps']
            params['version'] = batch_params['version']
            params["dt"] = batch_params["dt"]
            params["tau"] = batch_params["tau"]
            params['rectify_a'] = batch_params['rectify_a']
            params["cp_load"] = batch_params["cp_load"]
            params["cp_load_name"] = batch_params["cp_load_name"]
            
            schedule[0]['staircase'] = batch_params['staircase']
            schedule[0]['weight_lr'] = batch_params['lrs'][idx]
            schedule[0]['num_batches'] = batch_params['nbs'][idx]
            schedule[0]['decay_steps'] = batch_params['decay_steps'][idx]
            schedule[0]['decay_rate'] = batch_params['decay_rate'][idx]
            schedule[0]['sparse_mult'] = lam
            params['num_neurons'] = n
            if thresh == "l0":
                params['thresh_type'] = "hard"
            elif thresh == "l1":
                params['thresh_type'] = "soft"

            params['model_name'] = batch_params['model_type'] + '_' + str(n) + '_' + thresh + '_' + str(lam) + '_v' + params['version']
            params['cp_dir'] = params['base_dir'] + 'outputs/' + batch_params['model_class'] + '/' + params['model_name'] + '/checkpoints/'
            print('Beginning ' + params['model_name'])
            batch_params['process'](data, params, schedule)
            print('Training ' + params['model_name'] + ' Complete\n') 