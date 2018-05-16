import utils.rate_distortion as rd

"""
In params, specify:
    DIRECTORIES
        output_dir: (str) file containing learned coefficients, or a directory containing 
            multiple coeffs files (for lca mods).
        input_dir: (str) file (file containing images for which coeffs were fit).
        save_name: (str) where to save the computed rate_distortion df
    DATA PARAMS
        n_pixels: (int) number of pixels in each image patch
    MODEL PARAMS
        mod_names: (lst of str) contains model type, e.g. ['lca'] or ['rg']
        LCA-SPECIFIC (Set as [None] if not running for lca)
            lams: (lst of int) the range of lambdas used in multiple models
            costs: (lst of str) the range of costs used in multiple models
            n_neurons: (lst of int) the range of costs used in multiple models
            version: (str) version number, e.g. 'v_1.0'
    RD PARAMS
        disc_type: (str) method for quantization. either 'uniform' or 'vq'
        n_bins: (lst of int) list of number of bins to use in progressive quantization
        print
    RUN PARAMS
        print: (bool) prints the dataframe as it generates if True
    
        
This script generates and saves a dataframe containing model parameters and the following 
computed values for each level of quantization specified in n_bins:

"p_active": percent non-zero coefficients
"mse": mean squared error
"mse_sd": standard deviation of mean squared error across image patches
"entropy": mean marginal entropy of the coefficients
"transmission_rate": mean marginal entropy * overcompleteness
"""

params = {
    ## DIRECTORIES
    "input_dir": '/media/tbell/sanborn/rd_analysis/inputs/vh_test_ftwhite2.npz', 
    "out_dir": '/media/tbell/sanborn/rd_analysis/outputs/lca_ft_posneg/coeffs/',
    "plot_dir": '/media/tbell/sanborn/rd_analysis/outputs/lca_ft_posneg/plots/',
    "save_name": '/media/tbell/sanborn/rd_analysis/outputs/lca_ft_posneg/rd_by_img_uniform_.9',
    ## DATA PARAMS
    "n_pixels": 256,
    "white": True,
    "im_edge_size": 1024,
    "num_images": 25,
    "whiten_method": "FT",
    "patch_edge_size": 16,
    ## MODEL PARAMS
    "mod_names": ['lca'], 
    ## LCA PARAMS
    "lams": [0.9],
    "costs": ["l0"],
    "n_neurons": [256],
    "version": 'v2.0',
    ## RD PARAMS
    "disc_type": 'uniform',
    "bins": [.1, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    "set_fixed_range": True,
    "range_min": -60,
    "range_max": 60,
    "adjust_bf_length": True,
    "crop_border": True,
    ## RUN PARAMS
    "print": True,
    "plot_coeff_hists": False
}

# params = {
#     ## DIRECTORIES
#     "input_dir": '/media/tbell/sanborn/rd_analysis/inputs/vh_test_lpf2.npz', 
#     "out_dir": '/media/tbell/sanborn/rd_analysis/outputs/pca/coeffs/',
#     "plot_dir": '/media/tbell/sanborn/rd_analysis/outputs/pca/plots/',
#     "save_name": '/media/tbell/sanborn/rd_analysis/outputs/pca/rd_by_img_lloyd_v2',
#     ## DATA PARAMS
#     "n_pixels": 256,
#     "white": False,
#     "im_edge_size": 1024,
#     "num_images": 25,
#     "whiten_method": "FT",
#     "patch_edge_size": 16,
#     ## MODEL PARAMS
#     "mod_names": ['pca'], 
#     ## LCA PARAMS
#     "lams": [None], 
#     "costs": [None],
#     "n_neurons": [256],
#     "version": 'v1.0',
#     ## RD PARAMS
#     "disc_type": 'lloyd_max',
#     "bins": [.002, .007],
#     "set_fixed_range": True,
#     "range_min": -3,
#     "range_max": 3,
#     "adjust_bf_length": False,
#     "crop_border": True,
#     ## RUN PARAMS
#     "print": True,
#     "plot_coeff_hists": False
# }


# params = {
#     ## DIRECTORIES
#     "input_dir": '/media/tbell/sanborn/rd_analysis/inputs/vh_test_lpf2.npz', 
#     "out_dir": '/media/tbell/sanborn/rd_analysis/outputs/ica/coeffs/',
#     "plot_dir": '/media/tbell/sanborn/rd_analysis/outputs/ica/plots/',
#     "save_name": '/media/tbell/sanborn/rd_analysis/outputs/ica/rd_by_img_lloyd_v2_lowent',
#     ## DATA PARAMS
#     "n_pixels": 256,
#     "white": False,
#     "im_edge_size": 1024,
#     "num_images": 25,
#     "whiten_method": "FT",
#     "patch_edge_size": 16,
#     ## MODEL PARAMS
#     "mod_names": ['ica'], 
#     ## LCA PARAMS
#     "lams": [None], 
#     "costs": [None],
#     "n_neurons": [256],
#     "version": 'v1.0',
#     ## RD PARAMS
#     "disc_type": 'lloyd_max',
#     "bins": [30, 40, 50, 60, 70, 80],
#     "set_fixed_range": True,
#     "range_min": -40,
#     "range_max": 40,
#     "adjust_bf_length": False,
#     "crop_border": True,
#     ## RUN PARAMS
#     "print": True,
#     "plot_coeff_hists": False
# }

rate_distortion = rd.rate_dist(params)
rate_distortion.to_pickle(params["save_name"])
