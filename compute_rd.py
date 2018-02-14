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
    "input_dir": '/media/tbell/sanborn/rd_analysis/inputs/test_pca_wht.npz', 
    "out_dir": '/media/tbell/sanborn/rd_analysis/outputs/rg_pca/coeffs/rg_pca_coeffs.p',
    "save_name": '/media/tbell/sanborn/rd_analysis/outputs/test_mods/rd_test',
    ## DATA PARAMS
    "n_pixels": 256,
    ## MODEL PARAMS
    "mod_names": ['rg'], 
    ## LCA PARAMS
    "lams": [None], 
    "costs": [None],
    "n_neurons": [256],
    "version": None,
    ## RD PARAMS
    "disc_type": 'uniform',
    "n_bins": [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 100, 500, 1000],
    ## RUN PARAMS
    "print": True
}

rate_distortion = rd.rate_dist(params)
rate_distortion.to_pickle(params["save_name"])
