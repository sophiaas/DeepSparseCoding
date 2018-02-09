import utils.rate_distortion as rd

params = {
    "n_pixels": 256,
    # Model type, e.g. 'lca' or 'rg'. Must run lca separately
    "mod_names": ['rg'], 
    # Range of lambdas, costs, and n_neurons. None for models other than lca
    "lams": [None], 
    "costs": [None],
    "n_neurons": [256],
    "version": None,
    # RD params
    "disc_type": 'uniform', # 'uniform' or 'vq'
    "unwhiten": False, 
    "n_bins": [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 100, 500, 1000],
    "print": True,
    # Directories
    # Input is images file.
    # Out is coeffs file (for rg, pca) or directory containing coeffs for multiple lca mods.
    # Save is where to save RD analysis df.
    "input_dir": '/media/tbell/sanborn/rd_analysis/inputs/test_pca_wht.npz', 
    "out_dir": '/media/tbell/sanborn/rd_analysis/outputs/rg_pca/coeffs/rg_pca_coeffs.p',
    "save_name": '/media/tbell/sanborn/rd_analysis/outputs/test_mods/rd_test'
}

rate_distortion = rd.rate_dist(params)
rate_distortion.to_pickle(params["save_name"])
