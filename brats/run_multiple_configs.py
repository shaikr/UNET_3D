import os
from pathlib import Path
import random
import pickle
import os
import json


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


# field_to_change_in_augment = {
#                               "contrast": {
#                                   'prob': 0.5,
#                                   'min_factor': 0.1,
#                                   'max_factor': 0.1
#                               },
#                               "poisson_noise": 0.5,
#                               "gaussian_filter": {
#                               "prob": 0.5,
#                               "max_sigma": 1.5
#                               },
#                               'translate': (15, 15, 7), "scale": (0.1, 0.1, 0), "rotate": (0, 0, 90),
#                               "speckle_noise": {
#                                             "prob": 0.5,
#                                             "sigma": 0.05
#                                         },
#                               "piecewise_affine": {
#                                         'scale': 2
#                                     },
#                               "elastic_transform": {
#                                             'alpha': 5,
#                                             'sigma': 1.5
#                                         },
#                               "intensity_multiplication": [0.6, 1.4],
#                               "coarse_dropout": {
#                                             "rate": 0.2,
#                                             "size_percent": [0.10, 0.30],
#                                             "per_channel": True
#                                         },
#                               "transpose_prob": 0.5
#                               }

# field_to_change_in_arch = [#{"truth_index":2, "input_shape":[64,64,5], "patch_shape":[64,64], "patch_depth":5},
#                            #{"truth_index":2, "input_shape":[96,96,5], "patch_shape":[96,96], "patch_depth":5},
#                            {"truth_index":1, "input_shape":[96,96,3], "patch_shape":[96,96], "patch_depth":3},
#                            {"truth_index":3, "input_shape":[96,96,7], "patch_shape":[96,96], "patch_depth":7},
#                            {"truth_index":2, "input_shape":[128,128,5], "patch_shape":[128, 128], "patch_depth":5},
#                            {"truth_index":1, "input_shape":[64,64,3], "patch_shape":[64, 64], "patch_depth":3}]

field_to_change_in_arch = [
    {"normalization": 'each_stretch'},
    {"normalization": 'each_stretch_and'},
    {"normalization": 'each_clip'},
    {"normalization": 'each_clip_and'},
    {"normalization": 'each_minmax'}
]

field_to_change_in_arch = [
    {"3D": True, "truth_size": 16, "input_shape": [1,64,64,16], "patch_depth": 16, "truth_index": 0, "model_name": "unet_model_3d"},
    {"3D": True, "truth_size": 6, "input_shape": [1,64,64,16], "patch_depth": 16, "truth_index": 5, "model_name": "unet_model_3d"}
]

field_to_change_in_arch = [
    {"3D": True, "pred_index": 0, "pred_size": 16, "truth_size": 16, "input_shape": [1,64,64,32], "patch_depth": 16, "truth_index": 16, "model_name": "unet_model_3d"},
    {"3D": True, "pred_index": 0, "pred_size": 16, "truth_size": 6, "input_shape": [1,64,64,32], "patch_depth": 16, "truth_index": 21, "model_name": "unet_model_3d"}
]

base_config_path = r"/datadrive/configs/20200429_single_res_with_pred_again/config.json"


def init_exp(exp_name, config):
    full_exp_path = os.path.join('../../../../../datadrive/configs', exp_name)
    config["base_dir"] = full_exp_path
    config["overwrite"] = False
    config["data_file"] = os.path.join(config["base_dir"], "fetal_data.h5")
    config["model_file"] = os.path.join(config["base_dir"], "fetal_net_model")
    Path(config["base_dir"]).mkdir(parents=True, exist_ok=True)
    config["data_file"] = os.path.join(config["base_dir"], "fetal_data.h5")
    config["model_file"] = os.path.join(config["base_dir"], "fetal_net_model")
    with open(os.path.join(full_exp_path, 'config.json'), mode='w') as f:
        json.dump(config, f, indent=2)


# k = "truth_index", "input_shape"
experiment_names = ['20200619_3d_unet_attempts_with_pred_0{}', '20200621_3d_unet_attempts_with_pred_later_slices_0{}']
for i, d in enumerate(field_to_change_in_arch):
    print(d)
    with open(base_config_path) as f:
        config = json.load(f)
    experiment_name = '20200621_3d_unet_attempts_with_pred_later_slices_0{}'.format(i)
    for k in d:
        config[k] = d[k]
        #experiment_name = experiment_name + '_' + str(k) + '_' + str(d[k])
    init_exp(experiment_name, config)
    cmd = "python train_fetal.py --experiment_name='{}'".format(experiment_name)
    print(cmd)
    os.system(cmd)
    print("Finished training") #, now running on test")
    #conf_dir = '../../../../../datadrive/configs/' + experiment_name
    #cmd = "python predict.py --split='test' --config='{}'".format(conf_dir)
    #print(cmd)
    #os.system(cmd)


# with open(base_config_path) as f:
#     config = json.load(f)
# # Run 1st experiment
# experiment_name = 'base_exp_0602'
# init_exp(experiment_name, config)
# cmd = f"python train_fetal.py --experiment_name='{experiment_name}'"
# print(cmd)
# os.system(cmd)