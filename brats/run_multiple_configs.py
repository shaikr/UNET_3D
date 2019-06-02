import os
from pathlib import Path
import random
import pickle
import os
import json


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


field_to_change_in_augment = {'translate': (15, 15, 7), "scale": (0.1, 0.1, 0), "rotate": (0, 0, 90),
                              "contrast": {
                                  'prob': 0.5,
                                  'min_factor': 0.1,
                                  'max_factor': 0.1
                              },
                              "poisson_noise": 0.5,
                              "gaussian_filter": {
                              "prob": 0.5,
                              "max_sigma": 1.5
                              },
                              "speckle_noise": {
                                            "prob": 0.5,
                                            "sigma": 0.05
                                        },
                              "piecewise_affine": {
                                        'scale': 2
                                    },
                              "elastic_transform": {
                                            'alpha': 5,
                                            'sigma': 1.5
                                        },
                              "intensity_multiplication": [0.6, 1.4],
                              "coarse_dropout": {
                                            "rate": 0.2,
                                            "size_percent": [0.10, 0.30],
                                            "per_channel": True
                                        },
                              "transpose_prob": 0.5
                              }

base_config_path = ''

with open(base_config_path) as f:
    config = json.load(f)


def init_exp(exp_name, config):
    full_exp_path = os.path.join('../../../../../datadrive/configs', exp_name)
    config["base_dir"] = full_exp_path
    config["data_file"] = os.path.join(config["base_dir"], "fetal_data.h5")
    config["model_file"] = os.path.join(config["base_dir"], "fetal_net_model")
    Path(config["base_dir"]).mkdir(parents=True, exist_ok=True)
    config["data_file"] = os.path.join(config["base_dir"], "fetal_data.h5")
    config["model_file"] = os.path.join(config["base_dir"], "fetal_net_model")
    with open(os.path.join(full_exp_path, 'config.json'), mode='w') as f:
        json.dump(config, f, indent=2)


# Run 1st experiment
experiment_name = 'base_exp_0602'
init_exp(experiment_name, config)
cmd = f"python train_fetal.py --experiment_name='{experiment_name}'"
print(cmd)
os.system(cmd)

for i, k in field_to_change_in_augment:
    with open(base_config_path) as f:
        config = json.load(f)
    config["augment"][k] = field_to_change_in_augment[k]
    experiment_name = f'add_augmentation_{k}'
    init_exp(experiment_name, config)
    cmd = f"python train_fetal.py --experiment_name='{experiment_name}'"
    print(cmd)
    os.system(cmd)
