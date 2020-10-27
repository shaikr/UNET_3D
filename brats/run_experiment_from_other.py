import os
import shutil
import argparse
import sys
import json


fields_to_replace = ["base_dir", "split_dir", "model_file", "training_file", "validation_file", "test_file"]


def main(configs_folder, new_experiment, to_imitate):
    os.makedirs(os.path.join(configs_folder, new_experiment))
    shutil.copytree(os.path.join(configs_folder,to_imitate,'debug_split'), os.path.join(configs_folder,new_experiment,'debug_split'))
    conf_path = os.path.join(configs_folder, to_imitate, 'config.json')
    with open(os.path.join(configs_folder, to_imitate, 'config.json')) as f:
        config = json.load(f)
    
    for f in fields_to_replace:
        config[f] = config[f].replace(to_imitate, new_experiment)
    with open(os.path.join(configs_folder, new_experiment, 'config.json'), mode='w') as f:
        json.dump(config, f, indent=2)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_folder", help="specifies config folder path",
                        type=str, required=False, default=r"/datadrive/configs")
    parser.add_argument("--new_experiment", help="new experiment folder name",
                        type=str, required=True)
    parser.add_argument("--to_imitate", help="specifies what experiment to copy config and split from",
                        type=str, required=True)
    opts = parser.parse_args()

    main(opts.configs_folder, opts.new_experiment, opts.to_imitate)