import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib

from fetal_net.prediction import run_validation_cases
import argparse

from fetal_net.utils.cut_relevant_areas import find_bounding_box, cut_bounding_box


def main(config, split='test', overlap_factor=1, config2=None, use_augmentations=False):
    prediction_dir = os.path.abspath(os.path.join(config['base_dir'], 'predictions', split))

    indices_file = {
      "test": config["test_file"],
      "val": config["validation_file"],
      "train": config["training_file"]
    }[split]
    run_validation_cases(validation_keys_file=indices_file,
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         hdf5_file=config["data_file"],
                         output_dir=prediction_dir,
                         overlap_factor=overlap_factor,
                         patch_shape=config["patch_shape"] + [config["patch_depth"]],
                         prev_truth_index=config["prev_truth_index"],
                         prev_truth_size=config["prev_truth_size"],
                         pred_index=config["pred_index"],
                         pred_size=config["pred_size"],
                         use_augmentations=use_augmentations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=False, default='../../../../../datadrive/configs/trainset_experiment_2_val_20_train_0_iter')
    #type=str, required=False, default='../../../../../datadrive/configs/trainset_experiment_0_val_20_train_0_iter')
    parser.add_argument("--split", help="What split to predict on? (test/val)",
                        type=str, default='test')
    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=float, default=0.9) # 0.9
    parser.add_argument("--use_augmentations", help="specifies whether to predict on augmentations",
                        type=bool, default=False) # False
    opts = parser.parse_args()

    with open(os.path.join(opts.config_dir, 'config.json')) as f:
        config = json.load(f)

    main(config, opts.split, opts.overlap_factor, use_augmentations=opts.use_augmentations)
