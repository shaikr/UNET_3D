import json
import os
import sys
import argparse
import numpy as np
import nibabel as nib
from fetal_net.prediction import batch_iterator, patch_wise_prediction
from fetal_net.training import load_old_model
from fetal_net.postprocess import postprocess_prediction
from fetal_net.metrics import dice_coefficient_np, get_surface_distances
from .utils import get_last_model_path

VOLUME_FILENAME = 'volume.nii'
TRUTH_FILENAME = 'volume.nii'
CONFIG_FILENAME = 'config.json'


def get_subject_id_data(data_folder, subject_id):
    vol_im = nib.load(os.path.join(data_folder, subject_id, VOLUME_FILENAME))
    truth_im = nib.load(os.path.join(data_folder, subject_id, TRUTH_FILENAME))
    return vol_im.get_data(), truth_im.get_data(), truth_im.affine, truth_im.header


def get_starting_slice_ind(val_slices, previous_truth_ind):
    if type(val_slices) == list:
        val_slices = min(val_slices)
    if previous_truth_ind > val_slices:
        return None
    starting_slice = val_slices - previous_truth_ind
    return starting_slice


def get_config_and_model(configs_folder, model_folder):
    with open(os.path.join(configs_folder, model_folder, CONFIG_FILENAME)) as f:
        config = json.load(f)
    model = load_old_model(get_last_model_path(config["model_file"]))
    return config, model


def get_placenta_start_ind(truth_data):
    truth_sums = np.sum(np.sum(truth_data, axis=0), axis=0)
    pla_start_ind = next(x for x, val in enumerate(truth_sums) if val > 0)
    return pla_start_ind


def main(configs_folder, data_folder, model_folder, subject_id, validated_ind, overlap_factor, propagate, pred_path):

    vol, truth, affine, header = get_subject_id_data(data_folder, subject_id)
    pred = nib.load(pred_path).get_data()
    pred_bin = postprocess_prediction(pred)
    print(f"Initial DICE: {dice_coefficient_np(truth, pred_bin)}")
    config, model = get_config_and_model(configs_folder, model_folder)
    if validated_ind < 0:
        validated_ind = get_placenta_start_ind(truth)
    starting_slice = get_starting_slice_ind(validated_ind, config["prev_truth_index"])
    fixing_slice = starting_slice + config["truth_index"]
    if not starting_slice or starting_slice >= vol.shape[-1]:
        print(f"Error! Validated slice {validated_ind} is out of range")
        sys.exit(0)

    new_pred, new_var = patch_wise_prediction(model, vol, config["patch_shape"], overlap_factor, is3d=config["3D"],
                                              truth_data=truth, prev_truth_index=config["prev_truth_index"],
                                              prev_truth_size=config["prev_truth_size"], specific_slice=starting_slice)
    pred[:, :, fixing_slice] = new_pred[:, :, fixing_slice]
    pred_bin = postprocess_prediction(pred)
    print(f"DICE after updating slice {fixing_slice}: {dice_coefficient_np(truth, pred_bin)}")
    validated_ind = fixing_slice
    starting_slice = get_starting_slice_ind(validated_ind, config["prev_truth_index"])

    if propagate:
        while starting_slice <= (vol.shape[-1] - config["patch_depth"]):
            fixing_slice = starting_slice + config["truth_index"]
            new_pred, new_var = patch_wise_prediction(model, vol, config["patch_shape"], overlap_factor,
                                                      is3d=config["3D"],
                                                      truth_data=truth, prev_truth_index=config["prev_truth_index"],
                                                      prev_truth_size=config["prev_truth_size"],
                                                      specific_slice=starting_slice)
            pred[:, :, fixing_slice] = new_pred[:, :, fixing_slice]
            pred_bin = postprocess_prediction(pred)
            print(f"DICE after updating slice {fixing_slice}: {dice_coefficient_np(truth, pred_bin)}")
            validated_ind = fixing_slice
            starting_slice = get_starting_slice_ind(validated_ind, config["prev_truth_index"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_folder", help="specifies config folder path",
                        type=str, required=False, default=r"/datadrive/configs")
    parser.add_argument("--data_folder", help="where is the relevant data",
                        type=str, required=False, default=r"/home/Shai/placenta_data")
    parser.add_argument("--overlap_factor", help="specifies adjacent patches overlap",
                        type=float, required=False, default=0.8)
    parser.add_argument("--propagate", help="specifies whether to propagate the correction",
                        type=bool, default=False, action='store_true')
    parser.add_argument("--model_folder", help="specifies what model to load, must be relevant to working with truth",
                        type=str, required=True)
    parser.add_argument("--subject_id", help="specifies which scan to go over",
                        type=str, required=True)
    parser.add_argument("--validated_ind", help="specifies what slice is validated",
                        type=int, required=False, default=-1)
    parser.add_argument("--starting_pred_path", help="specifies adjacent patches overlap",
                        type=str, required=True)
    # Add option to just start where placenta starts
    # Add option to track progress when improving a prediction
    opts = parser.parse_args()
    main(opts.configs_folder, opts.data_folder, opts.model_folder, opts.subject_id, opts.validated_ind,
         opts.overlap_factor, opts.propagate, opts.pred_path)
