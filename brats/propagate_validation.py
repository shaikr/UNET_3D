import json
import os
import sys
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from matplotlib.pyplot import imsave
from fetal_net.prediction import batch_iterator, patch_wise_prediction
from fetal_net.training import load_old_model
from fetal_net.postprocess import postprocess_prediction
from fetal_net.metrics import dice_coefficient_np, get_surface_distances
from fetal_net.normalize import *

from utils import get_last_model_path

VOLUME_FILENAME = 'volume.nii'
TRUTH_FILENAME = 'truth.nii'
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


def main(configs_folder, data_folder, model_folder, subject_id, validated_ind, overlap_factor, propagate, pred_path, flip=False):

    vol, truth, affine, header = get_subject_id_data(data_folder, subject_id)
    
    if flip:
        vol = vol[:,:,::-1]
        truth = truth[:,:,::-1]
    
    inflated_vol = np.expand_dims(vol, axis=0).astype(np.float)
    inflated_vol, _, _ = normalize_data_storage_each(inflated_vol)
    inflated_truth = np.expand_dims(truth, axis=0)
    pred = nib.load(pred_path).get_data()
    pred_bin = postprocess_prediction(pred)
    
    if flip:
        pred = pred[:,:,::-1]
        pred_bin = pred_bin[:,:,::-1]
    
    config, model = get_config_and_model(configs_folder, model_folder)
    print(f"Initial DICE: {dice_coefficient_np(truth, pred_bin)}")
    if validated_ind < 0:
        validated_ind = get_placenta_start_ind(truth)
    starting_slice = get_starting_slice_ind(validated_ind, config["prev_truth_index"])
    while not starting_slice:
        validated_ind += 1
        starting_slice = get_starting_slice_ind(validated_ind, config["prev_truth_index"])
    fixing_slice = starting_slice + config["truth_index"]
    imsave('before.png', pred[:,:,fixing_slice])
    if not starting_slice or starting_slice >= vol.shape[-1]:
        print(f"Error! Validated slice {validated_ind} is out of range")
        sys.exit(0)

    new_pred, new_var = patch_wise_prediction(model, inflated_vol, config["patch_shape"] + [config["patch_depth"]],
                                              overlap_factor, is3d=config["3D"], truth_data=inflated_truth,
                                              prev_truth_index=config["prev_truth_index"], batch_size=16,
                                              prev_truth_size=config["prev_truth_size"], specific_slice=starting_slice)
    new_pred = new_pred.squeeze()
    # d_before = dice_coefficient_np(truth[:,:,fixing_slice], pred_bin[:,:,fixing_slice])
    pred[:, :, fixing_slice] = new_pred[:, :, fixing_slice]
    imsave('after.png', pred[:,:,fixing_slice])
    pred_bin = postprocess_prediction(pred)
    #print(f"DICE after updating slice {fixing_slice}: {dice_coefficient_np(truth, pred_bin)}")
    validated_ind = fixing_slice
    inflated_truth[0][:, :, validated_ind] = pred_bin[:, :, fixing_slice]
    starting_slice = get_starting_slice_ind(validated_ind, config["prev_truth_index"])

    if propagate:
        while starting_slice <= (vol.shape[-1] - config["patch_depth"]):
            fixing_slice = starting_slice + config["truth_index"]
            new_pred, new_var = patch_wise_prediction(model, inflated_vol,
                                                      config["patch_shape"] + [config["patch_depth"]],
                                                      overlap_factor, is3d=config["3D"], truth_data=inflated_truth,
                                                      prev_truth_index=config["prev_truth_index"],
                                                      prev_truth_size=config["prev_truth_size"],
                                                      specific_slice=starting_slice, batch_size=16)
            new_pred = new_pred.squeeze()
            pred[:, :, fixing_slice] = new_pred[:, :, fixing_slice]
            pred_bin = postprocess_prediction(pred)
            #print(f"DICE after updating slice {fixing_slice}: {dice_coefficient_np(truth, pred_bin)}")
            validated_ind = fixing_slice
            inflated_truth[0][:, :, validated_ind] = pred_bin[:, :, fixing_slice]
            starting_slice = get_starting_slice_ind(validated_ind, config["prev_truth_index"])
    print(f"DICE after updating slice {fixing_slice}: {dice_coefficient_np(truth, pred_bin)}")
    return pred, truth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_folder", help="specifies config folder path",
                        type=str, required=False, default=r"/datadrive/configs")
    parser.add_argument("--data_folder", help="where is the relevant data",
                        type=str, required=False, default=r"/home/Shai/placenta_data")
    parser.add_argument("--overlap_factor", help="specifies adjacent patches overlap",
                        type=float, required=False, default=0.8)
    parser.add_argument("--propagate", help="specifies whether to propagate the correction",
                        default=False, action='store_true')
    parser.add_argument("--model_folder", help="specifies what model to load, must be relevant to working with truth",
                        type=str, required=True)
    parser.add_argument("--subject_id", help="specifies which scan to go over",
                        type=str, required=True)
    parser.add_argument("--validated_ind", help="specifies what slice is validated",
                        type=int, required=False, default=-1)
    parser.add_argument("--starting_pred_path", help="specifies adjacent patches overlap",
                        type=str, required=True)
    opts = parser.parse_args()
    
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    sess = tf.Session(config=config_gpu)
    set_session(sess)
    
    orig_pred = nib.load(opts.pred_path).get_data()
    orig_pred_bin = postprocess_prediction(orig_pred)
    
    pred_1, truth = main(opts.configs_folder, opts.data_folder, opts.model_folder, opts.subject_id, opts.validated_ind,
         opts.overlap_factor, opts.propagate, opts.starting_pred_path)
    print(f"DICE between new pred and original: {dice_coefficient_np(orig_pred_bin, postprocess_prediction(pred_1))}")
    pred_2, _ = main(opts.configs_folder, opts.data_folder, opts.model_folder, opts.subject_id, opts.validated_ind,
         opts.overlap_factor, opts.propagate, opts.starting_pred_path, flip=True)
    pred_2 = pred_2[:,:,::-1]
    print(f"DICE between new pred and original: {dice_coefficient_np(orig_pred_bin, postprocess_prediction(pred_2))}")
    n_slices = truth.shape[-1]
    
    # Option 1 - concatenate begining in both #
    pred = np.concatenate([pred_1[:,:,:int(np.ceil(n_slices/2))], pred_2[:,:,int(np.ceil(n_slices/2)):]], -1)
    pred_bin = postprocess_prediction(pred)
    print(f"DICE after concatenating both: {dice_coefficient_np(truth, pred_bin)}")
    
    # Option 2 - average #
    pred = (pred_1 + pred_2) / 2
    pred_bin = postprocess_prediction(pred)
    print(f"DICE after averaging both: {dice_coefficient_np(truth, pred_bin)}")
