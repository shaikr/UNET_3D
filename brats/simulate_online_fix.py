### This file is meant to simulate online segmentation handling when context is taken into consideration in network. ###
import itertools
import os

import nibabel as nib
import numpy as np
import tables
from keras import Model
from tqdm import tqdm
from scipy import ndimage

from brats.utils import get_last_model_path
from fetal_net.utils.threaded_generator import ThreadedGenerator
from fetal_net.utils.utils import get_image, resize
from fetal_net.training import load_old_model
from fetal_net.prediction import predict_augment, patch_wise_prediction
from notebooks.notebook_utils import *
import json
from fetal_net.utils import pickle_load

# Step 1: get several networks' prediction for a specific scan, based on the scan alone (several - for conf. est.)
# Step 2: get a different network's prediction for that scan, based on the scan + previous prediction
# (measure current DICE, VOD, etc.)
# Step 3: fix a single slice in the prediction so that it is correct
# Step 4: refresh the prediction using the network from stage 2
# (measure current DICE, VOD, etc.)
# Step 5: iterate steps 3-4 until full fix

step_1_network_experiments_paths = ['/datadrive/configs/...', '/datadrive/configs/...', '/datadrive/configs/...']
step_2_network_experiments_paths = ['/datadrive/configs/...']
output_prediction_dir = r""
if not os.path.exists(output_prediction_dir):
    os.mkdir(output_prediction_dir)
#subject_ids = ['40']
subject_index = []
overlap_factor = 0.9

hdf5_file = r""
data_file = tables.open_file(hdf5_file, "a")
filters = tables.Filters(complevel=5, complib='blosc')
pred_storage = data_file.create_vlarray(data_file.root, 'pred', tables.ObjectAtom(), filters=filters,
                                        expectedrows=len(subject_index)) # TODO: needs to be same length as other arrays

step_1_networks = [load_old_model(get_last_model_path(os.path.join(exp_folder, "fetal_net_model")))
                   for exp_folder in step_1_network_experiments_paths]

step_1_configs = []
n_step_1_models = len(step_1_networks)
for i in range(n_step_1_models):
    with open(os.path.join(step_1_network_experiments_paths[i], 'config.json')) as f:
        config = json.load(f)
        step_1_configs.append(config)

step_2_network = [load_old_model(get_last_model_path(os.path.join(exp_folder, "fetal_net_model")))
                   for exp_folder in step_2_network_experiments_paths]
with open(os.path.join(step_2_network_experiments_paths[0], 'config.json')) as f:
    step_2_config = json.load(f)

########################### Step 1 ###########################
print("################## Step 1 ####################")
for sid in subject_index:
    subject_id = data_file.root.subject_ids[sid]
    print("In ID {}".format(subject_id))
    os.mkdir(os.path.join(output_prediction_dir, subject_id))
    test_data = np.asarray([data_file.root.data[sid]])
    test_truth_data = np.asarray([data_file.root.truth[sid]])

    # Get all predictions
    for i, model in enumerate(step_1_networks):
        config = step_1_configs[i]
        # step 1 - no use of context for prediction
        if config["use_augmentations"]: # TODO - add this key to configs, default False
            prediction = predict_augment(data=test_data, model=model, overlap_factor=overlap_factor,
                                         patch_shape=config["patch_shape"])
        else:
            prediction, _ = \
                patch_wise_prediction(model=model, data=test_data, overlap_factor=overlap_factor,
                                      patch_shape=config["patch_shape"], permute=config["augment"]["permute"])
        prediction = prediction.squeeze()
        prediction_image = get_image(prediction) # NIB format
        filename = os.path.join(output_prediction_dir, subject_id, "prediction_{}.nii.gz".format(i))
        prediction_image.to_filename(filename)

        if i > 0:
            avg_pred += prediction
        else:
            avg_pred = prediction

    # Get the final averaged prediction and write to file
    avg_pred /= n_step_1_models
    nib.save(nib.Nifti1Image(avg_pred, prediction_image.affine, header=prediction_image.header),
             os.path.join(output_prediction_dir, subject_id, f'averaged_prediction.nii'))
    pred_storage[sid] = np.asarray(avg_pred).astype(np.float)

    bin_avg_pred = adapted_postprocess_pred(os.path.join(output_prediction_dir, subject_id, f'averaged_prediction.nii'))
    cur_sc = dice_coefficient(test_truth_data, bin_avg_pred) * 100
    print("Using averaged prediction of step 1 - {} DICE".format(cur_sc))

########################### Step 2 ###########################
print("################## Step 2 ####################")


def get_prediction(sid, model, config, slices_range=None):
    # Assumes the data_file has been updated
    test_data = np.asarray([data_file.root.data[sid]])
    test_truth_data = np.asarray([data_file.root.truth[sid]])
    test_pred_data = np.asarray([data_file.root.pred[sid]])
    if config["use_augmentations"]:  # TODO - add this key to configs, default False
        prediction = predict_augment(data=test_data, model=model, overlap_factor=overlap_factor,
                                     patch_shape=config["patch_shape"])
    else:
        prediction, _ = \
            patch_wise_prediction(model=model, data=test_data, overlap_factor=overlap_factor,
                                  patch_shape=config["patch_shape"], permute=config["augment"]["permute"],
                                  truth_data=test_truth_data, prev_truth_index=config["prev_truth_index"],
                                  prev_truth_size=config["prev_truth_size"],
                                  pred_data=test_pred_data, pred_index=config["pred_index"],
                                  pred_size=config["pred_size"],
                                  slices_range=slices_range
                                  )
    prediction = prediction.squeeze()
    return prediction


# only one model here
config = step_2_config
model = step_2_network[0]
for sid in subject_index:
    subject_id = data_file.root.subject_ids[sid]
    print("In ID {}".format(subject_id))

    prediction = get_prediction(sid, model, config)
    prediction_image = get_image(prediction)  # NIB format
    filename = os.path.join(output_prediction_dir, subject_id, "step_2_prediction.nii.gz")
    prediction_image.to_filename(filename)
    bin_avg_pred = adapted_postprocess_pred(os.path.join(output_prediction_dir, subject_id, f'step_2_prediction.nii.gz'))
    cur_sc = dice_coefficient(test_truth_data, bin_avg_pred) * 100
    print("Using averaged prediction of step 2 - {} DICE".format(cur_sc))
    # TomLovesShai

########################### Step 3 ###########################
print("################## Step 3-5 ####################")

# Note - excruciating process. After every slice fix, we have to predict again,
# and after every order we try we need to reset the prediction of the subject id for the next test
# Note - we need to not change slices we already "fixed"


def random_baseline_order(bin_prediction):
    return np.random.permutation(np.arange(bin_prediction.shape[-1]))


def next_slice_to_correct_best(bin_prediction, truth, remaining_inds, metric_fcn):
    values = [metric_fcn(truth[:, :, j], bin_prediction[:, :, j]) for j in remaining_inds]
    worst_slice = remaining_inds[np.asarray(values).argmin()]  # Note - assumes lowest is worst (VOD, DICE)
    return worst_slice


metric_fcn = vod_coefficient
# TODO: this is probably excruciatingly long. Need to narrow down the prediction updates
for sid in subject_index:
    subject_id = data_file.root.subject_ids[sid]
    truth = np.asarray([data_file.root.truth[sid]])
    print("In ID: {}".format(subject_id))

    print("-----------------Simulating best possible fixing process-----------------")
    cur_pred = nib.load(os.path.join(output_prediction_dir, subject_id, "step_2_prediction.nii.gz")).get_data()
    cur_bin_pred = nib.load(os.path.join(output_prediction_dir, subject_id, "binary_step_2_prediction.nii.gz")).get_data()
    data_file.root.pred[sid] = np.asarray(cur_pred).astype(np.float)
    remaining_inds = np.arange(cur_pred.shape[-1])
    progress_values_best = [0] * (cur_pred.shape[-1]+1)
    progress_values_best[0] = metric_fcn(truth, cur_bin_pred)

    for t in range(cur_pred.shape[-1]):
        # whats the currently worst slice not fixed
        next_slice_to_fix = next_slice_to_correct_best(cur_bin_pred, truth, remaining_inds, metric_fcn)
        # fix it
        cur_pred[:, :, next_slice_to_fix] = truth[:, :, next_slice_to_fix]
        remaining_inds.remove(next_slice_to_fix)
        data_file.root.pred[sid] = np.asarray(cur_pred).astype(np.float)
        # get new prediction - TODO: set the slices range, and - should the correction propagate?
        new_pred = get_prediction(sid, model, config)
        # update only the untouched slices so far
        for s in remaining_inds:
            cur_pred[:, :, s] = new_pred[:, :, s]
        cur_bin_pred = adapted_postprocess_pred(cur_pred, to_save=False)
        progress_values_best[t+1] = metric_fcn(truth, cur_bin_pred)

    # print("-----------------Simulating fixing process based on cetrainty estimation -----------------")
    # cur_pred = nib.load(os.path.join(output_prediction_dir, subject_id, "step_2_prediction.nii.gz")).get_data()
    # cur_bin_pred = nib.load(
    #     os.path.join(output_prediction_dir, subject_id, "binary_step_2_prediction.nii.gz")).get_data()
    # data_file.root.pred[sid] = np.asarray(cur_pred).astype(np.float)
    # remaining_inds = np.arange(cur_pred.shape[-1])
    # progress_values_best = [0] * (cur_pred.shape[-1] + 1)
    # progress_values_best[0] = metric_fcn(truth, cur_bin_pred)
    #
    # for t in range(cur_pred.shape[-1]):
    #     # whats the currently worst slice not fixed
    #     next_slice_to_fix = next_slice_to_correct_est(cur_bin_pred, truth, remaining_inds, metric_fcn)
    #     # fix it
    #     cur_pred[:, :, next_slice_to_fix] = truth[:, :, next_slice_to_fix]
    #     remaining_inds.remove(next_slice_to_fix)
    #     data_file.root.pred[sid] = np.asarray(cur_pred).astype(np.float)
    #     # get new prediction
    #     new_pred = get_prediction(sid, model, config)
    #     # update only the untouched slices so far
    #     for s in remaining_inds:
    #         cur_pred[:, :, s] = new_pred[:, :, s]
    #     cur_bin_pred = adapted_postprocess_pred(cur_pred, to_save=False)
    #     progress_values_best[t + 1] = metric_fcn(truth, cur_bin_pred)

data_file.close()
