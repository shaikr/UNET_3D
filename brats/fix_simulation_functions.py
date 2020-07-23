import numpy as np
from tqdm import tqdm
from notebooks.notebook_utils import vod_coefficient, dice_coefficient, accuracy_score, get_contour_mask


### Functions that determine slices order of fixing
# Can return the full order (like baseline_func, baseline_seq_func) or return the next slice to fix (like best_func2)
# Arguments:
# prediction - uint8 prediction 3d matrix
# truth - uint8 ground-truth 3d matrix
# metric_fcn - function by which to rank slices
# bool_arr - boolean array the length of #slices, 1 if slice was already fixed, else 0
# prediction_var - used for functions that need some side data (variance, other predictions...)
# use_contour_roi - if true, rank slices by metrics applied to area around the contour only


# decide on slices fix order randomly
def baseline_func(prediction, truth, metric_fcn, bool_arr=None, prediction_var=None, use_contour_roi=False):
    return np.random.permutation(np.arange(prediction.shape[-1]))


# fix slices in sequential order, starting with first non-blank slice
def baseline_seq_func(prediction, truth, metric_fcn, bool_arr=None, prediction_var=None, use_contour_roi=False):
    # Assumes binary prediction
    A_sums = np.sum(np.sum(prediction, axis=0), axis=0)
    non_blank_inds = [i for i in range(prediction.shape[-1]) if A_sums[i] > 0]
    starting_ind = non_blank_inds[0]
    return np.roll(np.arange(prediction.shape[-1]), -1 * starting_ind)


# fix slices in actual optimal solution - determine next slice by greatest VO improvement with GT
# NOTE: DO NOT USE, very high run-time
def best_func(prediction, truth, metric_fcn, bool_arr=None, prediction_var=None):
    cur_metric = metric_fcn(truth, prediction)
    best_metric = cur_metric
    best_slice = -1
    if bool_arr is None:
        remaining_inds = [i for i in range(prediction.shape[-1])]
    else:
        remaining_inds = [i for i in range(prediction.shape[-1]) if not bool_arr[i]]
    for i in remaining_inds:
        pred_tmp = prediction.copy()
        pred_tmp[:, :, i] = truth[:, :, i]
        new_val = metric_fcn(truth, pred_tmp)
        if is_better(best_metric, new_val, metric_fcn):
            best_slice = i
            best_metric = new_val
    return best_slice


# fix slices in some 2d קירוב of optimal solution - determine next slice by its' VO with GT
def best_func2(prediction, truth, metric_fcn, bool_arr=None, prediction_var=None, use_contour_roi=False):
    if bool_arr is None:
        remaining_inds = [i for i in range(prediction.shape[-1])]
    else:
        remaining_inds = [i for i in range(prediction.shape[-1]) if not bool_arr[i]]
    bool_mask = np.array(np.ones_like(prediction), dtype=np.bool, copy=True)
    if use_contour_roi:
        for i in range(bool_mask.shape[-1]):
            bool_mask[:, :, i] = get_contour_mask(prediction[:, :, i])
    vals = [metric_fcn(truth[:, :, j][bool_mask[:, :, j]], prediction[:, :, j][bool_mask[:, :, j]]) for j in
            remaining_inds]
    best_slice = remaining_inds[np.asarray(vals).argmin()]  # Note - assumes lowest is worst (VO, DICE)
    return best_slice


# fix by some estimation of the slice's VO - depends on specific folder structure my experiments produced
# Note - actually *needs* the prediction_var
def by_estimated_vo(prediction, truth, metric_fcn, bool_arr=None, prediction_var=None, use_contour_roi=False):
    # Must have prediction_var - list of multiple predictions to use for estimation

    # here for compatability - currently, runs one-time
    if bool_arr is None:
        remaining_inds = [i for i in range(prediction.shape[-1])]
    else:
        remaining_inds = [i for i in range(prediction.shape[-1]) if not bool_arr[i]]

    bool_mask = np.array(np.ones_like(prediction), dtype=np.bool, copy=True)
    if use_contour_roi:
        for i in range(bool_mask.shape[-1]):
            bool_mask[:, :, i] = get_contour_mask(prediction[:, :, i])

    slices_mets_est = [0] * len(remaining_inds)

    for i in range(len(remaining_inds)):
        cur_slice = remaining_inds[i]
        curr_mets_est = [metric_fcn(prediction_var[j][:, :, cur_slice][bool_mask[:, :, cur_slice]],
                                    prediction[:, :, cur_slice][bool_mask[:, :, cur_slice]]) for j in
                         range(len(prediction_var))]
        slices_mets_est[i] = np.mean(curr_mets_est)
    order_of_slices = np.array(remaining_inds)[np.argsort(slices_mets_est)]
    return order_of_slices


# So I can use the same comparison with all metrics
def is_better(val1, val2, metric_func):
    if (metric_func == dice_coefficient) | (metric_func == vod_coefficient) | (metric_func == accuracy_score):
        if val2 >= val1:
            return True
        else:
            return False
    else:
        if val2 > val1:
            return False
        else:
            return True


# run the process of fixing a specific scan according to a specific policy
def get_scan_progress(prediction, truth, next_slice_fcn, metric_fcn=dice_coefficient,
                      prediction_var=None, use_contour_roi=False):
    n_slices = prediction.shape[-1]
    y_values = [0] * (n_slices + 1)  # y_values - depending on the metrics
    prediction_copy = prediction.copy()
    was_slice_done = [False] * n_slices

    # currently the only function that really works iteratively
    if next_slice_fcn == best_func2:
        for i in tqdm(range(n_slices)):
            y_values[i] = metric_fcn(truth, prediction_copy)
            next_slice_ind = next_slice_fcn(prediction, truth, metric_fcn, bool_arr=was_slice_done,
                                            prediction_var=prediction_var, use_contour_roi=use_contour_roi)
            prediction_copy[:, :, next_slice_ind] = truth[:, :, next_slice_ind]
            was_slice_done[next_slice_ind] = True
    else:
        slices_order = next_slice_fcn(prediction, truth, metric_fcn, bool_arr=was_slice_done,
                                      prediction_var=prediction_var, use_contour_roi=use_contour_roi)
        for i in tqdm(range(n_slices)):
            y_values[i] = metric_fcn(truth, prediction_copy)
            next_slice_ind = slices_order[i]
            prediction_copy[:, :, next_slice_ind] = truth[:, :, next_slice_ind]
    y_values[-1] = metric_fcn(truth, prediction_copy)
    return y_values


# Go from different amount of slices per scan to 34 datapoints (so can compare different scans on same graph)
def reduce_progress_for_graph(y_values):
    x_vals = [i for j in (range(0, 55, 3), range(58, 101, 3)) for i in j]
    n_init_pts = len(y_values)
    n_datapoints = 34
    final_vals = [0] * n_datapoints
    final_vals[0] = y_values[0]

    # I.E. on 90 slices: 0, 2, ..., 48, 51, 54, ...,86, 89
    # I.E. on 70 slices: 0, 1, 3, 37, 40, 42, ..., 67, 69
    gen = (i for j in (range(3, 55, 3), range(58, 101, 3)) for i in j)
    new_ind = 1
    for x in gen:
        cur_ind = int(round(n_init_pts * x / 100) - 1)
        final_vals[new_ind] = y_values[cur_ind]
        new_ind += 1
    assert len(x_vals) == len(final_vals), "lengths do not match"
    return x_vals, final_vals

