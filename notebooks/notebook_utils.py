import nibabel as nib
import os
import numpy as np
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
import skimage.morphology as morphology
import SimpleITK as sitk
from enum import Enum
# Use enumerations to represent the various evaluation measures


# Get mask of area around a predicted slice's contour
def get_contour_mask(binarized_pred_slice, rad=3):
    elem = np.ones(shape=(2*rad+1, 2*rad+1))
    dilated = morphology.binary_dilation(binarized_pred_slice, elem)
    eroded = morphology.binary_erosion(binarized_pred_slice, elem)
    return dilated & ~eroded


def accuracy_score(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return sum(y_true_f == y_pred_f) / len(y_pred_f)


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def vod_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def get_main_connected_component(data):
    labeled_array, num_features = label(data)
    if num_features > 0:
        i = np.argmax([np.sum(labeled_array == _) for _ in range(1, num_features + 1)]) + 1
    else:
        i = 1
    return labeled_array == i


def postprocess_prediction(pred, gaussian_std=1, threshold=0.5, fill_holes=True, connected_component=True):
    pred = gaussian_filter(pred, gaussian_std) > threshold
    if fill_holes:
        pred = binary_fill_holes(pred)
    if connected_component:
        pred = get_main_connected_component(pred)
    return pred


class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)


class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(5)


def adapted_postprocess_pred(prediction_or_path, to_save=True):
    if isinstance(prediction_or_path, str):
        pred = nib.load(prediction_or_path)
        pred_data = pred.get_data()
    else:
        pred_data = prediction_or_path
    pred_bin = postprocess_prediction(pred_data)
    pred_bin = nib.Nifti1Image(pred_bin, pred.affine, header=pred.header)
    if to_save and isinstance(prediction_or_path, str):
        pred_bin_path = os.path.join(os.path.dirname(prediction_or_path), 'binary_'+os.path.basename(prediction_or_path))
        nib.save(pred_bin, pred_bin_path)
    return pred_bin.get_data()


def get_volume_metrics(truth, pred_binary):
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    statistics_image_filter = sitk.StatisticsImageFilter()
    label = 1
    pred_bin_0 = sitk.GetImageFromArray(pred_binary)
    truth = sitk.GetImageFromArray(truth)
    surface_distance_results = np.zeros((len(SurfaceDistanceMeasures.__members__.items()),))
    overlap_results = np.zeros((len(OverlapMeasures.__members__.items()),))
#     print(truth.GetSize())
#     print(pred_bin_0.GetSize())
#     print(truth.GetPixelIDTypeAsString())
#     print(pred_bin_0.GetPixelIDTypeAsString())
    overlap_measures_filter.Execute(truth, pred_bin_0)
    overlap_results[OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
    overlap_results[OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
    overlap_results[OverlapMeasures.volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()
    overlap_results[OverlapMeasures.false_negative.value] = overlap_measures_filter.GetFalseNegativeError()
    overlap_results[OverlapMeasures.false_positive.value] = overlap_measures_filter.GetFalsePositiveError()
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False))
    reference_surface = sitk.LabelContour(truth)
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_bin_0, squaredDistance=False))
    segmented_surface = sitk.LabelContour(pred_bin_0)
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    if num_segmented_surface_pixels == 0:
        return [100, 100, 100, 100, 100], [100, 100, 100, 100, 100]
    hausdorff_distance_filter.Execute(truth, pred_bin_0)
    surface_distance_results[SurfaceDistanceMeasures.hausdorff_distance.value] = \
                        hausdorff_distance_filter.GetHausdorffDistance()
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0])
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0])
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
    all_surface_distances = seg2ref_distances + ref2seg_distances

    surface_distance_results[SurfaceDistanceMeasures.mean_surface_distance.value] = np.mean(all_surface_distances)
    surface_distance_results[SurfaceDistanceMeasures.median_surface_distance.value] = np.median(all_surface_distances)
    surface_distance_results[SurfaceDistanceMeasures.std_surface_distance.value] = np.std(all_surface_distances)
    surface_distance_results[SurfaceDistanceMeasures.max_surface_distance.value] = np.max(all_surface_distances)
    return overlap_results, surface_distance_results


def get_best_and_worst_slices(pred_dir, measure_func=dice_coefficient, top_k=1):
    truth = nib.load(os.path.join(pred_dir, 'truth.nii.gz')).get_data()
    pred_bin_0 = nib.load(os.path.join(pred_dir, 'prediction_binary_0.nii')).get_data()
    truth_sums = np.sum(np.sum(truth, axis=0), axis=0)

    non_blank_inds = [i for i in range(truth.shape[-1]) if truth_sums[i] > 0]
    blank_inds = [i for i in range(truth.shape[-1]) if truth_sums[i] == 0]

    measures = [measure_func(truth[:, :, i], pred_bin_0[:, :, i]) for i in range(truth.shape[-1])]

    measures_current = [measures[i] for i in non_blank_inds]
    cur_inds = np.argsort(measures_current)
    # the best and worst non-blank slices
    non_blank_best = [non_blank_inds[i] for i in cur_inds[-top_k:]]
    print(non_blank_best)
    non_blank_best = [(x, measures[x]) for x in non_blank_best]
    non_blank_worst = [non_blank_inds[i] for i in cur_inds[:top_k]]
    print(non_blank_worst)
    non_blank_worst = [(x, measures[x]) for x in non_blank_worst]

    measures_current = [measures[i] for i in blank_inds]
    cur_inds = np.argsort(measures_current)
    # the best and worst blank slices
    blank_best = [blank_inds[i] for i in cur_inds[-top_k:]]
    blank_best = [(x, measures[x]) for x in blank_best]
    blank_worst = [blank_inds[i] for i in cur_inds[:top_k]]
    blank_worst = [(x, measures[x]) for x in blank_worst]

    return non_blank_best, non_blank_worst, blank_best, blank_worst


def create_id_postprocess_prediction(pred_dir, cur_id, orig_data_folder):
    cur_pred_dir = os.path.join(pred_dir, cur_id)
    volume = nib.load(os.path.join(orig_data_folder, cur_id, 'volume.nii')).get_data()
    truth = nib.load(os.path.join(orig_data_folder, cur_id, 'truth.nii')).get_data()
    print(f'Volume of truth is: {np.sum(truth.flatten())}')
    pred = nib.load(os.path.join(cur_pred_dir, 'prediction.nii.gz'))
    pred_data = pred.get_data()
    if not os.path.exists(os.path.join(cur_pred_dir, 'prediction_binary_0.nii')):
        pred_bin_0 = postprocess_prediction(pred_data)
        pred_bin_0 = nib.Nifti1Image(pred_bin_0, pred.affine, header=pred.header)
        nib.save(pred_bin_0, os.path.join(cur_pred_dir, 'prediction_binary_0.nii'))
    else:
        pred_bin_0 = nib.load(os.path.join(cur_pred_dir, 'prediction_binary_0.nii'))
    pred_bin_data = pred_bin_0.get_data()

    # simple binary prediction
    pred_bin_1 = pred_data > 0.5
    pred_bin_1 = nib.Nifti1Image(pred_bin_1, pred.affine, header=pred.header)
    nib.save(pred_bin_1, os.path.join(cur_pred_dir, 'prediction_binary_simple.nii'))

    print(f'Volume of prediction is: {np.sum(pred_bin_data.flatten())}')
    print(f'overall vod in id {cur_id}: {vod_coefficient(truth, pred_bin_data)}')
    print(f'overall dice in id {cur_id}: {dice_coefficient(truth, pred_bin_data)}')


# Do what we need to obtain the multiple predictions
def get_all_preds(s_id, configs_folder, rel_experiments, group_name='test'):
    # group_name = 'train'/'val'/'test'
    # s_id - string of subject id
    predictions = []
    for exp in rel_experiments:
        pred_path = os.path.join(configs_folder, exp, 'predictions', group_name, s_id, 'prediction.nii.gz')
        pred = nib.load(pred_path).get_fdata()
        if pred.ndim == 4:
            for i in range(pred.shape[0]):
                predictions += [pred[i]]
        else:
            predictions += [pred]
    if len(predictions) == 1:
        predictions = predictions[0]
    else:
        predictions = np.stack(predictions, axis=0)
    return predictions.squeeze()
    