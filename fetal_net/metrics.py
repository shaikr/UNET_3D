from functools import partial
from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
from enum import Enum


class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = \
        range(5)


def false_positive_rate(y_true, y_pred, smooth=1.):
    y_true_f = (1 - y_true).flatten()
    y_pred_f = y_pred.flatten()
    false_positives = np.sum(y_true_f * y_pred_f)
    return false_positives / (np.sum(y_pred_f) + smooth)


def false_negative_rate(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = (1 - y_pred).flatten()
    false_negatives = np.sum(y_true_f * y_pred_f)
    return false_negatives / (np.sum(y_pred_f) + smooth)


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_np(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def vod_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def vod_coefficient_np(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def get_surface_distances(truth, pred_binary):
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    statistics_image_filter = sitk.StatisticsImageFilter()
    pred_bin_0 = sitk.GetImageFromArray(pred_binary)
    truth = sitk.GetImageFromArray(truth)
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False))
    reference_surface = sitk.LabelContour(truth)
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_bin_0, squaredDistance=False))
    segmented_surface = sitk.LabelContour(pred_bin_0)
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    if num_segmented_surface_pixels == 0:
        return 100

    hausdorff_distance_filter.Execute(truth, pred_bin_0)

    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
    all_surface_distances = seg2ref_distances + ref2seg_distances
    return hausdorff_distance_filter.GetHausdorffDistance(), np.mean(all_surface_distances)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def vod_coefficient_loss(y_true, y_pred):
    return -vod_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth / 2) / (K.sum(y_true,
                                                                axis=axis) + K.sum(y_pred,
                                                                                   axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[..., label_index], y_pred[..., label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def dice_and_xent(y_true, y_pred, xent_weight=1.0, weight_mask=None):
    return dice_coef_loss(y_true, y_pred) + \
           xent_weight * weighted_cross_entropy_loss(y_true, y_pred, weight_mask)


def weighted_cross_entropy_loss(y_true, y_pred, weight_mask=None):
    xent = K.binary_crossentropy(y_true, y_pred)
    if weight_mask is not None:
        xent = K.prod(weight_mask, xent)
    return K.mean(xent)


def _focal_loss(gamma=2., alpha=.5):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
binary_crossentropy_loss = binary_crossentropy
focal_loss = _focal_loss()