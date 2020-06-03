import os

import numpy as np
from nilearn.image import new_img_like

from fetal_net.utils.utils import resize, read_image_files
from .utils import crop_img, crop_img_to, read_image


def auto_contrast_enhancement(img_as_np_orig, max_cutoff=0.999):
    # get the new maximum level
    img_max = np.max(img_as_np_orig.flatten())
    img_min = np.min(img_as_np_orig.flatten()) # Assumes this will be 0 (?)
    img_half = 0.5 * img_max
    ww = np.quantile(img_as_np_orig, max_cutoff)
    wl = ww / 2.0
    out_range = [img_min, ww]
    img_as_np = img_as_np_orig.copy()

    img_as_np[img_as_np <= (wl - ww / 2.0)] = out_range[0]
    # img_as_np[(img_as_np > (wl - ww / 2.0)) & (img_as_np <= (wl + ww / 2.0))] = \
    #     ((img_as_np[(img_as_np > (wl - ww / 2.0)) & (img_as_np <= (wl + ww / 2.0))]
    #       - (wl - img_half)) / (ww - img_max) + img_half) * (out_range[1] - out_range[0]) + out_range[0]
    img_as_np[img_as_np > (wl + ww / 2.0)] = out_range[1]
    return img_as_np


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters(in_files):
    if len(in_files) > 1:
        foreground = get_complete_foreground(in_files)
    else:
        foreground = get_foreground_from_set_of_files(in_files[0], return_image=True)
    return crop_img(foreground, return_slices=True, copy=True)


def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False):
    if crop:
        crop_slices = get_cropping_parameters([in_files])
    else:
        crop_slices = None
    images = read_image_files(in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices)
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images


def get_complete_foreground(training_data_files):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001, return_image=False):
    for i, image_file in enumerate(set_of_files):
        image = read_image(image_file)
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1
    if return_image:
        return new_img_like(image, foreground)
    else:
        return foreground


def normalize_data(data, mean, std):
    data -= mean
    data /= std
    return data


def minmax_norm(data):
    min_ = data.min()
    max_ = data.norm()
    return (data - min_) / (max_ - min_)


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(-1, -2, -3)))
        stds.append(data.std(axis=(-1, -2, -3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage, mean, std


def normalize_data_storage_each(data_storage):
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        mean = data.mean(axis=(-1, -2, -3))
        std = data.std(axis=(-1, -2, -3))
        data_storage[index] = normalize_data(data, mean, std)
    return data_storage, None, None


def normalize_data_storage_each_minmax(data_storage):
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        min_ = data.min()
        max_ = data.max()
        data_storage[index] = (data - min_) / (max_ - min_)
    return data_storage, None, None


def normalize_data_storage_each_just_stretch(data_storage):
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        # stretch histogram
        data_max = data.max()
        data_quan99 = np.quantile(data, 0.999)
        ratio = data_max / data_quan99
        data = np.clip(data, data.min(), data_quan99)
        data = (data * ratio).round()
        data_storage[index] = data
    return data_storage, None, None


def normalize_data_storage_each_stretch_and_norm(data_storage):
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        # stretch histogram
        data_max = data.max()
        data_quan99 = np.quantile(data, 0.999)
        ratio = data_max / data_quan99
        data = np.clip(data, data.min(), data_quan99)
        data = (data * ratio).round()
        mean = data.mean(axis=(-1, -2, -3))
        std = data.std(axis=(-1, -2, -3))
        data_storage[index] = normalize_data(data, mean, std)
    return data_storage, None, None


def normalize_data_storage_each_clip_and_norm(data_storage):
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        # clip histogram
        data_quan99 = np.quantile(data, 0.999)
        data = np.clip(data, data.min(), data_quan99)
        mean = data.mean(axis=(-1, -2, -3))
        std = data.std(axis=(-1, -2, -3))
        data_storage[index] = normalize_data(data, mean, std)
    return data_storage, None, None


def normalize_data_storage_each_just_clip(data_storage):
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        # clip histogram
        data_quan99 = np.quantile(data, 0.999)
        data = np.clip(data, data.min(), data_quan99)
        data_storage[index] = data
    return data_storage, None, None
