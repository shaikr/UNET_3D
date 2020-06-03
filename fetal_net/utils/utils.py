import pickle
import os
import collections

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like

from .nilearn_custom_utils.nilearn_utils import crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset

from scipy.ndimage import map_coordinates
from scipy import ndimage 

import re


def transpose_if_needed(image_as_np):
    x,y,z = image_as_np.shape
    if x != y:
        if (x == z):
            image_as_np = image_as_np.transpose([2,0,1])
        elif (y == z):
            image_as_np = image_as_np.transpose([1,2,0])
        else:
            print("Problem")
        print('before_reshape: (x=%f, y=%f, z=%f)' % (x,y,z))
    return image_as_np


def scale_data(data, subject_id, dict_pkl, scale_xy=None):
    
    # TODO: handle case subject_id not existent / not in dict
    print("Data shape: {}".format(data.shape))
    if scale_xy:
        pat = "(.+)\*(.+)\*"
        res_dict = pickle.load(open(dict_pkl, 'rb'))
        res = res_dict[subject_id]
        match_obj_cur = re.match(pat, res)
        cur_x = float(res[match_obj_cur.span(1)[0]:match_obj_cur.span(1)[1]])
        cur_y = float(res[match_obj_cur.span(2)[0]:match_obj_cur.span(2)[1]])
        print("Current resolution: {}, {}".format(cur_x, cur_y))
        # TODO: export string literal to variable
        tgt_res = res_dict['main']
        match_obj_tgt = re.match(pat, tgt_res)
        tgt_x = float(tgt_res[match_obj_tgt.span(1)[0]:match_obj_tgt.span(1)[1]])
        tgt_y = float(tgt_res[match_obj_tgt.span(2)[0]:match_obj_tgt.span(2)[1]])
        x_scale = cur_x / tgt_x
        y_scale = cur_y / tgt_y
        print("Scaling by {}, {}".format(x_scale, y_scale))
    else:
        x_scale = 1
        y_scale = 1
    # TODO: handle z-axis scale better, and the deciphering of dimensions
    if data.ndim == 3:
        data = ndimage.zoom(data, [x_scale, y_scale, 1])
    elif data.ndim == 4 and data.shape[0] == 1:
        # Assumes dim 0 is 1, for batch
        data = ndimage.zoom(data, [1, x_scale, y_scale, 1])
    elif data.ndim == 4 and data.shape[-1] == 1:
        # Assumes last dim is 1, for batch
        data = ndimage.zoom(data, [x_scale, y_scale, 1, 1])
    else:
        print("Data shape is {}, not scaling!!!!!!".format(data.shape))
    return data 


def get_image(data, affine=None, nib_class=nib.Nifti1Image):
    if affine is None:
        affine = np.eye(4)
    return nib_class(dataobj=data, affine=affine)


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """
    
    :param image_files: 
    :param image_shape: 
    :param crop: 
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return: 
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return image_list


def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def read_img(in_file):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)


def interpolate_affine_coords(data, affine, coords, mode='constant', order=0, cval=0):
    in_vox_coords = np.array(np.meshgrid(*coords, indexing='ij'))
    coords_last = in_vox_coords.transpose(1, 2, 3, 0)
    mean_vox_coords = nib.affines.apply_affine(affine, coords_last)
    coords_first_again = mean_vox_coords.transpose(3, 0, 1, 2)
    resampled_mean_again = map_coordinates(data,
                                           coords_first_again,
                                           mode=mode, order=order, cval=cval)
    return resampled_mean_again


def interpolate_affine_range(data, affine, ranges, mode='constant', order=0, cval=0):
    return interpolate_affine_coords(data, affine, coords=[range(s, e) for s, e in ranges],
                                     mode=mode, order=order, cval=cval)
