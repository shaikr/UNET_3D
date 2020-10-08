import os

import numpy as np
import tables

from fetal_net.utils.utils import read_img, transpose_if_needed
from .normalize import normalize_data_storage, normalize_data_storage_each, normalize_data_storage_each_clip_and_norm, \
    normalize_data_storage_each_just_stretch, normalize_data_storage_each_stretch_and_norm, \
    normalize_data_storage_each_minmax, normalize_data_storage_each_just_clip


def create_data_file(out_file, add_pred, n_samples):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_storage = hdf5_file.create_vlarray(hdf5_file.root, 'data', tables.ObjectAtom(), filters=filters,
                                            expectedrows=n_samples)
    truth_storage = hdf5_file.create_vlarray(hdf5_file.root, 'truth', tables.ObjectAtom(), filters=filters,
                                             expectedrows=n_samples)
    if add_pred is not None:
        pred_storage = hdf5_file.create_vlarray(hdf5_file.root, 'pred', tables.ObjectAtom(), filters=filters,
                                                expectedrows=n_samples)
        return hdf5_file, data_storage, truth_storage, pred_storage
    else:
        return hdf5_file, data_storage, truth_storage, None


def write_image_data_to_file(image_files, data_storage, truth_storage, pred_storage, add_pred, truth_dtype=np.uint8):
    # TODO add scaling option as well
    for set_of_files in image_files:
        images = [read_img(_) for _ in set_of_files]
        subject_data = [transpose_if_needed(image.get_data()) for image in images]
        add_data_to_storage(data_storage, truth_storage, pred_storage, add_pred, subject_data, truth_dtype)
    return data_storage, truth_storage


def add_data_to_storage(data_storage, truth_storage, pred_storage, add_pred, subject_data, truth_dtype):
    # Can range from 1-3
    # TODO - add parameter to control which modalities are given here?
    # TODO - currently infers 2 modalities according to add_pred
    n_modalities = len(subject_data)
    
    data_ind = 0    
    if n_modalities == 2 and add_pred:
        pred_ind = 1
        truth_ind = -1
    elif n_modalities == 2 and not add_pred:
        pred_ind = -1
        truth_ind = 1 
    elif n_modalities == 3 and add_pred:
        pred_ind = 1
        truth_ind = 2
    elif n_modalities == 1:
        pred_ind = -1
        truth_ind = -1
    else:
        raise ValueError(f"Amount of modalities is {n_modalities}, add_pred is {add_pred}. What should happen?")
    
    data_storage.append(np.asarray(subject_data[data_ind]).astype(np.float))
    if pred_ind > 0:
        pred_storage.append(np.asarray(subject_data[pred_ind]).astype(np.float))
    if truth_ind > 0:
        truth_storage.append(np.asarray(subject_data[truth_ind], dtype=truth_dtype))


def write_data_to_file(training_data_files, out_file, truth_dtype=np.uint8,
                       subject_ids=None, normalize='all', add_pred=None):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image. 
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param truth_dtype: Default is 8-bit unsigned integer.
    :return: Location of the hdf5 file with the image data written to it. 
    """
    n_samples = len(training_data_files)
    try:
        hdf5_file, data_storage, truth_storage, pred_storage = create_data_file(out_file, add_pred, n_samples=n_samples)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(training_data_files, data_storage, truth_storage, pred_storage, add_pred,
                             truth_dtype=truth_dtype)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if isinstance(normalize, str):
        _, mean, std = {
            'all': normalize_data_storage,
            'each': normalize_data_storage_each,
            'each_stretch': normalize_data_storage_each_just_stretch,
            'each_stretch_and': normalize_data_storage_each_stretch_and_norm,
            'each_clip': normalize_data_storage_each_just_clip,
            'each_clip_and': normalize_data_storage_each_clip_and_norm,
            'each_minmax': normalize_data_storage_each_minmax,
        }[normalize](data_storage)
    else:
        mean, std = None, None
    hdf5_file.close()
    return out_file, (mean, std)


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
