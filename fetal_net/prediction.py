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
from fetal_net.utils.utils import get_image, resize, scale_data
from .training import load_old_model
from .utils import pickle_load
from .utils.patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices, \
    get_set_of_patch_indices
from .augment import permute_data, generate_permutation_keys, reverse_permute_data, contrast_augment


def flip_it(data_, axes):
    for ax in axes:
        data_ = np.flip(data_, ax)
    return data_


def predict_augment(data, model, overlap_factor, patch_shape, num_augments=5, is3d=False):
    data_max = data.max()
    data_min = data.min()
    data = data.squeeze()

    order = 2
    cval = np.percentile(data, q=1)

    predictions = []
    for _ in range(num_augments):
        # pixel-wise augmentations
        val_range = data_max - data_min
        contrast_min_val = data_min + 0.10 * np.random.uniform(-1, 1) * val_range
        contrast_max_val = data_max + 0.10 * np.random.uniform(-1, 1) * val_range
        curr_data = contrast_augment(data, contrast_min_val, contrast_max_val)

        # spatial augmentations
        rotate_factor = np.random.uniform(-30, 30)
        to_flip = np.arange(0, 3)[np.random.choice([True, False], size=3)]
        to_transpose = np.random.choice([True, False])

        curr_data = flip_it(curr_data, to_flip)

        if to_transpose:
            curr_data = curr_data.transpose([1, 0, 2])

        curr_data = ndimage.rotate(curr_data, rotate_factor, order=order, reshape=False, mode='constant', cval=cval)

        curr_prediction, _ = patch_wise_prediction(model=model, data=curr_data[np.newaxis, ...],
                                                   overlap_factor=overlap_factor, patch_shape=patch_shape, is3d=is3d)

        curr_prediction = curr_prediction.squeeze()

        curr_prediction = ndimage.rotate(curr_prediction, -rotate_factor, order=0, reshape=False, mode='constant', cval=0)

        if to_transpose:
            curr_prediction = curr_prediction.transpose([1, 0, 2])

        curr_prediction = flip_it(curr_prediction, to_flip)
        predictions += [curr_prediction.squeeze()]

    res = np.stack(predictions, axis=0)
    return res


def predict_flips(data, model, overlap_factor, config):
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(0, len(s) + 1))

    def predict_it(data_, axes=()):
        data_ = flip_it(data_, axes)
        curr_pred = \
            patch_wise_prediction(model=model,
                                  data=np.expand_dims(data_.squeeze(), 0),
                                  overlap_factor=overlap_factor,
                                  patch_shape=config["patch_shape"] + [config["patch_depth"]], is3d=config["3D"]).squeeze()
        curr_pred = flip_it(curr_pred, axes)
        return curr_pred

    predictions = []
    for axes in powerset([0, 1, 2]):
        predictions += [predict_it(data, axes).squeeze()]

    return predictions


def get_set_of_patch_indices_full(start, stop, step):
    indices = []
    for start_i, stop_i, step_i in zip(start, stop, step):
        indices_i = list(range(start_i, stop_i + 1, step_i))
        if stop_i % step_i > 0:
            indices_i += [stop_i]
        indices += [indices_i]
    return np.array(list(itertools.product(*indices)))


def batch_iterator(indices, batch_size, data_0, patch_shape,
                   truth_0, prev_truth_index, truth_patch_shape,
                   pred_0, pred_index, pred_patch_shape):
    i = 0
    while i < len(indices):
        batch = []
        curr_indices = []
        while len(batch) < batch_size and i < len(indices):
            curr_index = indices[i]
            patch = get_patch_from_3d_data(data_0, patch_shape=patch_shape, patch_index=curr_index)

            if pred_0 is not None:
                cur_pred_index = list(curr_index[:2]) + [curr_index[2] + pred_index]
                pred_patch = get_patch_from_3d_data(pred_0, patch_shape=pred_patch_shape,
                                                     patch_index=cur_pred_index)
                patch = np.concatenate([patch, pred_patch], axis=-1)

            if truth_0 is not None:
                truth_index = list(curr_index[:2]) + [curr_index[2] + prev_truth_index]
                truth_patch = get_patch_from_3d_data(truth_0, patch_shape=truth_patch_shape,
                                                     patch_index=truth_index)
                patch = np.concatenate([patch, truth_patch], axis=-1)
            batch.append(patch)
            curr_indices.append(curr_index)
            i += 1
        yield [batch, curr_indices]
    # print('Finished! {}-{}'.format(i, len(indices)))


def patch_wise_prediction(model: Model, data, patch_shape, overlap_factor=0, batch_size=5, is3d=False,
                          permute=False, truth_data=None, prev_truth_index=None, prev_truth_size=None,
                          pred_data=None, pred_index=None, pred_size=None):
    """
    :param truth_data:
    :param permute:
    :param overlap_factor:
    :param batch_size:
    :param model:
    :param data:
    :return:
    """

    if is3d:
        prediction_shape = model.output_shape[-3:]
    else:
        prediction_shape = model.output_shape[-3:-1] + (1,)  # patch_shape[-3:-1] #[64,64]#
    min_overlap = np.subtract(patch_shape, prediction_shape)
    max_overlap = np.subtract(patch_shape, (1, 1, 1))
    overlap = min_overlap + (overlap_factor * (max_overlap - min_overlap)).astype(np.int)
    data_0 = np.pad(data[0],
                    [(np.ceil(_ / 2).astype(int), np.floor(_ / 2).astype(int)) for _ in
                     np.subtract(patch_shape, prediction_shape)],
                    mode='constant', constant_values=np.percentile(data[0], q=1))
    pad_for_fit = [(np.ceil(_ / 2).astype(int), np.floor(_ / 2).astype(int)) for _ in
                   np.maximum(np.subtract(patch_shape, data_0.shape), 0)]
    data_0 = np.pad(data_0,
                    [_ for _ in pad_for_fit],
                    'constant', constant_values=np.percentile(data_0, q=1))

    if pred_data is not None:
        pred_0 = np.pad(pred_data[0],
                         [(np.ceil(_ / 2).astype(int), np.floor(_ / 2).astype(int)) for _ in
                          np.subtract(patch_shape, prediction_shape)],
                         mode='constant', constant_values=0)
        pred_0 = np.pad(pred_0, [_ for _ in pad_for_fit],
                         'constant', constant_values=0)
        pred_patch_shape = list(patch_shape[:2]) + [pred_size]
    else:
        pred_0 = None
        pred_patch_shape = None

    if truth_data is not None:
        truth_0 = np.pad(truth_data[0],
                         [(np.ceil(_ / 2).astype(int), np.floor(_ / 2).astype(int)) for _ in
                          np.subtract(patch_shape, prediction_shape)],
                         mode='constant', constant_values=0)
        truth_0 = np.pad(truth_0, [_ for _ in pad_for_fit],
                         'constant', constant_values=0)

        truth_patch_shape = list(patch_shape[:2]) + [prev_truth_size]
    else:
        truth_0 = None
        truth_patch_shape = None

    indices = get_set_of_patch_indices_full((0, 0, 0),
                                            np.subtract(data_0.shape, patch_shape),
                                            np.subtract(patch_shape, overlap))

    b_iter = batch_iterator(indices, batch_size, data_0, patch_shape,
                            truth_0, prev_truth_index, truth_patch_shape,
                            pred_0, pred_index, pred_patch_shape)
    tb_iter = iter(ThreadedGenerator(b_iter, queue_maxsize=50))

    data_shape = list(data.shape[-3:] + np.sum(pad_for_fit, -1))
    if is3d:
        data_shape += [model.output_shape[1]]
    else:
        data_shape += [model.output_shape[-1]]
    predicted_output = np.zeros(data_shape)
    predicted_count = np.zeros(data_shape, dtype=np.int16)
    with tqdm(total=len(indices)) as pbar:
        for [curr_batch, batch_indices] in tb_iter:
            curr_batch = np.asarray(curr_batch)
            if is3d:
                curr_batch = np.expand_dims(curr_batch, 1)
            prediction = predict(model, curr_batch, permute=permute)

            if is3d:
                prediction = prediction.transpose([0, 2, 3, 4, 1])
            else:
                prediction = np.expand_dims(prediction, -2)

            for predicted_patch, predicted_index in zip(prediction, batch_indices):
                # predictions.append(predicted_patch)
                x, y, z = predicted_index
                x_len, y_len, z_len = predicted_patch.shape[:-1]
                predicted_output[x:x + x_len, y:y + y_len, z:z + z_len, :] += predicted_patch
                predicted_count[x:x + x_len, y:y + y_len, z:z + z_len] += 1
            pbar.update(batch_size)

    assert np.all(predicted_count > 0), 'Found zeros in count'

    if np.sum(pad_for_fit) > 0:
        # must be a better way :\
        x_pad, y_pad, z_pad = [[None if p2[0] == 0 else p2[0],
                                None if p2[1] == 0 else -p2[1]] for p2 in pad_for_fit]
        predicted_count = predicted_count[x_pad[0]: x_pad[1],
                          y_pad[0]: y_pad[1],
                          z_pad[0]: z_pad[1]]
        predicted_output = predicted_output[x_pad[0]: x_pad[1],
                           y_pad[0]: y_pad[1],
                           z_pad[0]: z_pad[1]]

    assert np.array_equal(predicted_count.shape[:-1], data[0].shape), 'prediction shape wrong'

    final_prediction = predicted_output / predicted_count
    prediction_variance = np.zeros(predicted_count.shape)
    # b_iter = batch_iterator(indices, batch_size, data_0, patch_shape,
    #                         truth_0, prev_truth_index, truth_patch_shape,
    #                         pred_0, pred_index, pred_patch_shape)
    # tb_iter = iter(ThreadedGenerator(b_iter, queue_maxsize=50))
    # with tqdm(total=len(indices)) as pbar:
    #     for [curr_batch, batch_indices] in tb_iter:
    #         curr_batch = np.asarray(curr_batch)
    #         if is3d:
    #             curr_batch = np.expand_dims(curr_batch, 1)
    #         prediction = predict(model, curr_batch, permute=permute)
    #
    #         if is3d:
    #             prediction = prediction.transpose([0, 2, 3, 4, 1])
    #         else:
    #             prediction = np.expand_dims(prediction, -2)
    #
    #         for predicted_patch, predicted_index in zip(prediction, batch_indices):
    #             # predictions.append(predicted_patch)
    #             x, y, z = predicted_index
    #             x_len, y_len, z_len = predicted_patch.shape[:-1]
    #             prediction_variance[x:x + x_len, y:y + y_len, z:z + z_len, :] += \
    #                 np.power(predicted_patch[:, :, :] - final_prediction[x:x + x_len, y:y + y_len, z:z + z_len], 2)
    #         pbar.update(batch_size)

    return final_prediction, prediction_variance / predicted_count
    # return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=data_shape)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=1)
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist()[1:]:
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def get_test_indices(testing_file):
    return pickle_load(testing_file)


def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])


def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)


def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(model, open_data_file.root.data[index], open_data_file.root.affine)


def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)


# obsolete
def prediction_to_image(prediction, label_map=False, threshold=0.5, labels=None):
    if prediction.shape[0] == 1:
        data = prediction[0]
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    elif prediction.shape[1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    return get_image(data)


def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(get_image(prediction[0, i]))
    return prediction_images


def run_validation_case_from_image_simple(output_dir, model, processed_image, patch_shape, image_gt=None,
                                          image_pred=None, overlap_factor=0.8, prev_truth_index=None,
                                          prev_truth_size=None, pred_index=None, pred_size=None, is3d=False):
    """

    :param output_dir: folder to save prediction results in
    :param model: loaded trained model
    :param processed_image: as a numpy ndarray, after any processing (scaling, normalization, augmentation) wanted
    :param patch_shape: size of patch on which to run model
    :param image_gt: optional, {0,1} ndarray of ground truth segmentation
    :param image_pred: optional, {0-1} ndarray of predicted segmentation
    :param overlap_factor: amount of overlap between consecutive patches, float 0-1
    :param prev_truth_index: if truth used in prediction - the starting index in input batch (depth), else None
    :param prev_truth_size:if truth used in prediction - amount of truth slices in input batch, else None
    :param pred_index: if other pred used in prediction - the starting index in input batch (depth), else None
    :param pred_size: if other pred used in prediction - amount of pred slices in input batch (depth), else None
    :return: prediction + path to saved location
    """

    prediction, _ = \
        patch_wise_prediction(model=model, data=np.expand_dims(processed_image.squeeze(), 0), overlap_factor=overlap_factor,
                              patch_shape=patch_shape, truth_data=image_gt, prev_truth_index=prev_truth_index,
                              prev_truth_size=prev_truth_size, pred_data=image_pred, pred_index=pred_index,
                              pred_size=pred_size, is3d=is3d)  # [np.newaxis]
    prediction = prediction.squeeze()
    prediction_image = get_image(prediction)

    filename = os.path.join(output_dir, "prediction.nii.gz")
    name_counter = 0
    while os.path.exists(filename):
        name_counter += 1
        filename = os.path.join(output_dir, "prediction_{0}.nii.gz".format(name_counter))
    print("Saving to {}".format(filename))
    prediction_image.to_filename(filename)
    return prediction, filename


def run_validation_cases_from_image_simple(list_of_processed_images, model_file, list_of_output_folders,
                                           patch_shape, list_of_images_gt=None, list_of_images_preds=None,
                                           overlap_factor=0.8, prev_truth_index=None,
                                           prev_truth_size=None, pred_index=None, pred_size=None):
    """

    :param list_of_processed_images:
    :param model_file:
    :param list_of_output_folders:
    :param patch_shape:
    :param overlap_factor:
    :param prev_truth_index:
    :param prev_truth_size:
    :param pred_index:
    :param pred_size:
    :return:
    """
    model = load_old_model(model_file)
    image_gt = None
    image_pred = None
    file_names = []
    predictions = []
    for i, processed_image in enumerate(list_of_processed_images):
        if list_of_images_gt:
            image_gt = list_of_images_gt[i]
        if list_of_images_preds:
            image_pred = list_of_images_preds[i]
        cur_pred, pred_path = run_validation_case_from_image_simple(list_of_output_folders[i], model, processed_image,
                                                                    patch_shape, image_gt=image_gt,
                                                                    image_pred=image_pred,
                                                                    overlap_factor=overlap_factor,
                                                                    prev_truth_index=prev_truth_index,
                                                                    prev_truth_size=prev_truth_size,
                                                                    pred_index=pred_index, pred_size=pred_size)
        file_names.append(pred_path)
        predictions.append(cur_pred)
    return predictions, file_names


def run_validation_case(data_index, output_dir, model, data_file, training_modalities, patch_shape,
                        overlap_factor=0, permute=False, prev_truth_index=None, prev_truth_size=None, is3d=False,
                        pred_index=None, pred_size=None, use_augmentations=False, scale_xy=None, resolution_file=''):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is 
    considered a positive result and will be assigned a label.  
    :param labels:
    :param training_modalities:
    :param data_file:
    :param scale_xy: wether to attempt and scale the image to main resolution
    :param resolution_file: a file containing a dict of all existing scans' resolutions
    :param model:
    """
    cur_subject_id = data_file.root.subject_ids[data_index].decode()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_data = np.asarray([data_file.root.data[data_index]])
    test_data = scale_data(test_data, cur_subject_id, dict_pkl=resolution_file, scale_xy=scale_xy)
    if prev_truth_index is not None:
        test_truth_data = np.asarray([data_file.root.truth[data_index]])
        test_truth_data = scale_data(test_truth_data, cur_subject_id, dict_pkl=resolution_file, scale_xy=scale_xy)
    else:
        test_truth_data = None

    if pred_index is not None:
        test_pred_data = np.asarray([data_file.root.pred[data_index]])
        test_pred_data = scale_data(test_pred_data, cur_subject_id, dict_pkl=resolution_file, scale_xy=scale_xy)
    else:
        test_pred_data = None

    # for i, modality in enumerate(training_modalities):
    #     image = get_image(test_data[i])
    #     image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))
    try:
        test_truth = np.asarray([data_file.root.truth[data_index]])
        test_truth = get_image(scale_data(test_truth, cur_subject_id, dict_pkl=resolution_file, scale_xy=scale_xy))
        test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))
    except:
        pass

    if patch_shape == test_data.shape[-3:]:
        print("Warning - went in where it wasn't expected!!!!!")
        prediction = predict(model, test_data, permute=permute)
    else:
        if use_augmentations:
            prediction = predict_augment(data=test_data, model=model, overlap_factor=overlap_factor,
                                         patch_shape=patch_shape)
        else:
            prediction, prediction_var = \
                patch_wise_prediction(model=model, data=test_data, overlap_factor=overlap_factor,
                                      patch_shape=patch_shape, permute=permute,
                                      truth_data=test_truth_data, prev_truth_index=prev_truth_index,
                                      prev_truth_size=prev_truth_size,
                                      pred_data=test_pred_data, pred_index=pred_index, pred_size=pred_size, is3d=is3d) #[np.newaxis]
    # if prediction.shape[-1] > 1:
    #     prediction = prediction[..., 1]
    prediction = prediction.squeeze()
    prediction_image = get_image(prediction)
    # prediction_var = prediction_var.squeeze()
    # prediction_var_image = get_image(prediction_var)

    name_counter = 0
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            filename = os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1))
            while os.path.exists(filename):
                name_counter += 1
                filename = os.path.join(output_dir, "prediction_{0}_{1}.nii.gz".format(i + 1, name_counter))
            image.to_filename(filename)
    else:
        filename = os.path.join(output_dir, "prediction.nii.gz")
        # var_fname = os.path.join(output_dir, "prediction_variance.nii.gz")
        while os.path.exists(filename):
            name_counter += 1
            filename = os.path.join(output_dir, "prediction_{0}.nii.gz".format(name_counter))
            var_fname = os.path.join(output_dir, "prediction_variance_{0}.nii.gz".format(name_counter))
        print("Saving to {}".format(filename))
        prediction_image.to_filename(filename)
        # prediction_var_image.to_filename(var_fname)
    return filename


def run_validation_cases(validation_keys_file, model_file, training_modalities, hdf5_file, patch_shape,
                         output_dir=".", overlap_factor=0, permute=False, is3d=False,
                         prev_truth_index=None, prev_truth_size=None, pred_index=None, pred_size=None,
                         use_augmentations=False, scale_xy=None, resolution_file=''):
    file_names = []
    validation_indices = pickle_load(validation_keys_file)
    # validation_indices = [23, 24, 7]  # 2
    # validation_indices = [5, 11, 15] # 1
    # validation_indices = [23, 5, 6] # 0
    model = load_old_model(get_last_model_path(model_file))
    data_file = tables.open_file(hdf5_file, "r")
    for index in validation_indices:
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        file_names.append(
            run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                                training_modalities=training_modalities, overlap_factor=overlap_factor, is3d=is3d,
                                permute=permute, patch_shape=patch_shape, prev_truth_index=prev_truth_index,
                                prev_truth_size=prev_truth_size, pred_index=pred_index, pred_size=pred_size,
                                use_augmentations=use_augmentations, scale_xy=scale_xy, resolution_file=resolution_file))
    data_file.close()
    return file_names


def predict(model, data, permute=False):
    if permute:
        predictions = list()
        for batch_index in range(data.shape[0]):
            predictions.append(predict_with_permutations(model, data[batch_index]))
        return np.asarray(predictions)
    else:
        return model.predict(data)


def predict_with_permutations(model, data):
    predictions = list()
    for permutation_key in generate_permutation_keys():
        temp_data = permute_data(data, permutation_key)[np.newaxis]
        predictions.append(reverse_permute_data(model.predict(temp_data)[0], permutation_key))
    return np.mean(predictions, axis=0)
