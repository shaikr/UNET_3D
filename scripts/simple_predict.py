import os
import numpy as np
import nibabel as nib
from fetal_net.prediction import run_validation_cases_from_image_simple
from tqdm import tqdm

model_file_path = r"/datadrive/configs/20200404_single_resolution_iter3/fetal_net_model-epoch15-loss202.728-acc0.922.h5" #'path/trained/model'
images_dir = r"/data/home/Shai/side_data_dir" # 'path/scans/folder'
output_dir_path = r"/data/home/Shai/simple_results" #'path/results'

# Load all inference images
# Assumes structure of: images_dir -> subject ids folders -> image saved as 'volume.nii', but change as you please
subject_id_folders = os.listdir(images_dir)
volume_filename = 'volume.nii'
volumes_list = []
output_folders_path = []
for subject_id in subject_id_folders:
    cur_volume_path = os.path.join(images_dir, subject_id, volume_filename)
    volumes_list.append(nib.load(cur_volume_path).get_data())
    output_folders_path.append(os.path.join(output_dir_path, subject_id))


# Normalize - currently 'hard coded' for my existing normalization
def norm_image(image_as_np):
    m = image_as_np.mean(axis=(-1, -2, -3))
    s = image_as_np.std(axis=(-1, -2, -3))
    image_as_np = np.subtract(image_as_np, m, out=image_as_np, casting='unsafe')
    image_as_np = np.divide(image_as_np, s, out=image_as_np, casting='unsafe')
    return image_as_np


for i, im in enumerate(volumes_list):
    volumes_list[i] = norm_image(im)

# create output folders
for output_path in output_folders_path:
    os.makedirs(output_path, exist_ok=True)

# run prediction
predictions, predictions_paths = run_validation_cases_from_image_simple(volumes_list, model_file_path,
                                                                        output_folders_path, patch_shape=[64, 64, 5],
                                                                        overlap_factor=0.8)
