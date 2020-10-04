import nibabel as nib
import os
from fetal_net.metrics import dice_coefficient_np, vod_coefficient_np, get_surface_distances, \
    false_negative_rate, false_positive_rate
from fetal_net.postprocess import postprocess_prediction
import glob
import pandas as pd


configs_folder = r"/datadrive/configs"
data_folder = r"/home/Shai/all_placenta_data/"
exp_name_prefix = "20200911_single_res_new_data_cross_val_train_"
exp_name_prefix = "20200930_reconstruct_2d_base_multiple_res_cross_val_train_"
output_file_path_volume = os.path.join(configs_folder, exp_name_prefix) + 'volume_results.xlsx'

all_rel_exps = glob.glob(os.path.join(configs_folder, exp_name_prefix + '*'))
subjects_results_dict = {}


### VOLUME METRICS (not slice-wise) ###

for exp in all_rel_exps:
    print(f"In exp: {exp}")
    pred_dir = os.path.join(exp, 'predictions', 'test')
    if not os.path.exists(pred_dir):
        print(f"Skipping: {exp}")
        continue    
    sub_ids = os.listdir(pred_dir)
    for sub_id in sub_ids:
        pred = nib.load(os.path.join(pred_dir, sub_id, 'prediction.nii.gz')).get_data()
        binary_pred = postprocess_prediction(pred)
        truth = nib.load(os.path.join(data_folder, sub_id, 'truth.nii')).get_data()

        cur_dice = dice_coefficient_np(truth, binary_pred)
        cur_vo = vod_coefficient_np(truth, binary_pred)
        cur_haus, cur_assd = get_surface_distances(truth.astype('uint8'), binary_pred.astype('uint8'))
        fnr = false_negative_rate(truth, binary_pred)
        fpr = false_positive_rate(truth, binary_pred)

        if sub_id not in subjects_results_dict:
            subjects_results_dict[sub_id] = {'dice': cur_dice, 'vo': cur_vo, 'hausdorff': cur_haus, 'assd': cur_assd,
                                             'fnr': fnr, 'fpr': fpr}
        else:
            while sub_id in subjects_results_dict:
                sub_id = sub_id + "_"
            subjects_results_dict[sub_id] = {'dice': cur_dice, 'vo': cur_vo, 'hausdorff': cur_haus, 'assd': cur_assd,
                                             'fnr': fnr, 'fpr': fpr}

df = pd.DataFrame.from_dict(subjects_results_dict).transpose()
df.to_excel(output_file_path_volume)
print(df.mean())
print(df.std())

