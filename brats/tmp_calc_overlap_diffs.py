import nibabel as nib
import os
from fetal_net.metrics import dice_coefficient_np, vod_coefficient_np, get_surface_distances, \
    false_negative_rate, false_positive_rate, recall_rate
from fetal_net.postprocess import postprocess_prediction
import glob
import numpy as np
import pandas as pd
import argparse


def main(configs_folder, data_folder, exp_name):
    output_file_path_volume = os.path.join(configs_folder, exp_name) + 'volume_results.xlsx'
    output_file_path_area = os.path.join(configs_folder, exp_name) + 'area_results.xlsx'

    subjects_vol_results = pd.DataFrame(columns=['EXPERIMENT_NAME', 'TEST_ID', 'NSLICES', 'VO_COEFF', 'DICE_COEFF',
                                                 'HAUSDORFF_DISTANCE', 'ASSD', 'RECALL', 'FPR'])

    subjects_area_results = pd.DataFrame(columns=['EXPERIMENT_NAME', 'TEST_ID', 'ISBLANKS', 'NSLICES',
                                                  'MEAN_VO_COEFF', 'STD_VO_COEFF',
                                                  'MEAN_DICE_COEFF', 'STD_DICE_COEFF',
                                                  'MEAN_RECALL', 'MEAN_FPR'])

    cur_results_row_vol = 0
    cur_results_row_area = 0
    pred_dir = os.path.join(configs_folder, exp_name, 'predictions', 'test')
    sub_ids = os.listdir(pred_dir)
    for sub_id in sub_ids:
        avail_preds = glob.glob(os.path.join(pred_dir, sub_id, "prediction"))
        for avail_pred in avail_preds:
            try:
                exp_suffix = os.path.basename(avail_pred)
                pred_im = nib.load(avail_pred)
                pred = pred_im.get_data()
                binary_pred = postprocess_prediction(pred)
                truth = nib.load(os.path.join(data_folder, sub_id, 'truth.nii')).get_data()

                ### VOLUME METRICS (not slice-wise) ###
                cur_dice = dice_coefficient_np(truth, binary_pred)
                cur_vo = vod_coefficient_np(truth, binary_pred)
                cur_haus, cur_assd = get_surface_distances(truth.astype('uint8'), binary_pred.astype('uint8'))
                fnr = recall_rate(truth, binary_pred)
                fpr = false_positive_rate(truth, binary_pred)

                subjects_vol_results.loc[cur_results_row_vol] = [exp_name + '_' + exp_suffix, sub_id, truth.shape[-1],
                                                                 cur_vo, cur_dice, cur_haus, cur_assd, fnr, fpr]
                cur_results_row_vol += 1

                ### AREA METRICS - all ###
                dice_coeffs = [dice_coefficient_np(truth[:, :, i], binary_pred[:, :, i]) for i in range(truth.shape[-1])]
                vods = [vod_coefficient_np(truth[:, :, i], binary_pred[:, :, i]) for i in range(truth.shape[-1])]
                fnrs = [recall_rate(truth[:, :, i], binary_pred[:, :, i]) for i in range(truth.shape[-1])]
                fprs = [false_positive_rate(truth[:, :, i], binary_pred[:, :, i]) for i in range(truth.shape[-1])]
                subjects_area_results.loc[cur_results_row_area] = [exp_name + '_' + exp_suffix, sub_id, -1,
                                                                   truth.shape[-1], np.mean(vods), np.std(vods),
                                                                   np.mean(dice_coeffs), np.std(dice_coeffs),
                                                                   np.mean(fnrs), np.mean(fprs)]
                cur_results_row_area += 1

                truth_sums = np.sum(np.sum(truth, axis=0), axis=0)
                non_blank_inds = [i for i in range(truth.shape[-1]) if truth_sums[i] > 0]
                blank_inds = [i for i in range(truth.shape[-1]) if truth_sums[i] == 0]

                ### AREA METRICS - non-blanks ###
                subjects_area_results.loc[cur_results_row_area] = [exp_name + '_' + exp_suffix, sub_id, 0,
                                                                   len(non_blank_inds),
                                                                   np.mean([vods[i] for i in non_blank_inds]),
                                                                   np.std([vods[i] for i in non_blank_inds]),
                                                                   np.mean([dice_coeffs[i] for i in non_blank_inds]),
                                                                   np.std([dice_coeffs[i] for i in non_blank_inds]),
                                                                   np.mean([fnrs[i] for i in non_blank_inds]),
                                                                   np.mean([fprs[i] for i in non_blank_inds])]
                cur_results_row_area += 1

                ### AREA METRICS - blanks ###
                subjects_area_results.loc[cur_results_row_area] = [exp_name + '_' + exp_suffix, sub_id, 1,
                                                                   len(blank_inds),
                                                                   np.mean([vods[i] for i in blank_inds]),
                                                                   np.std([vods[i] for i in blank_inds]),
                                                                   np.mean([dice_coeffs[i] for i in blank_inds]),
                                                                   np.std([dice_coeffs[i] for i in blank_inds]),
                                                                   np.mean([fnrs[i] for i in blank_inds]),
                                                                   np.mean([fprs[i] for i in blank_inds])]
                cur_results_row_area += 1
            except Exception as e:
                print(e)
                continue

    subjects_vol_results.to_excel(output_file_path_volume)
    print(subjects_vol_results.mean())
    print(subjects_vol_results.std())
    subjects_area_results.to_excel(output_file_path_area)
    print(subjects_area_results.mean())
    print(subjects_area_results.std())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_folder", help="specifies config folder path",
                        type=str, required=False, default=r"/datadrive/configs")
    parser.add_argument("--data_folder", help="where is the relevant data",
                        type=str, required=False, default=r"/home/Shai/placenta_data/")
    parser.add_argument("--experiment", help="specifies what experiment to go over",
                        type=str, required=True)
    opts = parser.parse_args()

    main(opts.configs_folder, opts.data_folder, opts.experiment)
