import nibabel as nib
import os
from fetal_net.metrics import dice_coefficient_np, vod_coefficient_np, get_surface_distances, \
    false_negative_rate, false_positive_rate, recall_rate
from fetal_net.postprocess import postprocess_prediction
import glob
import numpy as np
import pandas as pd
import argparse


def main(configs_folder, data_folder, exp_name_prefix):
    output_file_path_volume = os.path.join(configs_folder, exp_name_prefix) + 'volume_results.xlsx'
    output_file_path_area = os.path.join(configs_folder, exp_name_prefix) + 'area_results.xlsx'
    all_rel_exps = glob.glob(os.path.join(configs_folder, exp_name_prefix + '*'))
    subjects_vol_results_dict = {}
    subjects_area_results_dict = {}

    subjects_vol_results = pd.DataFrame(columns=['EXPERIMENT_NAME', 'TEST_ID', 'NSLICES', 'VO_COEFF', 'DICE_COEFF',
                                                 'HAUSDORFF_DISTANCE', 'ASSD', 'RECALL', 'FPR'])

    subjects_area_results = pd.DataFrame(columns=['EXPERIMENT_NAME', 'TEST_ID', 'ISBLANKS', 'NSLICES',
                                                  'MEAN_VO_COEFF', 'STD_VO_COEFF',
                                                  'MEAN_DICE_COEFF', 'STD_DICE_COEFF',
                                                  'MEAN_RECALL', 'MEAN_FPR'])

    cur_results_row_vol = 0
    cur_results_row_area = 0
    for exp in all_rel_exps:
        exp_base = os.path.basename(exp)
        print(f"In exp: {exp_base}")
        pred_dir = os.path.join(exp, 'predictions', 'test')
        if not os.path.exists(pred_dir):
            print(f"Skipping: {exp}")
            continue
        sub_ids = os.listdir(pred_dir)
        for sub_id in sub_ids:
            try:
                pred_im = nib.load(os.path.join(pred_dir, sub_id, 'prediction.nii.gz'))
                pred = pred_im.get_data()
                binary_pred = postprocess_prediction(pred)
                #save_path = os.path.join(data_folder, sub_id, f'{exp_base}_bin_pred.nii')
                #os.remove(save_path)
                #save_path = os.path.join(data_folder, sub_id, f'prediction_2d_depth3_bin_pred.nii')
                #nib.save(nib.Nifti1Image(binary_pred, pred_im.affine, header=pred_im.header), save_path)
                truth = nib.load(os.path.join(data_folder, sub_id, 'truth.nii')).get_data()

                ### VOLUME METRICS (not slice-wise) ###
                cur_dice = dice_coefficient_np(truth, binary_pred)
                cur_vo = vod_coefficient_np(truth, binary_pred)
                cur_haus, cur_assd = get_surface_distances(truth.astype('uint8'), binary_pred.astype('uint8'))
                fnr = recall_rate(truth, binary_pred)
                fpr = false_positive_rate(truth, binary_pred)

                subjects_vol_results.loc[cur_results_row_vol] = [os.path.basename(exp), sub_id, truth.shape[-1], cur_vo,
                                                                 cur_dice, cur_haus, cur_assd, fnr, fpr]
                cur_results_row_vol += 1

                ### AREA METRICS - all ###
                dice_coeffs = [dice_coefficient_np(truth[:, :, i], binary_pred[:, :, i]) for i in range(truth.shape[-1])]
                vods = [vod_coefficient_np(truth[:, :, i], binary_pred[:, :, i]) for i in range(truth.shape[-1])]
                fnrs = [recall_rate(truth[:, :, i], binary_pred[:, :, i]) for i in range(truth.shape[-1])]
                fprs = [false_positive_rate(truth[:, :, i], binary_pred[:, :, i]) for i in range(truth.shape[-1])]
                subjects_area_results.loc[cur_results_row_area] = [os.path.basename(exp), sub_id, -1, truth.shape[-1],
                                                                   np.mean(vods), np.std(vods), np.mean(dice_coeffs),
                                                                   np.std(dice_coeffs), np.mean(fnrs), np.mean(fprs)]
                cur_results_row_area += 1

                truth_sums = np.sum(np.sum(truth, axis=0), axis=0)
                non_blank_inds = [i for i in range(truth.shape[-1]) if truth_sums[i] > 0]
                blank_inds = [i for i in range(truth.shape[-1]) if truth_sums[i] == 0]

                ### AREA METRICS - non-blanks ###
                subjects_area_results.loc[cur_results_row_area] = [os.path.basename(exp), sub_id, 0, len(non_blank_inds),
                                                                   np.mean([vods[i] for i in non_blank_inds]),
                                                                   np.std([vods[i] for i in non_blank_inds]),
                                                                   np.mean([dice_coeffs[i] for i in non_blank_inds]),
                                                                   np.std([dice_coeffs[i] for i in non_blank_inds]),
                                                                   np.mean([fnrs[i] for i in non_blank_inds]),
                                                                   np.mean([fprs[i] for i in non_blank_inds])]
                cur_results_row_area += 1

                ### AREA METRICS - blanks ###
                subjects_area_results.loc[cur_results_row_area] = [os.path.basename(exp), sub_id, 1, len(blank_inds),
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
    parser.add_argument("--prefix", help="specifies what experiments to go over by their prefix",
                        type=str, required=True)
    opts = parser.parse_args()

    main(opts.configs_folder, opts.data_folder, opts.prefix)
