import nibabel as nib
import os
from pathlib import Path
import argparse


PRED_KEY = 'preds'
HEADER_KEY = 'header'
AFFINE_KEY = 'affine'


def main(configs_folder, resulting_folder, participating_exps):
    new_preds_path = os.path.join(configs_folder, resulting_folder, 'predictions', 'test' )
    Path(new_preds_path).mkdir(exist_ok=True, parents=True)
    dict_preds = {}

    for exp in participating_exps:
        full_exp_path = os.path.join(configs_folder, exp)
        print(f"In exp: {exp}")
        pred_dir = os.path.join(full_exp_path, 'predictions', 'test')
        if not os.path.exists(pred_dir):
            print(f"Skipping: {exp}")
            continue
        sub_ids = os.listdir(pred_dir)
        for sub_id in sub_ids:
            try:
                pred_im = nib.load(os.path.join(pred_dir, sub_id, 'prediction.nii.gz'))
                pred = pred_im.get_data()
                if sub_id in dict_preds:
                    dict_preds[sub_id][PRED_KEY].append(pred)
                else:
                    dict_preds[sub_id][PRED_KEY] = [pred]
                    dict_preds[sub_id][HEADER_KEY] = pred_im.header
                    dict_preds[sub_id][AFFINE_KEY] = pred_im.affine
            except Exception as e:
                print(e)
    for sub_id in dict_preds:
        avg_pred = sum(dict_preds[sub_id][PRED_KEY]) / len(dict_preds[sub_id][PRED_KEY])
        save_path = os.path.join(new_preds_path, sub_id, f'prediction.nii.gz')
        nib.save(nib.Nifti1Image(avg_pred, dict_preds[sub_id][AFFINE_KEY], header=dict_preds[sub_id][HEADER_KEY]),
                 save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_folder", help="specifies config folder path",
                        type=str, required=False, default=r"/datadrive/configs")
    parser.add_argument("--resulting_folder", help="specifies what folder to save results in",
                        type=str, required=True)
    parser.add_argument("--participating_exps", help="specifies what experiments to average",
                        type=list, required=True)
    opts = parser.parse_args()

    main(opts.configs_folder, opts.resulting_folder, opts.participating_exps)
