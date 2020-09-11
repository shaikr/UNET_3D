import os
from pathlib import Path
import random
import numpy as np
import pickle
import tables
import os


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


# def delete_existing_files():
#     os.remove(r"/data/home/Shai/UNET_3D_NEW/debug_split/validation_ids.pkl")
#     os.remove(r"/data/home/Shai/UNET_3D_NEW/debug_split/training_ids.pkl")
#     os.remove(r"/data/home/Shai/UNET_3D_NEW/debug_split/test_ids.pkl")


def create_files(cur_exp_name, training_list, validation_list, test_list):
    exp_folder = os.path.join(r"/datadrive/configs", cur_exp_name)
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    os.mkdir(os.path.join(exp_folder, "debug_split"))
    pickle_dump(training_list, os.path.join(exp_folder, "debug_split", "training_ids.pkl"))
    pickle_dump(validation_list, os.path.join(exp_folder, "debug_split", "validation_ids.pkl"))
    pickle_dump(test_list, os.path.join(exp_folder, "debug_split", "test_ids.pkl"))


def run_cross_val_training(existing_data_file_path, exp_names_prefix, conf_to_imitate=None):
    data_file = tables.open_file(existing_data_file_path, "r")
    all_list = list(range(len(data_file.root.subject_ids)))
    data_file.close()

    print('# of subjects: {}'.format(len(all_list)))
    random.shuffle(all_list)
    all_list_temp = all_list
    # all_list_temp.remove(5)
    # all_list_temp.remove(6)
    # all_list_temp.remove(23)

    all_experiement_names = []

    n_test = 3
    n_iters = int(np.ceil(float(len(all_list_temp)) / n_test))
    for i in range(n_iters):
        print("In round {} out of {}".format(i+1, n_iters))

        test_list = all_list_temp[:n_test]  # [5, 6, 23]
        validation_list = all_list_temp[n_test:2*n_test]
        training_list = all_list_temp[2*n_test:]

        cur_exp_name = '{}_cross_val_train_{}'.format(exp_names_prefix, i+1)

        try:
            create_files(cur_exp_name, training_list, validation_list, test_list)
        except:
            print('Problem creating files')

        print('Created files, now training')
        if conf_to_imitate is None:
            cmd = "python train_fetal.py --experiment_name='{}'".format(cur_exp_name)
        else:
            cmd = "python train_fetal.py --experiment_name='{}'" \
                  " --imitate_experiment='{}'".format(cur_exp_name, conf_to_imitate)
        print(cmd)
        os.system(cmd)

        print("Finished training, now running on test")
        conf_dir = '../../../../../datadrive/configs/' + '{}'.format(cur_exp_name)
        cmd = "python predict.py --split='test' --config='{}'".format(conf_dir)
        print(cmd)
        os.system(cmd)

        print('Finished forward')
        all_list_temp = all_list_temp[n_test:] + all_list_temp[:n_test]
        all_experiement_names = all_experiement_names + ['{}'.format(cur_exp_name)]
    return all_experiement_names


conf_to_imitate = r"20200906_single_res_new_data"
existing_data_fpath = r"/datadrive/configs/20200906_single_res_new_data/fetal_data.h5"
exp_names_prefix = r"20200911_single_res_new_data"
run_cross_val_training(existing_data_file_path=existing_data_fpath, exp_names_prefix=exp_names_prefix,
                       conf_to_imitate=conf_to_imitate)

# import glob
# import nibabel as nib
# data_pref = r"/data/home/Shai/placenta_data"
# exps = glob.glob('/datadrive/configs/64_64_5_cross_val_*')
#
# for e in exps:
#     cur_p = os.path.join(e, 'predictions', 'test')
#     subs = os.listdir(cur_p)
#     for s_id in subs:
#         print("Adding prediction to subject {}".format(s_id))
#         if not os.path.exists(os.path.join(data_pref, s_id, 'prediction.nii')):
#             a_tmp = nib.load(os.path.join(cur_p, s_id, 'prediction.nii.gz'))
#             nib.save(a_tmp, os.path.join(data_pref, s_id, 'prediction.nii'))
