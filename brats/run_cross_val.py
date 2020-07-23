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


def delete_existing_files():
    os.remove(r"/data/home/Shai/UNET_3D_NEW/debug_split/validation_ids.pkl")
    os.remove(r"/data/home/Shai/UNET_3D_NEW/debug_split/training_ids.pkl")
    os.remove(r"/data/home/Shai/UNET_3D_NEW/debug_split/test_ids.pkl")


def create_files(training_list, validation_list, test_list):
    pickle_dump(training_list, r"/data/home/Shai/UNET_3D_NEW/debug_split/training_ids.pkl")
    pickle_dump(validation_list, r"/data/home/Shai/UNET_3D_NEW/debug_split/validation_ids.pkl")
    pickle_dump(test_list, r"/data/home/Shai/UNET_3D_NEW/debug_split/test_ids.pkl")


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

        test_list = all_list_temp[:3]  # [5, 6, 23]
        validation_list = all_list_temp[3:6]
        training_list = all_list_temp[6:]

        try:
            delete_existing_files()
        except:
            print('Problem deleting files')

        try:
            create_files(training_list, validation_list, test_list)
        except:
            print('Problem deleting files')

        print('Created files, now training')
        if conf_to_imitate is None:
            cmd = "python train_fetal.py --experiment_name='{}_cross_val_train_{}'".format(exp_names_prefix, i+1)
        else:
            cmd = "python train_fetal.py --experiment_name='{}_cross_val_train_{}'" \
                  " --imitate_experiment='{}'".format(exp_names_prefix, i+1, conf_to_imitate)
        print(cmd)
        os.system(cmd)

        print("Finished training, now running on test")
        conf_dir = '../../../../../datadrive/configs/' + '{}_cross_val_train_{}'.format(exp_names_prefix, i+1)
        cmd = "python predict.py --split='test' --config='{}'".format(conf_dir)
        print(cmd)
        os.system(cmd)

        print('Finished forward')
        all_list_temp = all_list_temp[3:] + all_list_temp[:3]
        all_experiement_names = all_experiement_names + ['{}_cross_val_train_{}'.format(exp_names_prefix, i+1)]
    return all_experiement_names


conf_to_imitate = r"new_arch_exp_64_64_5"
conf_to_imitate = r"20200404_single_resolution_iter3"
existing_data_fpath = r"/datadrive/configs/dump_folder/new_arch_exp_64_64_5/fetal_data.h5"
existing_data_fpath = r"/datadrive/configs/20200404_single_resolution_iter3/fetal_data_orig.h5"
exp_names_prefix = r"single_resolution_64_64_5_cross_val_train"
run_cross_val_training(existing_data_file_path=existing_data_fpath, exp_names_prefix=exp_names_prefix, conf_to_imitate=conf_to_imitate)

