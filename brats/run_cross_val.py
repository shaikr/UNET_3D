import os
from pathlib import Path
import random
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
    test_list = [5, 6, 23]  # all_list_temp[:3]
    print('# of subjects: {}'.format(len(all_list)))
    random.shuffle(all_list)
    all_list_temp = all_list
    all_list_temp.remove(5)
    all_list_temp.remove(6)
    all_list_temp.remove(23)

    all_experiement_names = []

    # TODO: currently assumes 27 scans
    for i in range(9):
        print(f"In round {i+1} out of 9")

        validation_list = all_list_temp[:3]
        training_list = all_list_temp[3:]

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
            cmd = f"python train_fetal.py --experiment_name='{exp_names_prefix}_cross_val_train_{i+1}'"
        else:
            cmd = f"python train_fetal.py --experiment_name='{exp_names_prefix}_cross_val_train_{i+1}'" \
                  f" --imitate_experiment='{conf_to_imitate}"
        print(cmd)
        os.system(cmd)

        print("Finished training, now running on test")
        conf_dir = '../../../../../datadrive/configs/' + f'{exp_names_prefix}_cross_val_train_{i+1}'
        cmd = f"python predict.py --split='test' --config='{conf_dir}'"
        print(cmd)
        os.system(cmd)

        print('Finished forward')
        all_list_temp = all_list_temp[3:] + all_list_temp[:3]
        all_experiement_names = all_experiement_names + [f'{exp_names_prefix}_cross_val_train_{i+1}']
    return all_experiement_names


conf_to_imitate = r"new_arch_exp_64_64_5"
existing_data_fpath = r"/datadrive/configs/dump_folder/new_arch_exp_64_64_5/fetal_data.h5"
exp_names_prefix = r"64_64_5_cross_val_train"