import os
from pathlib import Path
import random
import pickle
import os

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def delete_existing_files():
    os.remove(r"/home/Shai/PycharmProjects/UNET_3D/debug_split/validation_ids.pkl")
    os.remove(r"/home/Shai/PycharmProjects/UNET_3D/debug_split/training_ids.pkl")
    os.remove(r"/home/Shai/PycharmProjects/UNET_3D/debug_split/test_ids.pkl")


def create_files(training_list, validation_list, test_list):
    pickle_dump(training_list, r"/home/Shai/PycharmProjects/UNET_3D/debug_split/training_ids.pkl")
    pickle_dump(validation_list, r"/home/Shai/PycharmProjects/UNET_3D/debug_split/validation_ids.pkl")
    pickle_dump(test_list, r"/home/Shai/PycharmProjects/UNET_3D/debug_split/test_ids.pkl")


all_list = [0, 10, 20, 6, 12, 16, 1, 3, 5, 9, 13, 14, 15, 17, 18, 21, 22, 23, 24, 25, 26, 19, 11, 8, 7, 4, 2]
random.shuffle(all_list)
nb_samples = 27
sample_list = list(range(nb_samples))
all_list_temp = all_list

for i in range(9):
    print(f"In round {i+1} out of 9")

    test_list = all_list_temp[:3]
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
    cmd = f"python train_fetal.py --experiment_name='cross_val_train_{i+1}'"
    print(cmd)
    os.system(cmd)

    print("Finished training, now running on test")
    conf_dir = '../../../../../datadrive/configs/' + f'cross_val_train_{i+1}'
    cmd = f"python predict.py --split='test' --config='{conf_dir}'"
    print(cmd)
    os.system(cmd)

    print('Finished forward')
    all_list_temp = all_list_temp[3:] + all_list_temp[:3]
