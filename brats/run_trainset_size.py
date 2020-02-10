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


def run_training_size_exp(subject_ids, exp_names_prefix, conf_to_imitate=None):
    all_list = subject_ids #[0, 10, 20, 6, 12, 16, 1, 3, 5, 9, 13, 14, 15, 17, 18, 21, 22, 23, 24, 25, 26, 19, 11, 8, 7, 4, 2]

    trainset_sizes = [4, 8, 12, 16, 20]
    n_iters = [5, 4, 3, 2, 1]

    all_experiment_names = []

    # checks for 3 random sets of validation+test set
    for iValMode in range(3):
        print(f'Started iteration {iValMode+1} - fixing validation and test set')
        random.shuffle(all_list)
        all_list_temp = all_list
        test_list = all_list_temp[:3]
        validation_list = all_list_temp[3:6]
        training_list = all_list_temp[6:]
        print(f"Test set: {test_list}")
        print(f"Val set: {validation_list}")
        for i in range(len(trainset_sizes)):
            print(f"In round {i+1} out of {len(trainset_sizes)}")

            for j in range(n_iters[i]):
                print(f"In sample {j+1} out of {n_iters[i]}")
                try:
                    delete_existing_files()
                except:
                    print('Problem deleting files')

                try:
                    cur_training_list = random.sample(training_list, trainset_sizes[i])
                    create_files(cur_training_list, validation_list, test_list)
                except:
                    print('Problem deleting files')

                print('Created files, now training')
                cur_exp_name = f"trainset_experiment_{iValMode}_val_{trainset_sizes[i]}_train_{j}_iter"
                if conf_to_imitate is None:
                    cmd = f"python train_fetal.py --experiment_name='{cur_exp_name}'"
                else:
                    cmd = f"python train_fetal.py --experiment_name='{cur_exp_name}'" \
                          f" --imitate_experiment='{conf_to_imitate}'"
                print(cmd)
                os.system(cmd)

                print("Finished training, now running on test")
                conf_dir = '../../../../../datadrive/configs/' + f'{cur_exp_name}'
                cmd = f"python predict.py --split='test' --config='{conf_dir}'"
                print(cmd)
                os.system(cmd)

                print('Finished forward')
                all_experiment_names = all_experiment_names + [cur_exp_name]
    return all_experiment_names

if __name__ == "__main__":
    subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    exp_names_prefix = ''
    to_imit = 'experiment_no_augs'
    run_training_size_exp(subject_ids, exp_names_prefix, to_imit)
    