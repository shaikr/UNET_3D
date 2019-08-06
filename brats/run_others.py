import os
from pathlib import Path


#exp_names = os.listdir('/datadrive/configs')
overlap_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
exp_names = ["trainset_experiment_0_val_20_train_0_iter", "trainset_experiment_1_val_20_train_0_iter", "trainset_experiment_2_val_20_train_0_iter"]
for e in exp_names:
    conf_dir = '../../../../../datadrive/configs/' + e
    #conf_dir='../configs/' + e
    #if not Path(os.path.join(conf_dir, 'predictions')).exists():
    # cmd = f"python predict.py --split='val' --config='{conf_dir}'"
    # print(cmd)
    # os.system(cmd)
    for o_f in overlap_factors:
        cmd = f"python predict.py --split='test' --config='{conf_dir} --overlap_factor={o_f}'"
        print(cmd)
        os.system(cmd)