import os
from pathlib import Path


#exp_names = os.listdir('/datadrive/configs')
exp_names = ["experiment_all_samples_new", "experiment_minus_6", "experiment_minus_3"]
for e in exp_names:
    conf_dir = '../../../../../datadrive/configs/' + e
    #conf_dir='../configs/' + e
    #if not Path(os.path.join(conf_dir, 'predictions')).exists():
    cmd = f"python predict.py --split='test' --config='{conf_dir}'"
    print(cmd)
    os.system(cmd)
    cmd = f"python predict.py --split='val' --config='{conf_dir}'"
    print(cmd)
    os.system(cmd)