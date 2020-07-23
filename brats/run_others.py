import os
from pathlib import Path
import glob

#exp_names = os.listdir('/datadrive/configs')
overlap_factors = [0.15, 0.35, 0.55, 0.75, 0.95]
exp_names = ["20200526_normalization_each_minmax", "20200526_normalization_each_clip_and", "20200526_normalization_each_stretch_and"]
exp_names = ["20200619_3d_unet_attempts_with_pred_00", "20200621_3d_unet_attempts_with_pred_later_slices_00"]
for e in exp_names:
    conf_dir = '../../../../../datadrive/configs/' + e
    #os.rename(os.path.join(conf_dir, 'fetal_data.h5'), os.path.join(conf_dir, 'fetal_data_original.h5'))
    #os.rename(os.path.join(conf_dir, 'april_data.h5'), os.path.join(conf_dir, 'fetal_data.h5'))
    #conf_dir='../configs/' + e
    #if not Path(os.path.join(conf_dir, 'predictions')).exists():
    cmd = "python predict.py --split='test' --config='{}'".format(conf_dir)
    print(cmd)
    os.system(cmd)

    # for o_f in overlap_factors[:1]:
    #     cmd = "python predict.py --split='test' --config='{}' --overlap_factor={}".format(conf_dir, o_f)
    #     print(cmd)
    #     os.system(cmd)