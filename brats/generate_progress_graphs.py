import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np


configs_folder = r"/datadrive/configs"
all_exps = os.listdir(configs_folder)

for exp in tqdm(all_exps):
    if os.path.exists(os.path.join(configs_folder, exp, 'training')): #and not os.path.exists(os.path.join(configs_folder, exp, 'progress_graph.png')):
        try:    
            df = pd.read_csv(os.path.join(configs_folder, exp, 'training'))
            if any(np.isinf(df).all()):
                print(f'found column all infs in {exp}')
            elif not all(np.isfinite(df).all()):
                print(f'found column some infs in {exp}')
            fig, axs = plt.subplots(2,1, sharex=True, figsize=(15,10))
            x = list(range(len(df)))
            axs[0].plot(x, df['loss'], label='Training loss')
            axs[0].plot(x, df['val_loss'], label='Validation loss')
            axs[0].legend(prop={'size': 14})
            axs[0].tick_params(axis="x", labelsize=12)
            axs[0].tick_params(axis="y", labelsize=12)
            if 'dice_coefficient' in df:
                axs[1].plot(x, df['dice_coefficient'], label='DICE')
                axs[1].plot(x, df['val_dice_coefficient'], label='Validation DICE')
                axs[1].legend(prop={'size': 14})
                axs[1].tick_params(axis="x", labelsize=12)
                axs[1].tick_params(axis="y", labelsize=12)
            fig.savefig(os.path.join(configs_folder, exp, 'progress_graph.png'))
            plt.close()
        except:
            print(f'Something went wrong in {exp}')

