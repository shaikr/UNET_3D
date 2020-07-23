import os
from pathlib import Path
import random
import pickle
import os
import json
from .run_cross_val import un_cross_val_training
from .train_fetal import main_train

### Training using prediction has several stages:
# 1. Cross-training using some non-prediction-using configuration
# 2. Run prediction of all the cross-trained experiments to get unbiased predictions of all scans
# 3. move all predictions to the data folders
# 4. Run a prediction-using training configuration
# 5. Run the new model prediction
# 6. If want to apply consecutively - save step 5's results in data folder, recreate data file and run pred


### Base model cross val train ###

# create dummy experiment with wanted config


