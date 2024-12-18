import os 
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch

def set_all_seeds(seed=42):
    
    # python's seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # torch's seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_training_data(df, cat_cols, cols2drop, target_cols, verbose=True, test_size=0.2):
    features2drop = cols2drop 
    targets = target_cols  
    cat_features = cat_cols 

    filtered_features = [i for i in df.columns if (i not in targets and i not in features2drop)]
    num_features = [i for i in filtered_features if i not in cat_features]

    if verbose:
        print('cat_features :', len(cat_features), cat_features)
        print('num_features :', len(num_features), num_features)
        print('targets', targets)

    X = df[filtered_features].drop(targets, axis=1, errors='ignore')
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def get_holidays():
    return {
    'halloween':    [ 7,  59, 112, 164, 216, 268, 320, 372],
    'thanksgiving': [11,  63, 116, 168, 220, 272, 324, 377],
    'christmas':    [15,  67, 119, 172, 224, 276, 328, 380],
    'newyear':      [16,  68, 120, 173, 225, 277, 329, 381],
    'pres_day':     [23,  75, 128, 180, 232, 284, 336, 389],
    'easter':       [28,  81, 133, 185, 238, 289, 341, 393],
    'mem_day':      [37,  89, 141, 194, 246, 298, 350],
    'indep_day':    [42,  95, 147, 199, 251, 303, 356],
    'labor_day':    [51, 103, 156, 208, 260, 312, 364]
    }

def get_holiday_names():
    return ['halloween', 'thanksgiving', 'christmas', 'newyear', 'pres_day', 'easter', 'mem_day', 'indep_day', 'labor_day']