import pandas as pd
import numpy as np
import yaml

param = yaml.safe_load(open('src/param.yaml'))['feature']

data = pd.read_csv('data/train.csv')

data = data[param['imp']]

data.to_csv('data/feature_selected.csv', index = False)