import yaml
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import load_config, load_data

def split(config):
    split_config = config['split_data']
    feature_config = config['features']

    data = pd.read_csv(feature_config['path'])
    x_train, x_test = train_test_split(data, 
                                test_size=0.2, random_state=42)    
    x_train.to_csv(split_config['xtrain'], index = False)
    x_test.to_csv(split_config['xtest'], index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help = 'config file path', default= 'params.yaml')
    args = parser.parse_args()
    config = load_config(args.config)  
    split(config)