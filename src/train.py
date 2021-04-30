import yaml
import joblib
import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from get_data import load_config
import logging

logging.basicConfig(filename='logs/log.txt', filemode= 'a',
                    format='%(asctime)s %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')



def model_train(config):
    model_config = config['DecisionTree']
    split_config = config['split_data']
    schema = config['train_schema']
    model_path = config['model_path']['path']
    depth = model_config['max_depth']
    
    train_data = pd.read_csv(split_config['xtrain'])
    if len(train_data.columns) == int(schema['n_columns']):    
        x = train_data.drop(schema['target'],axis =1)
        y = train_data[schema['target']]
        model = DecisionTreeClassifier(max_depth = depth)
        model.fit(x, y)
        joblib.dump(model, model_path)
    else:
        logging.error('Data Mismatch')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help = 'config file path', default= 'src/param.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    model_train(config)