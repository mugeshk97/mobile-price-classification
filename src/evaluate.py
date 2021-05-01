import joblib
import argparse
import yaml
import pandas as pd
from get_data import load_config
from sklearn.metrics import accuracy_score
import json

def write_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent= 4)

def metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    return {'accuracy': accuracy}

def generate_report(config):
    split_config = config['split_data']
    test = pd.read_csv(split_config['xtest'])
    schema = config['train_schema']
    model_path = config['model_path']['path']
    report_path = config['report']['path']

    x = test.drop(schema['target'], axis =1)
    y = test[schema['target']]

    model = joblib.load(model_path)    
    y_pred = model.predict(x)

    report = metrics(y, y_pred)
    write_json(report_path, report)


    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help = 'config file path', default= 'params.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    generate_report(config)
