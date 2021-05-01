import pandas as pd
import yaml
import argparse
import logging

logging.basicConfig(filename='logs/log.txt', filemode= 'a',
                    format='%(asctime)s %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def load_config(path):
    param = yaml.safe_load(open(path))
    return param

def load_data(config):
    data_path = config['data_src']['main_data']
    feature_config = config['features']
    try:
        data = pd.read_csv(data_path)
        data = data[feature_config['selected']]
        data.to_csv(feature_config['path'], index = False)
    except Exception as e:
        logging.error(e)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help = 'config file path', default= 'params.yaml')
    args = parser.parse_args()
    config = load_config(args.config)  
    load_data(config)