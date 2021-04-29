import pandas as pd
import joblib
import yaml

param = yaml.safe_load(open('src/param.yaml'))['feature']


model = joblib.load('model/model.pkl')
test_data = pd.read_csv('data/test.csv')

inp_feature = param['imp'][:9]
x = test_data[inp_feature]

pred = model.predict(x)

print(pred)
