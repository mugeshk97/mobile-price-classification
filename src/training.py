import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import yaml

param = yaml.safe_load(open('src/param.yaml'))['train']
data = pd.read_csv('data/feature_selected.csv')

depth = param['max_depth']

x = data.drop('price_range', axis =1)
y = data['price_range']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth = depth)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

joblib.dump(model, 'model/model.pkl') 