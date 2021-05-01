import joblib
import json
import numpy as np

schema_path = 'service/schema.json'

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)


def load_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

def predict(data):
    model = joblib.load('service/Model/model.pkl')
    prediction = model.predict(data)
    return prediction
    