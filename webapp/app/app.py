import os
import pickle
import pandas as pd
import json

from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

from flask import Flask, request, jsonify

app = Flask(__name__)


def get_data():
    links = ["https://www.nordpoolgroup.com/4adabc/globalassets/marketdata-excel-files/elspot-prices_2020_hourly_eur.xls", 
             "https://www.nordpoolgroup.com/4adabc/globalassets/marketdata-excel-files/elspot-prices_2019_hourly_eur.xls"]

    df_list = []

    for link in links:
        df = pd.read_html(link, decimal=",", thousands=".")[0]
        df.columns = [c[-1] for c in df.columns]
        df = df.rename(columns={"Unnamed: 0_level_2": "Date"})
        df_list.append(df)

    df = pd.concat(df_list)

    for col in df:
        if f"{col}.1" in df.columns:
            df[col] = df[col].fillna(df[f"{col}.1"])
            df = df.drop(f"{col}.1", axis=1)

    df = df.dropna().reset_index(drop=True)

    return df

def filter_data(data, date_from, date_to):
    data['Date'] = pd.to_datetime(data['Date'])

    date_from = datetime.strptime(date_from, "%Y%m%d")
    date_to = datetime.strptime(date_to, "%Y%m%d")
    
    data = data[data["Date"] > date_from]
    data = data[data["Date"] < date_to]

    return data

def train_model(df, model, features, target="FR", train_share = 0.8):

    models_list = {
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "lasso": Lasso(),
        "regression": LinearRegression()
    }
    
    x_train = df.drop([target, "Date", "Hours"], axis=1).iloc[:int(train_share * len(df))]
    y_train = df[target].reindex(x_train.index)
    x_test = df.drop([target, "Date", "Hours"], axis=1).drop(x_train.index, axis=0)
    y_test = df[target].reindex(x_test.index)

    x_train = x_train[features]


    with open('features.json', 'w') as fp:
        json.dump({'features': features}, fp)

    models_list[model].fit(x_train, y_train)
    pickle.dump(models_list[model], open(model + '.pkl', 'wb'))


def test_model(df, model, target="FR", train_share = 0.8):

    x_train = df.drop([target, "Date", "Hours"], axis=1).iloc[:int(train_share * len(df))]
    y_train = df[target].reindex(x_train.index)
    x_test = df.drop([target, "Date", "Hours"], axis=1).drop(x_train.index, axis=0)
    y_test = df[target].reindex(x_test.index)

    predictions = pd.DataFrame(index=x_test.index)

    with open('features.json', 'r') as fp:
        features = json.load(fp)['features']

    x_test = x_test[features]

    loaded_model = pickle.load(open(model + '.pkl', 'rb'))
    predictions[model] = loaded_model.predict(x_test)

    return predictions.values, features

@app.route("/api/train", methods=['GET'])
def train_api():
    if 'train_from' in request.args:
        train_from = str(request.args['train_from'])
    else:
        return "Error: No train_from field provided. Please specify an train_from value."

    if 'train_to' in request.args:
        train_to = str(request.args['train_to'])
    else:
        return "Error: No train_to field provided. Please specify an train_to value."

    if 'model' in request.args:
        model = request.args['model']
    else:
        return "Error: No model field provided. Please specify an model to train."

    if 'target' in request.args:
        target = str(request.args['target'])
    else:
        return "Error: No target field provided. Please specify a target."

    if 'features' in request.args:
        features = request.args.getlist('features')
    else:
        return "Error: No target field provided. Please specify a target."        

    data = get_data()
    data = filter_data(data, train_from, train_to)

    train_model(data, model, features, target)

    return "Model {} trained sucessfully".format(model)

@app.route("/api/predict", methods=['GET'])
def predict_api():

    if 'test_from' in request.args:
        test_from = str(request.args['test_from'])
    else:
        return "Error: No test_from field provided. Please specify an test_from value."

    if 'test_to' in request.args:
        test_to = str(request.args['test_to'])
    else:
        return "Error: No test_to field provided. Please specify an test_to value."

    if 'model' in request.args:
        model = request.args['model']
    else:
        return "Error: No model field provided. Please specify an model to train."

    if 'target' in request.args:
        target = str(request.args['target'])
    else:
        return "Error: No target field provided. Please specify a target."

    data = get_data()
    data = filter_data(data, test_from, test_to)

    if not os.path.exists(model + '.pkl'):
        return "First train model {}".format(model)

    predictions, features = test_model(data, model, target)

    return jsonify({'predictions': [float(p) for p in predictions],
                    'features': features,
                    'model': model})

app.run(port=1234, host='0.0.0.0')