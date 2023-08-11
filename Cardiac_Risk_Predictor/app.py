from flask import Flask,request, jsonify, redirect
import joblib as jl
import pickle as pkl
import traceback
import pandas as pd
import numpy as np
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)


# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

#model = None
#model_columns = None
#scaler = None

#Load model
model = jl.load("log_reg_model.pkl") 

#Load model columns
model_columns = jl.load("log_reg_model_columns.pkl")

with open('log_reg_scaler.pkl', 'rb') as f:
   scaler = pkl.load(f)

# Automatically redirect to Swagger UI
@app.route('/')
def index():
    return redirect('/swagger', code=302)

@app.route('/predict', methods=[ 'POST'])
def predict():
    """
    API to handle the prediction process.
    """
    if model:
        try:
            json_request = request.json
            print(json_request)

            # load to a dataframe
            df = pd.DataFrame(json_request)
            print(df.head())


            # reindex to match the columns of the model
            # any missing columns will be replaced with zero.
            df = df.reindex(columns=model_columns, fill_value=0)
            print(f'After reindexing : {df.head()}')

            # scale
            df = scaler.transform(df)

            # predict
            prediction_result = list(model.predict(df))
            print(prediction_result)
                
            return jsonify({'prediction_result': str(prediction_result)})

        except:
            return jsonify({'error_log_prediction': traceback.format_exc()})
    else:
        return ('Model not defined')


# Create a Swagger UI blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    '/swagger',
    '/static/swagger.json',  # Path to swagger.json file
    config={
        'app_name': "Cardiac Risk Prediction API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix="/swagger")
 
if __name__ == '__main__':
    app.run()
    #import os
    #HOST = os.environ.get('SERVER_HOST', 'localhost')
    #try:
    #    PORT = int(os.environ.get('SERVER_PORT', '5555'))
    #except ValueError:
    #    PORT = '5555'
    #app.run(HOST, PORT)

        