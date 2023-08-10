from flask import Flask,request, jsonify
import joblib as jl
import pickle as pkl
import routes

app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

@app.route('/api/predict', methods=[ 'POST'])
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
 
if __name__ == '__main__':
   
    #Load model
    model = jl.load("log_reg_model.pkl") 

    #Load model columns
    model_columns = jl.load("log_reg_model_columns.pkl")

    scaler = None
    with open('log_reg_scaler.pkl', 'rb') as f:
        scaler = pkl.load(f)

    app.run()
    #import os
    #HOST = os.environ.get('SERVER_HOST', 'localhost')
    #try:
    #    PORT = int(os.environ.get('SERVER_PORT', '5555'))
    #except ValueError:
    #    PORT = '5555'
    #app.run(HOST, PORT)

        