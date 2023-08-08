from flask import Flask,request, jsonify
import joblib as jl
import pickle as pkl
import routes

app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app


if __name__ == '__main__':
   
    #Load model
    model = jl.load("log_reg_model.pkl") 

    #Load model columns
    model_columns = jl.load("log_reg_model_columns.pkl")

    scaler = None
    with open('log_reg_scaler.pkl', 'rb') as f:
        scaler = pkl.load(f)
 
    routes.configure_routes(app,model,model_columns, scaler)

    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
