from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)


scaler_path = r'C:\aiml project\Model\predectionmodel.pkl'
model_path = r'C:\aiml project\Model\standardScalar.pkl'

scaler = pickle.load(open(scaler_path, "rb"))
model = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def make_prediction():
    result = ""
    
    if request.method == 'POST':
        
        temp = float(request.form['temperature'])
        rh = float(request.form['RH'])
        wind_speed = float(request.form['Ws'])
        rainfall = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        dc = float(request.form['DC'])
        isi = float(request.form['ISI'])
        bui = float(request.form['BUI'])
        fwi = float(request.form['FWI'])
        
       
        input_data = np.array([[temp, rh, wind_speed, rainfall, ffmc, dmc, dc, isi, bui, fwi]])
        
       
        scaled_input = scaler.transform(input_data)
        
       
        prediction = model.predict(scaled_input)
     
        if prediction[0] == 1:
            result = "FIRE DETECTED"
        else:
            result = "NO FIRE"
        
        return render_template('result.html', prediction=result)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
