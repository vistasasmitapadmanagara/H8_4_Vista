from flask import Flask, render_template, request
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from io import BytesIO
import base64

model = joblib.load(open('model/model.pkl', 'rb'))

app = Flask(__name__, template_folder='templates')
    
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI IKG
    '''
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='The cab price is ${}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)