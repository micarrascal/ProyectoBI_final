from flask import Flask, request, jsonify, make_response
import joblib
import pandas as pd
from flask_cors import CORS
import io
from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from utils import TextCleaner
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin

app = Flask(__name__)
CORS(app)  # Permitir solicitudes de origen cruzado

modelo = joblib.load('./modelo_entrenado.pkl')
pipeline = joblib.load('./mi_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            data = [data['Textos_espanol']]
            data = pd.DataFrame(data, columns=['Textos_espanol'])
            data['Textos_espanol'] = data['Textos_espanol'].astype(str)
            newdata = pipeline.transform(data['Textos_espanol'])
            prediction = modelo.predict(newdata)
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        df = pd.read_csv(file, delimiter=';') 
        if 'Textos_espanol' not in df.columns:
            return jsonify({'error': 'Invalid CSV format'})
            print(type(df))
        df['Textos_espanol'] = df['Textos_espanol'].astype(str)
        newdata = pipeline.transform(df['Textos_espanol'])
        prediction = modelo.predict(newdata)
        
        # Crear un DataFrame para exportar
        result_df = pd.DataFrame({
            'sdg': prediction
        })

        # Convertir el DataFrame a CSV y enviarlo como respuesta
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=prediction_results.csv"
        response.headers["Content-type"] = "text/csv"
        return response

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
