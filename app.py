import pickle
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from io import StringIO

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model and scalers
model = pickle.load(open('model.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
stand_scaler = pickle.load(open('standscaler.pkl', 'rb'))

# Azure Blob Storage configuration
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
container_name = 'cropdata'

# Set your Azure Blob Storage connection string and container name
blob_service_client = BlobServiceClient.from_connection_string(conn_str=connect_str)
container_client = blob_service_client.get_container_client(container=container_name)

# Crop dictionary for mapping predictions
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

def download_csv():
    try:
        blob_client = container_client.get_blob_client("predictions.csv")
        csv_data = blob_client.download_blob().readall().decode('utf-8')
        return pd.read_csv(StringIO(csv_data))
    except Exception as e:
        return pd.DataFrame(columns=['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Prediction'])

def upload_csv(df):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob_client = container_client.get_blob_client("predictions.csv")
    blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    records = []

    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Combine the input values into a single numpy array
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Apply scaling
        features_scaled = minmax_scaler.transform(features)
        features_standardized = stand_scaler.transform(features_scaled)

        # Make prediction
        prediction = model.predict(features_standardized).reshape(1, -1)[0][0]
        
        # Get the corresponding crop name
        crop = crop_dict.get(prediction, "Unknown Crop")

        prediction_text = f"The best crop to be cultivated is: {crop}"

        # Load existing data
        data = download_csv()

        # Append new data
        new_record = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall, crop]], 
                                  columns=['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Prediction'])
        data = pd.concat([data, new_record], ignore_index=True)

        # Upload updated data back to Azure Blob Storage
        upload_csv(data)

    # Load the data for display
    data = download_csv()
    records = data.to_dict(orient='records')

    return render_template('index.html', prediction_text=prediction_text, records=records)

@app.route('/delete_data/<int:record_id>', methods=['POST'])
def delete_data(record_id):
    data = download_csv()
    if 0 <= record_id < len(data):
        data = data.drop(index=record_id)  # Drop the specific row
        upload_csv(data)  # Upload the updated data back to Azure Blob Storage
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
