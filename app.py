from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model from reference book 
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the saved LabelEncoders again from reference book malli
with open('le_service.pkl', 'rb') as f:
    le_service = pickle.load(f)
with open('le_state.pkl', 'rb') as f:
    le_state = pickle.load(f)
with open('le_protocol.pkl', 'rb') as f:
    le_protocol = pickle.load(f)
with open('le_attack.pkl', 'rb') as f:
    le_attack = pickle.load(f)

# top 15 features from reference book
features = [
    'ct_dst_sport_ltm', 'ct_srv_dst', 'ct_srv_src', 'ct_dst_src_ltm', 
    'ct_src_dport_ltm', 'ct_dst_ltm', 'ct_src_ltm', 'destination time to live', 
    'dwin', 'swin', 'state', 'tcprtt', 'source time to live', 'ackdat', 'synack'
]

# initializing our friendly names as reference book lo done
feature_display_names = {
    'ct_dst_sport_ltm': 'Destination Port Connections (Last Time)',
    'ct_srv_dst': 'Destination Service Connections',
    'ct_srv_src': 'Source Service Connections',
    'ct_dst_src_ltm': 'Source-Destination Connections (Last Time)',
    'ct_src_dport_ltm': 'Source Port Connections (Last Time)',
    'ct_dst_ltm': 'Destination Connections (Last Time)',
    'ct_src_ltm': 'Source Connections (Last Time)',
    'destination time to live': 'Destination Packet Lifespan',
    'dwin': 'Destination TCP Window Size',
    'swin': 'Source TCP Window Size',
    'state': 'Connection State',
    'tcprtt': 'TCP Round-Trip Time',
    'source time to live': 'Source Packet Lifespan',
    'ackdat': 'TCP Acknowledgment Delay',
    'synack': 'TCP SYN-ACK Delay',
    'protocol': 'Network Protocol',
    'service': 'Network Service'
}

# Define categorical and numerical features
categorical_features = ['state']
numerical_features = [f for f in features if f not in categorical_features]

# user friendly for ellimudhra gaalaki understandable
label_map = {
    'Fuzzers': 'Unauthorized Login Attempts',
    'Dos': 'DOS Attacks',
    'Exploits': 'Malware',
    'Generic': 'Data Leaks'
}

@app.route('/')
def home():

    return render_template('index.html', 
                         features=features, 
                         display_names=feature_display_names,
                         protocols=le_protocol.classes_.tolist(),
                         services=le_service.classes_.tolist(),
                         states=le_state.classes_.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # district Collector colleciting user inputs
        input_data = {}
        for feature in features:
            value = request.form.get(feature)
            if value is None:
                raise ValueError(f"Missing required field: {feature}")
            input_data[feature] = value

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Preprocess (beautyparlor) categorical columns
        for feature in categorical_features:
            if feature in input_df.columns:
                try:
                    input_df[feature] = globals()[f'le_{feature}'].transform(input_df[feature])
                except ValueError as e:
                    raise ValueError(f"Invalid value for {feature}: {input_df[feature].iloc[0]}. Must be one of {globals()[f'le_{feature}'].classes_}")

        # Convert numerical features to float
        for feature in numerical_features:
            try:
                input_df[feature] = input_df[feature].astype(float)
            except ValueError as e:
                raise ValueError(f"Invalid numerical value for {feature}: {input_df[feature].iloc[0]}")

    
        input_df = input_df[features]

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        predicted_attack = le_attack.inverse_transform([prediction])[0]
    
        predicted_attack = label_map.get(predicted_attack, predicted_attack)

        return jsonify({
            'prediction': predicted_attack,
            'confidence': f"{max(prediction_proba) * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)