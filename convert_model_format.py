#!/usr/bin/env python3
"""
Convert model to a format compatible across scikit-learn versions
This creates a simpler model file that works across versions
"""

import pickle
import json
import numpy as np

def save_model_portable(model_path='models/hero_predictor_categorical.pkl'):
    """
    Save model in a portable format that works across sklearn versions
    """
    print("Converting model to portable format...")
    
    try:
        # Try to load the model (might fail due to version mismatch)
        import joblib
        model_data = joblib.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Can't load model due to version mismatch: {e}")
        print("\nPlease run this in Google Colab instead:")
        print(create_colab_export_script())
        return
    
    # Extract the essential components
    portable_model = {
        'model_type': 'categorical_predictor',
        'label_classes': model_data['label_encoder'].classes_.tolist(),
        'feature_names': [],
        'preprocessor_config': None
    }
    
    # Save in JSON format (universal)
    with open('models/model_config.json', 'w') as f:
        json.dump(portable_model, f)
    
    print("Portable model config saved to models/model_config.json")
    

def create_colab_export_script():
    """Generate script to run in Colab for exporting model"""
    return '''
# Run this in Google Colab to export your model in a portable format

import joblib
import json
import numpy as np
import pickle

# Load the model
model_data = joblib.load('models/hero_predictor_categorical.pkl')

# Create a simplified predictor class
class SimpleHeroPredictor:
    def __init__(self, model, label_encoder, preprocessor):
        self.model = model
        self.label_encoder = label_encoder
        self.preprocessor = preprocessor
        self.classes_ = label_encoder.classes_
        
    def predict_proba(self, X):
        """Predict probabilities"""
        X_transformed = self.preprocessor.transform(X)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        return self.model.predict_proba(X_transformed)
    
    def predict(self, X):
        """Predict single class"""
        X_transformed = self.preprocessor.transform(X)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        return self.model.predict(X_transformed)

# Create simplified model
simple_model = SimpleHeroPredictor(
    model_data['model'],
    model_data['label_encoder'],
    model_data['preprocessor']
)

# Save using pickle protocol 4 (compatible with Python 3.4+)
with open('models/hero_predictor_simple.pkl', 'wb') as f:
    pickle.dump(simple_model, f, protocol=4)

print("Simplified model saved to models/hero_predictor_simple.pkl")

# Also save configuration separately
config = {
    'classes': model_data['label_encoder'].classes_.tolist(),
    'n_features': model_data['preprocessor'].transform([[0]*17]).shape[1]
}

with open('models/model_config.json', 'w') as f:
    json.dump(config, f)

print("Configuration saved to models/model_config.json")

# Download the files
from google.colab import files
files.download('models/hero_predictor_simple.pkl')
files.download('models/model_config.json')
'''

if __name__ == "__main__":
    save_model_portable()