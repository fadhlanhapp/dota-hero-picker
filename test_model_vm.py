#!/usr/bin/env python3
"""
Test the categorical model in VM
"""

import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_model():
    print("Testing model loading and prediction...\n")
    
    try:
        # Load model
        print("1. Loading model...")
        model_data = joblib.load('models/hero_predictor_categorical.pkl')
        print("   ✓ Model loaded successfully")
        
        # Extract components
        model = model_data.get('model')
        preprocessor = model_data.get('preprocessor')
        label_encoder = model_data.get('label_encoder')
        
        print(f"   Model type: {type(model).__name__}")
        print(f"   Classes: {len(label_encoder.classes_)} heroes")
        
        # Create test data
        print("\n2. Creating test data...")
        test_data = {
            'pick_number': 3,
            'team_picking': 1,
            'skill': 0,
            'avg_mmr': 3500,
            'patch': 58,
            'radiant_pick_1': '14',
            'dire_pick_1': '26',
            'radiant_pick_2': 'none',
            'dire_pick_2': 'none',
            'radiant_pick_3': 'none',
            'dire_pick_3': 'none',
            'radiant_pick_4': 'none',
            'dire_pick_4': 'none',
            'radiant_pick_5': 'none',
            'dire_pick_5': 'none',
            'ally_picks_count': 1,
            'enemy_picks_count': 1
        }
        
        X = pd.DataFrame([test_data])
        print("   ✓ Test data created")
        
        # Transform data
        print("\n3. Transforming data...")
        X_transformed = preprocessor.transform(X)
        
        # Handle sparse matrix
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
            print(f"   ✓ Sparse matrix converted to dense")
        
        print(f"   Transformed shape: {X_transformed.shape}")
        
        # Make prediction
        print("\n4. Making prediction...")
        probabilities = model.predict_proba(X_transformed)[0]
        
        # Get top 10 predictions
        top_indices = np.argsort(probabilities)[-10:][::-1]
        top_heroes = label_encoder.inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        print("\n✅ TOP 10 HERO PREDICTIONS:")
        print("-" * 40)
        for i, (hero_id, prob) in enumerate(zip(top_heroes, top_probs), 1):
            print(f"{i:2d}. Hero {hero_id:3d}: {prob*100:5.2f}%")
        
        print("\n✅ Model is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    
    if success:
        print("\n🎉 You can now use the model in your web app!")
        print("Run: python3 src/app.py")
    else:
        print("\n⚠️ Please check the error above and fix it.")