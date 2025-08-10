#!/bin/bash

echo "🚀 Starting Dota 2 Hero Picker Web Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements_web.txt

# Check if model exists
if [ ! -f "models/hero_predictor_random_forest.pkl" ] && [ ! -f "models/hero_predictor_random_forest_fixed.pkl" ]; then
    echo "⚠️  Warning: No trained model found in models/ directory"
    echo "Please train a model first using: python3 src/model_training.py"
fi

# Start the Flask application
echo "Starting Flask server..."
echo "Access the application at: http://localhost:5000"
python3 src/app.py