#!/usr/bin/env python3
"""
Dota 2 Hero Picker Flask Web Application
"""

import os
import sys
import json
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor_service import HeroPredictorService
from config import Config

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize predictor service
try:
    predictor = HeroPredictorService()
    logger.info("Predictor service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor service: {e}")
    predictor = None

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if predictor:
        return jsonify(predictor.health_check())
    else:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Predictor service not initialized',
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/heroes', methods=['GET'])
def get_heroes():
    """Get all available heroes"""
    try:
        if not predictor:
            return jsonify({'error': 'Service not available'}), 503
        
        heroes = predictor.get_all_heroes()
        return jsonify({
            'success': True,
            'heroes': heroes,
            'count': len(heroes)
        })
    
    except Exception as e:
        logger.error(f"Error getting heroes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_heroes():
    """
    Predict best heroes based on current draft
    
    Expected JSON payload:
    {
        "team_heroes": [1, 2, 3],  // List of hero IDs on your team
        "enemy_heroes": [4, 5, 6],  // List of hero IDs on enemy team
        "top_k": 5,  // Number of recommendations (optional, default 5)
        "skill_level": "High",  // Skill bracket filter (optional)
        "patch": "7.35"  // Game patch filter (optional)
    }
    """
    try:
        if not predictor:
            return jsonify({'error': 'Service not available'}), 503
        
        # Parse request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Extract parameters
        team_heroes = data.get('team_heroes', [])
        enemy_heroes = data.get('enemy_heroes', [])
        top_k = data.get('top_k', 5)
        skill_level = data.get('skill_level', None)
        patch = data.get('patch', None)
        
        # Validate input
        if not isinstance(team_heroes, list) or not isinstance(enemy_heroes, list):
            return jsonify({
                'success': False,
                'error': 'team_heroes and enemy_heroes must be lists'
            }), 400
        
        # Validate hero counts
        if len(team_heroes) > 5:
            return jsonify({
                'success': False,
                'error': 'Maximum 5 heroes per team'
            }), 400
        
        if len(enemy_heroes) > 5:
            return jsonify({
                'success': False,
                'error': 'Maximum 5 heroes per team'
            }), 400
        
        # Validate that heroes aren't duplicated
        all_heroes = team_heroes + enemy_heroes
        if len(all_heroes) != len(set(all_heroes)):
            return jsonify({
                'success': False,
                'error': 'Duplicate heroes detected'
            }), 400
        
        # Make prediction with filters
        recommendations = predictor.predict_heroes(team_heroes, enemy_heroes, top_k, skill_level, patch)
        
        # Prepare response
        response = {
            'success': True,
            'recommendations': recommendations,
            'team_size': len(team_heroes),
            'enemy_size': len(enemy_heroes),
            'filters': {
                'skill_level': skill_level,
                'patch': patch
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made for team={team_heroes}, enemy={enemy_heroes}")
        return jsonify(response)
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/hero/<int:hero_id>', methods=['GET'])
def get_hero(hero_id):
    """Get specific hero information"""
    try:
        if not predictor:
            return jsonify({'error': 'Service not available'}), 503
        
        hero = predictor.get_hero_by_id(hero_id)
        
        if hero:
            return jsonify({
                'success': True,
                'hero': hero
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Hero with ID {hero_id} not found'
            }), 404
    
    except Exception as e:
        logger.error(f"Error getting hero {hero_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Get available skill levels, patches, and other metadata"""
    try:
        if not predictor:
            return jsonify({'error': 'Service not available'}), 503
        
        metadata = predictor.get_metadata()
        return jsonify({
            'success': True,
            'metadata': metadata
        })
    
    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

def main():
    """Run the Flask application"""
    config = Config()
    
    # Check if model exists
    model_path = f"{config.MODELS_DIR}/hero_predictor_random_forest_fixed.pkl"
    if not os.path.exists(model_path):
        model_path = f"{config.MODELS_DIR}/hero_predictor_random_forest.pkl"
        if not os.path.exists(model_path):
            logger.warning("No trained model found. Please train a model first.")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    logger.info(f"Starting Dota 2 Hero Picker on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Access the application at http://localhost:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )

if __name__ == '__main__':
    main()