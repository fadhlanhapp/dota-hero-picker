#!/usr/bin/env python3
"""
Dota 2 Hero Predictor Service
Handles model loading and prediction logic
"""

import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class HeroPredictorService:
    """Service for making hero predictions using trained models"""
    
    def __init__(self, model_path: str = None, heroes_path: str = None):
        """Initialize the predictor service"""
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        
        # Set paths
        self.model_path = model_path or f"{self.config.MODELS_DIR}/hero_predictor_random_forest_fixed.pkl"
        if not os.path.exists(self.model_path):
            # Fallback to original model if fixed version doesn't exist
            self.model_path = f"{self.config.MODELS_DIR}/hero_predictor_random_forest.pkl"
        
        self.heroes_path = heroes_path or f"{self.config.RAW_DIR}/heroes.json"
        
        # Model components
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_columns = []
        self.hero_names = {}
        self.hero_data = []
        self.all_hero_ids = set()
        
        # Load model and data
        self._load_model()
        self._load_heroes()
        
        self.logger.info("Predictor service initialized successfully")
    
    def _load_model(self):
        """Load the trained model and its components"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.logger.info(f"Loading model from {self.model_path}")
            model_data = joblib.load(self.model_path)
            
            # Extract model components
            self.model = model_data.get('model')
            self.label_encoder = model_data.get('label_encoder')
            self.preprocessor = model_data.get('preprocessor')
            self.feature_columns = model_data.get('feature_columns', [])
            self.hero_names = model_data.get('hero_names', {})
            
            # Check if model is trained
            if not model_data.get('is_trained', False):
                raise ValueError("Model is not trained")
            
            self.logger.info(f"Model loaded successfully. Type: {model_data.get('model_type', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_heroes(self):
        """Load hero data from JSON file"""
        try:
            if not os.path.exists(self.heroes_path):
                self.logger.warning(f"Heroes file not found: {self.heroes_path}")
                return
            
            with open(self.heroes_path, 'r') as f:
                self.hero_data = json.load(f)
            
            # Create hero lookup tables
            for hero in self.hero_data:
                hero_id = hero['id']
                self.hero_names[hero_id] = hero['localized_name']
                self.all_hero_ids.add(hero_id)
            
            self.logger.info(f"Loaded {len(self.hero_data)} heroes")
            
        except Exception as e:
            self.logger.error(f"Failed to load heroes: {e}")
    
    def get_all_heroes(self) -> List[Dict]:
        """Get list of all heroes with their data"""
        heroes_list = []
        
        for hero in self.hero_data:
            hero_info = {
                'id': hero['id'],
                'name': hero['localized_name'],
                'primary_attr': hero.get('primary_attr', 'unknown'),
                'attack_type': hero.get('attack_type', 'unknown'),
                'roles': hero.get('roles', []),
                'img': f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{hero['name'].replace('npc_dota_hero_', '')}_lg.png"
            }
            heroes_list.append(hero_info)
        
        return sorted(heroes_list, key=lambda x: x['name'])
    
    def _create_features(self, team_heroes: List[int], enemy_heroes: List[int]) -> pd.DataFrame:
        """Create feature vector matching the training format"""
        # Initialize feature dictionary
        features = {}
        
        # Basic features
        features['duration'] = 1800  # Average game duration in seconds (30 min)
        features['team_won'] = 0  # Neutral for prediction
        features['team_size'] = len(team_heroes)
        features['enemy_size'] = len(enemy_heroes)
        
        # Team and enemy hero slots (pad with 0 for empty slots)
        for i in range(4):
            features[f'team_hero_{i+1}'] = str(team_heroes[i]) if i < len(team_heroes) else 'none'
            features[f'enemy_hero_{i+1}'] = str(enemy_heroes[i]) if i < len(enemy_heroes) else 'none'
        
        # Role counts (simplified - in production, use actual role data)
        features['team_carry_count'] = 0
        features['enemy_carry_count'] = 0
        features['target_is_carry'] = 0
        features['team_support_count'] = 0
        features['enemy_support_count'] = 0
        features['target_is_support'] = 0
        features['team_tank_count'] = 0
        features['enemy_tank_count'] = 0
        features['target_is_tank'] = 0
        features['team_mid_count'] = 0
        features['enemy_mid_count'] = 0
        features['target_is_mid'] = 0
        features['team_versatile_count'] = len(team_heroes)
        features['enemy_versatile_count'] = len(enemy_heroes)
        features['target_is_versatile'] = 1
        
        # Strategic features
        features['team_role_diversity'] = 1
        features['enemy_role_diversity'] = 1
        features['synergy_count'] = 0
        features['has_synergy'] = 0
        features['countered_enemies'] = 0
        features['counter_strength'] = 0
        features['is_counter_pick'] = 0
        features['total_heroes_picked'] = len(team_heroes) + len(enemy_heroes)
        
        # Categorical features
        features['team_side'] = 'radiant'
        features['pick_phase'] = 'early' if features['total_heroes_picked'] < 4 else 'mid' if features['total_heroes_picked'] < 8 else 'late'
        features['pick_position'] = min(len(team_heroes) + 1, 5)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        return features_df
    
    def predict_heroes(self, team_heroes: List[int], enemy_heroes: List[int], top_k: int = 5) -> List[Dict]:
        """
        Predict best heroes for the given team composition
        
        Args:
            team_heroes: List of hero IDs on your team
            enemy_heroes: List of hero IDs on enemy team
            top_k: Number of top recommendations to return
        
        Returns:
            List of recommended heroes with probabilities
        """
        try:
            # Validate input
            team_heroes = [h for h in team_heroes if h in self.all_hero_ids]
            enemy_heroes = [h for h in enemy_heroes if h in self.all_hero_ids]
            
            # Get already picked heroes
            picked_heroes = set(team_heroes + enemy_heroes)
            available_heroes = self.all_hero_ids - picked_heroes
            
            if not available_heroes:
                return []
            
            predictions = []
            
            # For each available hero, calculate win probability
            for hero_id in available_heroes:
                try:
                    # Create features for this scenario
                    features_df = self._create_features(team_heroes, enemy_heroes)
                    
                    # Apply preprocessing if available
                    if self.preprocessor is not None:
                        # Handle categorical encoding
                        features_processed = self.preprocessor.transform(features_df)
                        
                        # Convert to DataFrame with correct column names
                        if hasattr(self.preprocessor, 'get_feature_names_out'):
                            feature_names = self.preprocessor.get_feature_names_out()
                        else:
                            feature_names = self.feature_columns
                        
                        if len(feature_names) > 0:
                            features_processed = pd.DataFrame(
                                features_processed, 
                                columns=feature_names[:features_processed.shape[1]]
                            )
                    else:
                        features_processed = features_df
                    
                    # Make prediction
                    if hasattr(self.model, 'predict_proba'):
                        # Get probability for each class
                        probabilities = self.model.predict_proba(features_processed)[0]
                        
                        # Find if this hero is in the label encoder
                        if hero_id in self.label_encoder.classes_:
                            hero_idx = np.where(self.label_encoder.classes_ == hero_id)[0][0]
                            win_prob = probabilities[hero_idx] if hero_idx < len(probabilities) else 0.0
                        else:
                            # Hero not in training data, use average probability
                            win_prob = np.mean(probabilities)
                    else:
                        # Model doesn't support probabilities, use binary prediction
                        prediction = self.model.predict(features_processed)[0]
                        win_prob = 1.0 if prediction == hero_id else 0.0
                    
                    # Get hero info
                    hero_name = self.hero_names.get(hero_id, f'Hero_{hero_id}')
                    hero_info = next((h for h in self.hero_data if h['id'] == hero_id), {})
                    
                    predictions.append({
                        'hero_id': int(hero_id),
                        'hero_name': hero_name,
                        'win_probability': float(win_prob),
                        'confidence': float(win_prob * 100),
                        'primary_attr': hero_info.get('primary_attr', 'unknown'),
                        'roles': hero_info.get('roles', []),
                        'img': f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{hero_info.get('name', '').replace('npc_dota_hero_', '')}_lg.png"
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Failed to predict for hero {hero_id}: {e}")
                    continue
            
            # Sort by win probability and return top k
            predictions.sort(key=lambda x: x['win_probability'], reverse=True)
            
            # If no good predictions, return random available heroes
            if not predictions or all(p['win_probability'] < 0.01 for p in predictions):
                self.logger.warning("No confident predictions, returning random heroes")
                random_heroes = list(available_heroes)[:top_k]
                predictions = []
                for hero_id in random_heroes:
                    hero_name = self.hero_names.get(hero_id, f'Hero_{hero_id}')
                    hero_info = next((h for h in self.hero_data if h['id'] == hero_id), {})
                    predictions.append({
                        'hero_id': int(hero_id),
                        'hero_name': hero_name,
                        'win_probability': 0.2,  # Default probability
                        'confidence': 20.0,
                        'primary_attr': hero_info.get('primary_attr', 'unknown'),
                        'roles': hero_info.get('roles', []),
                        'img': f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{hero_info.get('name', '').replace('npc_dota_hero_', '')}_lg.png"
                    })
            
            return predictions[:top_k]
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def get_hero_by_id(self, hero_id: int) -> Optional[Dict]:
        """Get hero information by ID"""
        for hero in self.hero_data:
            if hero['id'] == hero_id:
                return {
                    'id': hero['id'],
                    'name': hero['localized_name'],
                    'primary_attr': hero.get('primary_attr', 'unknown'),
                    'attack_type': hero.get('attack_type', 'unknown'),
                    'roles': hero.get('roles', []),
                    'img': f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{hero['name'].replace('npc_dota_hero_', '')}_lg.png"
                }
        return None
    
    def health_check(self) -> Dict:
        """Check if service is healthy and return status"""
        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'heroes_loaded': len(self.hero_data) > 0,
            'total_heroes': len(self.hero_data),
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat()
        }

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    service = HeroPredictorService()
    
    # Test prediction
    team_heroes = [1, 2, 3]  # Example hero IDs
    enemy_heroes = [4, 5, 6]
    
    recommendations = service.predict_heroes(team_heroes, enemy_heroes)
    
    print("Top 5 Hero Recommendations:")
    for i, hero in enumerate(recommendations, 1):
        print(f"{i}. {hero['hero_name']} - {hero['confidence']:.1f}% confidence")