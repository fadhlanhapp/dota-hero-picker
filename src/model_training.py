#!/usr/bin/env python3
"""
Dota 2 Hero Picker - Model Training Module
Trains ML models to predict optimal hero picks
"""

import pandas as pd
import numpy as np
import json
import pickle
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import sys
sys.path.append('..')
from config import Config

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# For feature importance and analysis
import matplotlib.pyplot as plt
import seaborn as sns

class HeroPickPredictor:
    """Machine learning model for hero pick prediction"""
    
    def __init__(self, model_type='random_forest'):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.hero_names = {}
        
        # Training state
        self.is_trained = False
        self.training_summary = {}
        
        # Model options
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'class_weight': 'balanced'
                }
            }
        }
        
        self.logger.info(f"Initialized {model_type} predictor")
    
    def load_data(self, features_file: str, heroes_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training data and hero information"""
        self.logger.info(f"Loading data from {features_file}")
        
        # Load features
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        features_df = pd.read_csv(features_file)
        
        # Load heroes for name mapping
        if os.path.exists(heroes_file):
            with open(heroes_file, 'r') as f:
                heroes = json.load(f)
            self.hero_names = {h['id']: h['localized_name'] for h in heroes}
        
        self.logger.info(f"Loaded {len(features_df)} training samples")
        self.logger.info(f"Feature dimensions: {features_df.shape}")
        
        # Basic data info
        if 'target_hero' in features_df.columns:
            unique_heroes = features_df['target_hero'].nunique()
            self.logger.info(f"Unique target heroes: {unique_heroes}")
        
        return features_df, pd.DataFrame(heroes) if 'heroes' in locals() else pd.DataFrame()
    
    def prepare_features(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for ML training"""
        self.logger.info("Preparing features for ML training...")
        
        # Identify feature and target columns
        target_col = 'target_hero'
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in features")
        
        # Drop non-feature columns
        drop_cols = [
            'match_id', 'target_hero_name', 'team_hero_1_name', 'team_hero_2_name',
            'team_hero_3_name', 'team_hero_4_name', 'enemy_hero_1_name', 'enemy_hero_2_name',
            'enemy_hero_3_name', 'enemy_hero_4_name', 'team_side', 'pick_phase'
        ]
        
        # Keep only columns that exist
        drop_cols = [col for col in drop_cols if col in features_df.columns]
        
        # Prepare features (X) and target (y)
        X = features_df.drop(columns=drop_cols + [target_col])
        y = features_df[target_col]
        
        # Handle any remaining categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            self.logger.info(f"One-hot encoding categorical columns: {list(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols, prefix_sep='_')
        
        # Store feature columns for prediction
        self.feature_columns = X.columns.tolist()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.logger.info(f"Prepared features: {X.shape}")
        self.logger.info(f"Target classes: {len(self.label_encoder.classes_)}")
        
        return X, pd.Series(y_encoded, index=y.index)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """Split data into train/validation/test sets"""
        self.logger.info(f"Splitting data: train/val/test")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for reduced dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        self.logger.info(f"Train set: {X_train.shape[0]} samples")
        self.logger.info(f"Validation set: {X_val.shape[0]} samples")  
        self.logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train the ML model"""
        self.logger.info(f"Training {self.model_type} model...")
        
        # Get model configuration
        config = self.model_configs[self.model_type]
        model_class = config['model']
        params = config['params'].copy()
        
        # Initialize model
        self.model = model_class(**params)
        
        # Train model
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = datetime.now() - start_time
        
        # Validation predictions
        val_predictions = self.model.predict(X_val)
        val_probabilities = self.model.predict_proba(X_val) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_predictions, average='weighted')
        
        # Training summary
        training_summary = {
            'model_type': self.model_type,
            'training_time': str(training_time),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': X_train.shape[1],
            'target_classes': len(self.label_encoder.classes_),
            'validation_accuracy': val_accuracy,
            'validation_precision': val_precision,
            'validation_recall': val_recall,
            'validation_f1': val_f1,
            'model_params': params
        }
        
        self.training_summary = training_summary
        self.is_trained = True
        
        self.logger.info(f"✅ Training completed in {training_time}")
        self.logger.info(f"📊 Validation accuracy: {val_accuracy:.3f}")
        self.logger.info(f"📊 Validation F1: {val_f1:.3f}")
        
        return training_summary
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model on test set"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info("Evaluating model on test set...")
        
        # Test predictions
        test_predictions = self.model.predict(X_test)
        test_probabilities = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, test_predictions, average='weighted')
        
        # Detailed classification report
        hero_names_for_report = [self.hero_names.get(hero_id, f'Hero_{hero_id}') 
                                for hero_id in self.label_encoder.classes_]
        
        classification_rep = classification_report(
            y_test, test_predictions,
            target_names=hero_names_for_report,
            output_dict=True
        )
        
        # Evaluation summary
        evaluation_summary = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(y_test, test_predictions).tolist()
        }
        
        self.logger.info(f"📊 Test accuracy: {test_accuracy:.3f}")
        self.logger.info(f"📊 Test F1: {test_f1:.3f}")
        
        return evaluation_summary
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get feature importance analysis"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if not hasattr(self.model, 'feature_importances_'):
            self.logger.warning("Model doesn't support feature importance")
            return {}
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Top features
        top_features = feature_importance.head(top_n)
        
        self.logger.info(f"Top {top_n} most important features:")
        for _, row in top_features.iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'top_features': top_features.to_dict('records'),
            'all_features': feature_importance.to_dict('records')
        }
    
    def predict_best_heroes(self, team_heroes: List[int], enemy_heroes: List[int], top_k: int = 5) -> List[Dict]:
        """Predict best heroes for given team composition"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get all possible target heroes
        all_possible_heroes = set(self.label_encoder.classes_)
        picked_heroes = set(team_heroes + enemy_heroes)
        available_heroes = all_possible_heroes - picked_heroes
        
        if not available_heroes:
            self.logger.warning("No available heroes for prediction")
            return []
        
        predictions = []
        
        for hero_id in available_heroes:
            try:
                # Create feature vector for this scenario
                feature_vector = self._create_prediction_features(team_heroes, enemy_heroes, hero_id)
                
                if feature_vector is not None:
                    # Predict probability
                    probabilities = self.model.predict_proba([feature_vector])[0]
                    
                    # Find probability for this hero
                    hero_encoded = self.label_encoder.transform([hero_id])[0]
                    win_probability = probabilities[hero_encoded] if hero_encoded < len(probabilities) else 0.0
                    
                    predictions.append({
                        'hero_id': int(hero_id),
                        'hero_name': self.hero_names.get(hero_id, f'Hero_{hero_id}'),
                        'win_probability': float(win_probability),
                        'confidence': float(max(probabilities))  # Overall confidence
                    })
                    
            except Exception as e:
                self.logger.debug(f"Failed to predict for hero {hero_id}: {e}")
                continue
        
        # Sort by win probability
        predictions.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return predictions[:top_k]
    
    def _create_prediction_features(self, team_heroes: List[int], enemy_heroes: List[int], target_hero: int) -> Optional[np.ndarray]:
        """Create feature vector for prediction"""
        try:
            # This is a simplified version - you'd need to reconstruct the exact features
            # used during training based on your feature engineering
            
            features = {}
            
            # Basic composition features
            for i in range(4):
                features[f'team_hero_{i+1}'] = team_heroes[i] if i < len(team_heroes) else 0
                features[f'enemy_hero_{i+1}'] = enemy_heroes[i] if i < len(enemy_heroes) else 0
            
            # Add other features that were used in training
            features['team_size'] = len(team_heroes)
            features['enemy_size'] = len(enemy_heroes)
            features['total_heroes_picked'] = len(team_heroes) + len(enemy_heroes)
            
            # Convert to format expected by model
            feature_vector = np.zeros(len(self.feature_columns))
            
            # This is simplified - in practice you'd need to reconstruct
            # the exact one-hot encoding used during training
            
            return feature_vector
            
        except Exception as e:
            self.logger.debug(f"Error creating features: {e}")
            return None
    
    def save_model(self, filepath: str):
        """Save trained model and metadata"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'hero_names': self.hero_names,
            'model_type': self.model_type,
            'training_summary': self.training_summary,
            'is_trained': self.is_trained
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save with joblib (more efficient for sklearn models)
        joblib.dump(model_data, filepath)
        
        self.logger.info(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and metadata"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.hero_names = model_data['hero_names']
        self.model_type = model_data['model_type']
        self.training_summary = model_data['training_summary']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"📂 Model loaded from {filepath}")
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"Training accuracy: {self.training_summary.get('validation_accuracy', 'N/A'):.3f}")

def create_training_pipeline(features_file: str, heroes_file: str, model_type: str = 'random_forest') -> Dict:
    """Complete training pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting model training pipeline...")
        
        # Initialize predictor
        predictor = HeroPickPredictor(model_type=model_type)
        
        # Load data
        features_df, heroes_df = predictor.load_data(features_file, heroes_file)
        
        # Prepare features
        X, y = predictor.prepare_features(features_df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(X, y)
        
        # Train model
        training_summary = predictor.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        evaluation_summary = predictor.evaluate_model(X_test, y_test)
        
        # Feature importance
        importance_summary = predictor.get_feature_importance()
        
        # Save model
        model_path = f"{predictor.config.MODELS_DIR}/hero_predictor_{model_type}.pkl"
        predictor.save_model(model_path)
        
        # Combine all results
        complete_summary = {
            'training': training_summary,
            'evaluation': evaluation_summary,
            'feature_importance': importance_summary,
            'model_path': model_path
        }
        
        # Save summary
        summary_path = f"{predictor.config.PROCESSED_DIR}/training_summary_{model_type}.json"
        with open(summary_path, 'w') as f:
            json.dump(complete_summary, f, indent=2, default=str)
        
        logger.info("✅ Training pipeline completed successfully!")
        logger.info(f"📊 Final test accuracy: {evaluation_summary['test_accuracy']:.3f}")
        logger.info(f"💾 Model saved to: {model_path}")
        logger.info(f"📋 Summary saved to: {summary_path}")
        
        return complete_summary
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise

def main():
    """CLI interface for model training"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Dota 2 hero pick prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_training.py                                    # Train Random Forest
  python model_training.py --model gradient_boosting         # Train Gradient Boosting
  python model_training.py --features data/processed/features_ml.csv  # Custom features file
  python model_training.py --compare                         # Train and compare all models
        """
    )
    
    parser.add_argument(
        '--features',
        default='data/processed/features_ml.csv',
        help='Path to features file'
    )
    
    parser.add_argument(
        '--heroes',
        default='data/raw/heroes.json',
        help='Path to heroes file'
    )
    
    parser.add_argument(
        '--model',
        choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
        default='random_forest',
        help='Model type to train'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Train and compare all model types'
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Train all models and compare
            print("🔄 Training and comparing all models...")
            
            results = {}
            for model_type in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                print(f"\n📊 Training {model_type}...")
                summary = create_training_pipeline(args.features, args.heroes, model_type)
                results[model_type] = summary['evaluation']['test_accuracy']
            
            # Show comparison
            print("\n📈 Model Comparison Results:")
            for model, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
                print(f"  {model}: {accuracy:.3f}")
            
        else:
            # Train single model
            summary = create_training_pipeline(args.features, args.heroes, args.model)
            
            print(f"\n🎉 Training completed!")
            print(f"Model: {args.model}")
            print(f"Test Accuracy: {summary['evaluation']['test_accuracy']:.3f}")
            print(f"Model saved to: {summary['model_path']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())