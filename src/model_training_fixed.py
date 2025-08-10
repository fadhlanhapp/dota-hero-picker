#!/usr/bin/env python3
"""
Fixed Dota 2 Hero Picker - Model Training Module
Properly handles categorical features and class imbalance
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, top_k_accuracy_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

class HeroPickPredictorFixed:
    """Fixed ML model for hero pick prediction with proper feature handling"""
    
    def __init__(self, model_type='random_forest'):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.feature_columns = []
        self.hero_names = {}
        
        # Training state
        self.is_trained = False
        self.training_summary = {}
        
        # Model options with better hyperparameters for multiclass
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 20,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1,
                    'class_weight': 'balanced_subsample'
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 10,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'subsample': 0.8,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 2000,
                    'class_weight': 'balanced',
                    'solver': 'saga',
                    'penalty': 'l2',
                    'C': 0.1,
                    'multi_class': 'multinomial'
                }
            }
        }
        
        self.logger.info(f"Initialized fixed {model_type} predictor")
    
    def load_data(self, features_file: str, heroes_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training data and hero information"""
        self.logger.info(f"Loading data from {features_file}")
        
        # Load features - use raw features, not ML features
        if 'features_ml.csv' in features_file:
            features_file = features_file.replace('features_ml.csv', 'features_raw.csv')
            self.logger.info(f"Using raw features from {features_file}")
        
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        features_df = pd.read_csv(features_file)
        
        # Load heroes for name mapping
        if os.path.exists(heroes_file):
            with open(heroes_file, 'r') as f:
                heroes = json.load(f)
            self.hero_names = {h['id']: h['localized_name'] for h in heroes}
            # Get all valid hero IDs
            self.all_hero_ids = set(h['id'] for h in heroes)
        
        self.logger.info(f"Loaded {len(features_df)} training samples")
        self.logger.info(f"Feature dimensions: {features_df.shape}")
        
        # Filter out invalid hero IDs
        if 'target_hero' in features_df.columns and hasattr(self, 'all_hero_ids'):
            valid_mask = features_df['target_hero'].isin(self.all_hero_ids)
            features_df = features_df[valid_mask]
            self.logger.info(f"After filtering invalid heroes: {len(features_df)} samples")
        
        # Basic data info
        if 'target_hero' in features_df.columns:
            unique_heroes = features_df['target_hero'].nunique()
            self.logger.info(f"Unique target heroes: {unique_heroes}")
        
        return features_df, pd.DataFrame(heroes) if 'heroes' in locals() else pd.DataFrame()
    
    def prepare_features(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for ML training with proper categorical encoding"""
        self.logger.info("Preparing features with proper categorical handling...")
        
        # Identify feature and target columns
        target_col = 'target_hero'
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in features")
        
        # Columns to drop
        drop_cols = ['match_id', 'target_hero_name']
        # Drop all name columns (we'll use IDs)
        name_cols = [col for col in features_df.columns if col.endswith('_name')]
        drop_cols.extend(name_cols)
        
        # Keep only columns that exist
        drop_cols = [col for col in drop_cols if col in features_df.columns]
        
        # Prepare features (X) and target (y)
        X = features_df.drop(columns=drop_cols + [target_col])
        y = features_df[target_col]
        
        # Identify categorical columns (hero IDs and other categoricals)
        hero_cols = [col for col in X.columns if 'hero_' in col and any(col.endswith(f'_{i}') for i in range(1, 5))]
        categorical_cols = ['team_side', 'pick_phase', 'pick_position'] + hero_cols
        categorical_cols = [col for col in categorical_cols if col in X.columns]
        
        # Identify numerical columns
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        self.logger.info(f"Categorical columns: {categorical_cols}")
        self.logger.info(f"Numerical columns: {numerical_cols}")
        
        # Create preprocessor with proper one-hot encoding
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        # For hero columns, handle missing values (0 means no hero picked yet)
        # Replace 0 with a special category
        for col in hero_cols:
            if col in X.columns:
                X[col] = X[col].replace(0, 'none').astype(str)
        
        # Convert other categoricals to string
        for col in ['team_side', 'pick_phase', 'pick_position']:
            if col in X.columns:
                X[col] = X[col].astype(str)
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ])
        
        # Fit and transform features
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        cat_features = []
        if categorical_cols:
            cat_features = list(self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
        feature_names = numerical_cols + cat_features
        
        # Convert to DataFrame for better handling
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        
        # Store feature columns for prediction
        self.feature_columns = feature_names
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.logger.info(f"Prepared features: {X_transformed.shape}")
        self.logger.info(f"Target classes: {len(self.label_encoder.classes_)}")
        
        return X_transformed, pd.Series(y_encoded, index=y.index)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, val_size: float = 0.2) -> Tuple:
        """Split data into train/validation/test sets - handle small datasets"""
        self.logger.info(f"Splitting data: train/val/test")
        
        # For very small datasets, use simple random split without stratification
        total_samples = len(X)
        self.logger.info(f"Total samples: {total_samples}")
        
        if total_samples < 100:
            self.logger.warning("Small dataset detected, using random split instead of stratification")
            
            # Simple random splits for small datasets
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42
            )
            
        else:
            # For larger datasets, try stratification with class filtering
            min_samples_per_class = 3
            class_counts = pd.Series(y).value_counts()
            small_classes = class_counts[class_counts < min_samples_per_class]
            
            if len(small_classes) > 0:
                self.logger.warning(f"Found {len(small_classes)} classes with < {min_samples_per_class} samples")
                # Remove samples from small classes for stratification to work
                valid_classes = class_counts[class_counts >= min_samples_per_class].index
                valid_mask = pd.Series(y).isin(valid_classes)
                X = X[valid_mask]
                y = y[valid_mask]
                self.logger.info(f"Filtered to {len(X)} samples for stratification")
            
            # Stratified split for larger datasets
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
            )
        
        self.logger.info(f"Train set: {X_train.shape[0]} samples")
        self.logger.info(f"Validation set: {X_val.shape[0]} samples")  
        self.logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train the ML model with better metrics"""
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
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_predictions, average='weighted', zero_division=0)
        
        # Calculate top-k accuracy (more relevant for hero picking)
        val_top5_accuracy = 0
        val_top10_accuracy = 0
        if val_probabilities is not None:
            # Get top 5 predictions for each sample
            top5_preds = np.argsort(val_probabilities, axis=1)[:, -5:]
            top10_preds = np.argsort(val_probabilities, axis=1)[:, -10:]
            
            # Check if true label is in top k
            val_top5_accuracy = np.mean([y_val.iloc[i] in top5_preds[i] for i in range(len(y_val))])
            val_top10_accuracy = np.mean([y_val.iloc[i] in top10_preds[i] for i in range(len(y_val))])
        
        # Training summary
        training_summary = {
            'model_type': self.model_type,
            'training_time': str(training_time),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': X_train.shape[1],
            'target_classes': len(self.label_encoder.classes_),
            'validation_accuracy': val_accuracy,
            'validation_top5_accuracy': val_top5_accuracy,
            'validation_top10_accuracy': val_top10_accuracy,
            'validation_precision': val_precision,
            'validation_recall': val_recall,
            'validation_f1': val_f1,
            'model_params': params
        }
        
        self.training_summary = training_summary
        self.is_trained = True
        
        self.logger.info(f"✅ Training completed in {training_time}")
        self.logger.info(f"📊 Validation accuracy: {val_accuracy:.3f}")
        self.logger.info(f"📊 Validation top-5 accuracy: {val_top5_accuracy:.3f}")
        self.logger.info(f"📊 Validation top-10 accuracy: {val_top10_accuracy:.3f}")
        self.logger.info(f"📊 Validation F1: {val_f1:.3f}")
        
        return training_summary
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model on test set with better metrics"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info("Evaluating model on test set...")
        
        # Test predictions
        test_predictions = self.model.predict(X_test)
        test_probabilities = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, test_predictions, average='weighted', zero_division=0)
        
        # Calculate top-k accuracy
        test_top5_accuracy = 0
        test_top10_accuracy = 0
        if test_probabilities is not None:
            top5_preds = np.argsort(test_probabilities, axis=1)[:, -5:]
            top10_preds = np.argsort(test_probabilities, axis=1)[:, -10:]
            
            test_top5_accuracy = np.mean([y_test.iloc[i] in top5_preds[i] for i in range(len(y_test))])
            test_top10_accuracy = np.mean([y_test.iloc[i] in top10_preds[i] for i in range(len(y_test))])
        
        # Get per-class accuracy for most common heroes
        class_accuracies = {}
        for class_idx in range(len(self.label_encoder.classes_)):
            mask = y_test == class_idx
            if mask.sum() > 0:
                class_acc = accuracy_score(y_test[mask], test_predictions[mask])
                hero_id = self.label_encoder.classes_[class_idx]
                hero_name = self.hero_names.get(hero_id, f'Hero_{hero_id}')
                class_accuracies[hero_name] = {
                    'accuracy': class_acc,
                    'samples': mask.sum()
                }
        
        # Sort by number of samples
        top_heroes = sorted(class_accuracies.items(), key=lambda x: x[1]['samples'], reverse=True)[:10]
        
        # Evaluation summary
        evaluation_summary = {
            'test_accuracy': test_accuracy,
            'test_top5_accuracy': test_top5_accuracy,
            'test_top10_accuracy': test_top10_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'top_heroes_accuracy': dict(top_heroes),
            'confusion_matrix_shape': confusion_matrix(y_test, test_predictions).shape
        }
        
        self.logger.info(f"📊 Test accuracy: {test_accuracy:.3f}")
        self.logger.info(f"📊 Test top-5 accuracy: {test_top5_accuracy:.3f}")
        self.logger.info(f"📊 Test top-10 accuracy: {test_top10_accuracy:.3f}")
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
        
        # Group by feature type
        grouped_importance = {}
        for _, row in feature_importance.iterrows():
            feature = row['feature']
            if 'team_hero' in feature:
                group = 'team_composition'
            elif 'enemy_hero' in feature:
                group = 'enemy_composition'
            elif any(role in feature for role in ['carry', 'support', 'tank', 'mid']):
                group = 'role_features'
            else:
                group = 'other'
            
            if group not in grouped_importance:
                grouped_importance[group] = 0
            grouped_importance[group] += row['importance']
        
        # Top features
        top_features = feature_importance.head(top_n)
        
        self.logger.info(f"Top {top_n} most important features:")
        for _, row in top_features.iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.logger.info(f"\nFeature importance by group:")
        for group, importance in sorted(grouped_importance.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {group}: {importance:.4f}")
        
        return {
            'top_features': top_features.to_dict('records'),
            'grouped_importance': grouped_importance
        }
    
    def save_model(self, filepath: str):
        """Save trained model and metadata"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns,
            'hero_names': self.hero_names,
            'model_type': self.model_type,
            'training_summary': self.training_summary,
            'is_trained': self.is_trained
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save with joblib
        joblib.dump(model_data, filepath)
        
        self.logger.info(f"💾 Model saved to {filepath}")

def create_training_pipeline(features_file: str, heroes_file: str, model_type: str = 'random_forest') -> Dict:
    """Complete training pipeline with fixed feature handling"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting fixed model training pipeline...")
        
        # Initialize predictor
        predictor = HeroPickPredictorFixed(model_type=model_type)
        
        # Load data
        features_df, heroes_df = predictor.load_data(features_file, heroes_file)
        
        # Prepare features with proper encoding
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
        config = Config()
        model_path = f"{config.MODELS_DIR}/hero_predictor_{model_type}_fixed.pkl"
        predictor.save_model(model_path)
        
        # Combine all results
        complete_summary = {
            'training': training_summary,
            'evaluation': evaluation_summary,
            'feature_importance': importance_summary,
            'model_path': model_path
        }
        
        # Save summary
        summary_path = f"{config.PROCESSED_DIR}/training_summary_{model_type}_fixed.json"
        with open(summary_path, 'w') as f:
            json.dump(complete_summary, f, indent=2, default=str)
        
        logger.info("✅ Training pipeline completed successfully!")
        logger.info(f"📊 Final test accuracy: {evaluation_summary['test_accuracy']:.3f}")
        logger.info(f"📊 Final test top-5 accuracy: {evaluation_summary['test_top5_accuracy']:.3f}")
        logger.info(f"📊 Final test top-10 accuracy: {evaluation_summary['test_top10_accuracy']:.3f}")
        logger.info(f"💾 Model saved to: {model_path}")
        logger.info(f"📋 Summary saved to: {summary_path}")
        
        return complete_summary
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise

def main():
    """CLI interface for fixed model training"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Dota 2 hero pick prediction models with proper feature handling',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
                results[model_type] = {
                    'top1': summary['evaluation']['test_accuracy'],
                    'top5': summary['evaluation']['test_top5_accuracy'],
                    'top10': summary['evaluation']['test_top10_accuracy']
                }
            
            # Show comparison
            print("\n📈 Model Comparison Results:")
            print(f"{'Model':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Top-10 Acc':<12}")
            print("-" * 56)
            for model, metrics in sorted(results.items(), key=lambda x: x[1]['top5'], reverse=True):
                print(f"{model:<20} {metrics['top1']:<12.3f} {metrics['top5']:<12.3f} {metrics['top10']:<12.3f}")
            
        else:
            # Train single model
            summary = create_training_pipeline(args.features, args.heroes, args.model)
            
            print(f"\n🎉 Training completed!")
            print(f"Model: {args.model}")
            print(f"Test Accuracy: {summary['evaluation']['test_accuracy']:.3f}")
            print(f"Test Top-5 Accuracy: {summary['evaluation']['test_top5_accuracy']:.3f}")
            print(f"Test Top-10 Accuracy: {summary['evaluation']['test_top10_accuracy']:.3f}")
            print(f"Model saved to: {summary['model_path']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())