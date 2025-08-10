#!/usr/bin/env python3
"""
Neural Network Model for Dota 2 Hero Prediction
Better handling of high-cardinality categorical features
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from torch.nn import functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Sklearn for preprocessing and fallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.ensemble import RandomForestClassifier

class HeroEmbeddingNet(nn.Module):
    """Neural network with hero embeddings for better categorical handling"""
    
    def __init__(self, n_heroes, n_other_features, n_classes, embedding_dim=32, hidden_dims=[256, 128, 64]):
        super(HeroEmbeddingNet, self).__init__()
        
        self.n_heroes = n_heroes
        self.embedding_dim = embedding_dim
        
        # Hero embeddings (learns representations for each hero)
        self.hero_embeddings = nn.Embedding(n_heroes + 1, embedding_dim, padding_idx=0)
        
        # We have 8 hero positions (4 team + 4 enemy)
        total_hero_embedding_size = 8 * embedding_dim
        
        # Input size: hero embeddings + other numerical features
        input_size = total_hero_embedding_size + n_other_features
        
        # Hidden layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_dims:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, n_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def forward(self, hero_ids, other_features):
        """Forward pass"""
        batch_size = hero_ids.shape[0]
        
        # Get embeddings for all hero positions
        # hero_ids shape: (batch_size, 8) for 8 hero positions
        hero_embeds = self.hero_embeddings(hero_ids)  # (batch_size, 8, embedding_dim)
        hero_embeds = hero_embeds.view(batch_size, -1)  # (batch_size, 8 * embedding_dim)
        
        # Concatenate hero embeddings with other features
        combined = torch.cat([hero_embeds, other_features], dim=1)
        
        # Pass through network
        output = self.network(combined)
        return output

class NeuralHeroPredictor:
    """Neural network-based hero predictor"""
    
    def __init__(self, device='cpu'):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.hero_to_idx = {}
        self.idx_to_hero = {}
        
        # Training state
        self.is_trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Check PyTorch availability
        if not PYTORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, falling back to RandomForest")
            self.use_neural = False
        else:
            self.use_neural = True
    
    def load_data(self, features_file: str, heroes_file: str) -> Tuple[pd.DataFrame, Dict]:
        """Load and prepare data"""
        self.logger.info(f"Loading data from {features_file}")
        
        # Load features - use raw features
        if 'features_ml.csv' in features_file:
            features_file = features_file.replace('features_ml.csv', 'features_raw.csv')
        
        features_df = pd.read_csv(features_file)
        
        # Load heroes
        with open(heroes_file, 'r') as f:
            heroes = json.load(f)
        
        hero_names = {h['id']: h['localized_name'] for h in heroes}
        all_hero_ids = set(h['id'] for h in heroes)
        
        # Filter valid heroes
        valid_mask = features_df['target_hero'].isin(all_hero_ids)
        features_df = features_df[valid_mask]
        
        self.logger.info(f"Loaded {len(features_df)} training samples")
        self.logger.info(f"Unique target heroes: {features_df['target_hero'].nunique()}")
        
        return features_df, hero_names
    
    def prepare_features(self, features_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare features for neural network"""
        self.logger.info("Preparing features for neural network...")
        
        # Create hero ID mapping
        all_hero_ids = set()
        hero_cols = ['team_hero_1', 'team_hero_2', 'team_hero_3', 'team_hero_4',
                    'enemy_hero_1', 'enemy_hero_2', 'enemy_hero_3', 'enemy_hero_4', 'target_hero']
        
        for col in hero_cols:
            if col in features_df.columns:
                all_hero_ids.update(features_df[col].unique())
        
        all_hero_ids.discard(0)  # Remove 'no hero' placeholder
        all_hero_ids = sorted(all_hero_ids)
        
        # Create mappings (0 reserved for 'no hero')
        self.hero_to_idx = {hero_id: idx + 1 for idx, hero_id in enumerate(all_hero_ids)}
        self.hero_to_idx[0] = 0  # 'no hero' maps to 0
        self.idx_to_hero = {idx: hero_id for hero_id, idx in self.hero_to_idx.items()}
        
        self.logger.info(f"Created hero vocabulary of size: {len(self.hero_to_idx)}")
        
        # Prepare hero features (8 positions)
        hero_features = []
        hero_input_cols = ['team_hero_1', 'team_hero_2', 'team_hero_3', 'team_hero_4',
                          'enemy_hero_1', 'enemy_hero_2', 'enemy_hero_3', 'enemy_hero_4']
        
        for col in hero_input_cols:
            if col in features_df.columns:
                hero_features.append(features_df[col].map(self.hero_to_idx).fillna(0).astype(int))
            else:
                hero_features.append(pd.Series([0] * len(features_df)))
        
        hero_tensor = torch.tensor(np.column_stack(hero_features), dtype=torch.long)
        
        # Prepare numerical features
        numerical_cols = [
            'duration', 'team_size', 'enemy_size', 'total_heroes_picked',
            'team_carry_count', 'enemy_carry_count', 'team_support_count', 'enemy_support_count',
            'team_tank_count', 'enemy_tank_count', 'team_mid_count', 'enemy_mid_count',
            'team_versatile_count', 'enemy_versatile_count', 'team_role_diversity', 'enemy_role_diversity',
            'synergy_count', 'has_synergy', 'countered_enemies', 'counter_strength', 'is_counter_pick'
        ]
        
        # Add categorical features as numerical (one-hot encoded)
        categorical_features = []
        if 'team_side' in features_df.columns:
            categorical_features.append(pd.get_dummies(features_df['team_side'], prefix='team_side'))
        if 'pick_phase' in features_df.columns:
            categorical_features.append(pd.get_dummies(features_df['pick_phase'], prefix='pick_phase'))
        if 'pick_position' in features_df.columns:
            categorical_features.append(pd.get_dummies(features_df['pick_position'], prefix='pick_position'))
        
        # Combine numerical features
        numerical_data = []
        for col in numerical_cols:
            if col in features_df.columns:
                numerical_data.append(features_df[col])
        
        if numerical_data:
            numerical_df = pd.concat(numerical_data, axis=1)
        else:
            numerical_df = pd.DataFrame(index=features_df.index)
        
        if categorical_features:
            categorical_df = pd.concat(categorical_features, axis=1)
            numerical_df = pd.concat([numerical_df, categorical_df], axis=1)
        
        # Scale numerical features
        numerical_scaled = self.scaler.fit_transform(numerical_df.fillna(0))
        numerical_tensor = torch.tensor(numerical_scaled, dtype=torch.float32)
        
        # Prepare targets
        y_encoded = self.label_encoder.fit_transform(features_df['target_hero'])
        target_tensor = torch.tensor(y_encoded, dtype=torch.long)
        
        self.logger.info(f"Hero features shape: {hero_tensor.shape}")
        self.logger.info(f"Numerical features shape: {numerical_tensor.shape}")
        self.logger.info(f"Target classes: {len(self.label_encoder.classes_)}")
        
        return hero_tensor, numerical_tensor, target_tensor
    
    def train_model(self, hero_features, numerical_features, targets, epochs=100, batch_size=64, lr=0.001):
        """Train neural network"""
        if not self.use_neural:
            return self._train_fallback_model(hero_features, numerical_features, targets)
        
        self.logger.info("Training neural network...")
        
        # Split data
        indices = np.arange(len(targets))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = TensorDataset(hero_features[train_idx], numerical_features[train_idx], targets[train_idx])
        val_dataset = TensorDataset(hero_features[val_idx], numerical_features[val_idx], targets[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        n_heroes = len(self.hero_to_idx)
        n_other_features = numerical_features.shape[1]
        n_classes = len(self.label_encoder.classes_)
        
        self.model = HeroEmbeddingNet(n_heroes, n_other_features, n_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        # Training loop
        best_val_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for hero_batch, num_batch, target_batch in train_loader:
                hero_batch = hero_batch.to(self.device)
                num_batch = num_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(hero_batch, num_batch)
                loss = criterion(outputs, target_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += target_batch.size(0)
                train_correct += (predicted == target_batch).sum().item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_top5_correct = 0
            
            with torch.no_grad():
                for hero_batch, num_batch, target_batch in val_loader:
                    hero_batch = hero_batch.to(self.device)
                    num_batch = num_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                    
                    outputs = self.model(hero_batch, num_batch)
                    loss = criterion(outputs, target_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += target_batch.size(0)
                    val_correct += (predicted == target_batch).sum().item()
                    
                    # Top-5 accuracy
                    _, top5_pred = torch.topk(outputs, 5, dim=1)
                    val_top5_correct += sum([target_batch[i] in top5_pred[i] for i in range(len(target_batch))])
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            val_top5_acc = val_top5_correct / val_total
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = self.model.state_dict().copy()
            
            if epoch % 10 == 0:
                self.logger.info(f'Epoch {epoch}: Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Val Top-5: {val_top5_acc:.3f}')
        
        # Load best model
        self.model.load_state_dict(best_model)
        self.is_trained = True
        
        self.logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.3f}")
        return {"val_accuracy": best_val_acc}
    
    def _train_fallback_model(self, hero_features, numerical_features, targets):
        """Fallback to RandomForest if PyTorch unavailable"""
        self.logger.info("Training RandomForest fallback model...")
        
        # Combine features (flatten hero features)
        hero_flat = hero_features.numpy().reshape(hero_features.shape[0], -1)
        numerical_np = numerical_features.numpy()
        X = np.concatenate([hero_flat, numerical_np], axis=1)
        y = targets.numpy()
        
        # Train RandomForest
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        val_acc = accuracy_score(y_val, self.model.predict(X_val))
        self.is_trained = True
        
        self.logger.info(f"Fallback model trained. Validation accuracy: {val_acc:.3f}")
        return {"val_accuracy": val_acc}
    
    def predict_top_heroes(self, hero_features, numerical_features, top_k=10):
        """Predict top k heroes"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.use_neural:
            self.model.eval()
            with torch.no_grad():
                hero_features = hero_features.to(self.device)
                numerical_features = numerical_features.to(self.device)
                outputs = self.model(hero_features, numerical_features)
                probabilities = F.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
                
            return top_indices.cpu().numpy(), top_probs.cpu().numpy()
        else:
            # Fallback model prediction
            hero_flat = hero_features.numpy().reshape(hero_features.shape[0], -1)
            numerical_np = numerical_features.numpy()
            X = np.concatenate([hero_flat, numerical_np], axis=1)
            
            probabilities = self.model.predict_proba(X)
            top_indices = np.argsort(probabilities, axis=1)[:, -top_k:][:, ::-1]
            top_probs = np.sort(probabilities, axis=1)[:, -top_k:][:, ::-1]
            
            return top_indices, top_probs
    
    def save_model(self, filepath):
        """Save model"""
        model_data = {
            'model_state_dict': self.model.state_dict() if self.use_neural else self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'hero_to_idx': self.hero_to_idx,
            'idx_to_hero': self.idx_to_hero,
            'use_neural': self.use_neural,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

def main():
    """Train neural network model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train neural network for hero prediction')
    parser.add_argument('--features', default='data/processed/features_raw.csv', help='Features file')
    parser.add_argument('--heroes', default='data/raw/heroes.json', help='Heroes file')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize predictor
        predictor = NeuralHeroPredictor()
        
        # Load data
        features_df, hero_names = predictor.load_data(args.features, args.heroes)
        
        # Prepare features
        hero_features, numerical_features, targets = predictor.prepare_features(features_df)
        
        # Train model
        results = predictor.train_model(
            hero_features, numerical_features, targets,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
        
        # Save model
        config = Config()
        model_path = f"{config.MODELS_DIR}/hero_predictor_neural.pkl"
        predictor.save_model(model_path)
        
        print(f"\n🎉 Neural network training completed!")
        print(f"📊 Validation accuracy: {results['val_accuracy']:.3f}")
        print(f"💾 Model saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())