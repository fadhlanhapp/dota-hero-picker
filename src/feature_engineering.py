#!/usr/bin/env python3
"""
Dota 2 Hero Picker - Feature Engineering Module
Converts raw match data into ML training features
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import itertools
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class HeroFeatureExtractor:
    """Extracts and engineers features from match data for ML training"""
    
    def __init__(self, heroes_data: List[Dict]):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Create hero lookup tables
        self.heroes_df = pd.DataFrame(heroes_data)
        self.hero_id_to_name = dict(zip(self.heroes_df['id'], self.heroes_df['localized_name']))
        self.hero_name_to_id = dict(zip(self.heroes_df['localized_name'], self.heroes_df['id']))
        self.all_hero_ids = set(self.heroes_df['id'].tolist())
        
        # Hero role mapping (simplified)
        self.hero_roles = self._create_hero_roles()
        
        self.logger.info(f"Initialized with {len(self.heroes_df)} heroes")
    
    def _create_hero_roles(self) -> Dict[int, List[str]]:
        """Create hero role mapping from hero data"""
        roles = {}
        
        for _, hero in self.heroes_df.iterrows():
            hero_id = hero['id']
            hero_name = hero['localized_name'].lower()
            
            # Simplified role assignment based on typical roles
            # In production, you'd use actual role data from OpenDota
            hero_roles = []
            
            # Carry heroes (high damage, item dependent)
            if any(name in hero_name for name in ['anti-mage', 'phantom assassin', 'spectre', 'medusa', 'drow ranger']):
                hero_roles.append('carry')
            
            # Support heroes (utility, less item dependent)
            elif any(name in hero_name for name in ['crystal maiden', 'lion', 'dazzle', 'witch doctor', 'warlock']):
                hero_roles.append('support')
            
            # Tank/Initiator heroes
            elif any(name in hero_name for name in ['pudge', 'axe', 'tidehunter', 'centaur', 'slardar']):
                hero_roles.append('tank')
            
            # Mid heroes (burst damage, level dependent)
            elif any(name in hero_name for name in ['invoker', 'shadow fiend', 'queen of pain', 'storm spirit']):
                hero_roles.append('mid')
            
            else:
                hero_roles.append('versatile')  # Default role
            
            roles[hero_id] = hero_roles
        
        return roles
    
    def extract_match_features(self, detailed_matches: List[Dict]) -> pd.DataFrame:
        """Main feature extraction pipeline"""
        self.logger.info(f"Extracting features from {len(detailed_matches)} matches...")
        
        all_features = []
        match_count = 0
        
        for match in detailed_matches:
            try:
                match_features = self._extract_single_match_features(match)
                all_features.extend(match_features)
                match_count += 1
                
                if match_count % 500 == 0:
                    self.logger.info(f"Processed {match_count} matches, generated {len(all_features)} samples")
                    
            except Exception as e:
                self.logger.warning(f"Failed to process match {match.get('match_id', 'unknown')}: {e}")
                continue
        
        features_df = pd.DataFrame(all_features)
        self.logger.info(f"Feature extraction complete: {len(features_df)} training samples from {match_count} matches")
        
        return features_df
    
    def _extract_single_match_features(self, match: Dict) -> List[Dict]:
        """Extract multiple training samples from a single match"""
        match_features = []
        
        # Basic match info
        match_id = match['match_id']
        duration = match.get('duration', 0)
        radiant_win = match.get('radiant_win', False)
        
        # Extract team compositions
        radiant_heroes, dire_heroes = self._extract_team_compositions(match)
        
        if len(radiant_heroes) != 5 or len(dire_heroes) != 5:
            return []  # Skip incomplete teams
        
        # Generate training samples for different pick positions
        # Simulate draft phases: after 1, 2, 3, 4 picks are made
        
        for pick_position in range(1, 5):  # Positions 2-5 (after 1-4 picks)
            # Radiant team scenarios
            if len(radiant_heroes) > pick_position:
                features = self._create_pick_scenario(
                    team_heroes=radiant_heroes[:pick_position],
                    enemy_heroes=dire_heroes[:pick_position],
                    target_hero=radiant_heroes[pick_position],
                    team_won=radiant_win,
                    match_id=match_id,
                    duration=duration,
                    pick_position=pick_position,
                    team_side='radiant'
                )
                match_features.append(features)
            
            # Dire team scenarios
            if len(dire_heroes) > pick_position:
                features = self._create_pick_scenario(
                    team_heroes=dire_heroes[:pick_position],
                    enemy_heroes=radiant_heroes[:pick_position],
                    target_hero=dire_heroes[pick_position],
                    team_won=not radiant_win,  # Dire wins when radiant doesn't
                    match_id=match_id,
                    duration=duration,
                    pick_position=pick_position,
                    team_side='dire'
                )
                match_features.append(features)
        
        return match_features
    
    def _extract_team_compositions(self, match: Dict) -> Tuple[List[int], List[int]]:
        """Extract hero picks for both teams"""
        radiant_heroes = []
        dire_heroes = []
        
        for player in match.get('players', []):
            hero_id = player.get('hero_id')
            player_slot = player.get('player_slot', 0)
            
            if hero_id and hero_id > 0:
                if player_slot < 128:  # Radiant team
                    radiant_heroes.append(hero_id)
                else:  # Dire team
                    dire_heroes.append(hero_id)
        
        return radiant_heroes, dire_heroes
    
    def _create_pick_scenario(
        self, 
        team_heroes: List[int], 
        enemy_heroes: List[int],
        target_hero: int,
        team_won: bool,
        match_id: int,
        duration: int,
        pick_position: int,
        team_side: str
    ) -> Dict:
        """Create a single training sample"""
        
        features = {
            # Meta information
            'match_id': match_id,
            'pick_position': pick_position,
            'team_side': team_side,
            'duration': duration,
            
            # Target variables
            'target_hero': target_hero,
            'target_hero_name': self.hero_id_to_name.get(target_hero, f'Hero_{target_hero}'),
            'team_won': int(team_won),
            
            # Basic composition features
            'team_size': len(team_heroes),
            'enemy_size': len(enemy_heroes),
        }
        
        # Individual hero features (one-hot style)
        # Team heroes (pad to 4 slots)
        for i in range(4):
            if i < len(team_heroes):
                features[f'team_hero_{i+1}'] = team_heroes[i]
                features[f'team_hero_{i+1}_name'] = self.hero_id_to_name.get(team_heroes[i], f'Hero_{team_heroes[i]}')
            else:
                features[f'team_hero_{i+1}'] = 0
                features[f'team_hero_{i+1}_name'] = 'None'
        
        # Enemy heroes (pad to 4 slots)
        for i in range(4):
            if i < len(enemy_heroes):
                features[f'enemy_hero_{i+1}'] = enemy_heroes[i]
                features[f'enemy_hero_{i+1}_name'] = self.hero_id_to_name.get(enemy_heroes[i], f'Hero_{enemy_heroes[i]}')
            else:
                features[f'enemy_hero_{i+1}'] = 0
                features[f'enemy_hero_{i+1}_name'] = 'None'
        
        # Advanced features
        features.update(self._extract_advanced_features(team_heroes, enemy_heroes, target_hero))
        
        return features
    
    def _extract_advanced_features(self, team_heroes: List[int], enemy_heroes: List[int], target_hero: int) -> Dict:
        """Extract advanced strategic features"""
        features = {}
        
        # Role distribution features
        team_roles = self._get_role_distribution(team_heroes)
        enemy_roles = self._get_role_distribution(enemy_heroes)
        target_roles = self.hero_roles.get(target_hero, ['versatile'])
        
        # Team role counts
        for role in ['carry', 'support', 'tank', 'mid', 'versatile']:
            features[f'team_{role}_count'] = team_roles.get(role, 0)
            features[f'enemy_{role}_count'] = enemy_roles.get(role, 0)
            features[f'target_is_{role}'] = int(role in target_roles)
        
        # Role balance features
        features['team_role_diversity'] = len([r for r in team_roles.values() if r > 0])
        features['enemy_role_diversity'] = len([r for r in enemy_roles.values() if r > 0])
        
        # Hero synergy features (simplified)
        features.update(self._extract_synergy_features(team_heroes, target_hero))
        
        # Counter-pick features (simplified)
        features.update(self._extract_counter_features(enemy_heroes, target_hero))
        
        # Meta features
        features['total_heroes_picked'] = len(team_heroes) + len(enemy_heroes)
        features['pick_phase'] = 'early' if len(team_heroes) <= 1 else 'mid' if len(team_heroes) <= 3 else 'late'
        
        return features
    
    def _get_role_distribution(self, heroes: List[int]) -> Dict[str, int]:
        """Get role counts for a list of heroes"""
        role_counts = defaultdict(int)
        
        for hero_id in heroes:
            roles = self.hero_roles.get(hero_id, ['versatile'])
            for role in roles:
                role_counts[role] += 1
        
        return dict(role_counts)
    
    def _extract_synergy_features(self, team_heroes: List[int], target_hero: int) -> Dict:
        """Extract hero synergy features (simplified version)"""
        features = {}
        
        # Known synergy pairs (in production, this would be data-driven)
        synergy_pairs = {
            # Example synergies (hero_id pairs that work well together)
            (1, 2): 'am_cm_synergy',  # Anti-Mage + Crystal Maiden
            (2, 4): 'cm_bloodseeker_synergy',  # Crystal Maiden + Bloodseeker
            # Add more based on game knowledge
        }
        
        synergy_count = 0
        synergy_types = []
        
        for team_hero in team_heroes:
            # Check if current hero has synergy with target
            pair1 = (min(team_hero, target_hero), max(team_hero, target_hero))
            pair2 = (min(target_hero, team_hero), max(target_hero, team_hero))
            
            if pair1 in synergy_pairs:
                synergy_count += 1
                synergy_types.append(synergy_pairs[pair1])
            elif pair2 in synergy_pairs:
                synergy_count += 1
                synergy_types.append(synergy_pairs[pair2])
        
        features['synergy_count'] = synergy_count
        features['has_synergy'] = int(synergy_count > 0)
        
        return features
    
    def _extract_counter_features(self, enemy_heroes: List[int], target_hero: int) -> Dict:
        """Extract counter-pick features (simplified version)"""
        features = {}
        
        # Known counter relationships (in production, this would be data-driven)
        counter_pairs = {
            # target_hero: [list of heroes it counters]
            1: [10, 11],  # Anti-Mage counters SF, Drow
            2: [5, 6],    # Crystal Maiden counters others
            # Add more based on game knowledge
        }
        
        countered_enemies = 0
        counter_strength = 0
        
        target_counters = counter_pairs.get(target_hero, [])
        
        for enemy_hero in enemy_heroes:
            if enemy_hero in target_counters:
                countered_enemies += 1
                counter_strength += 1  # Simple strength, could be weighted
        
        features['countered_enemies'] = countered_enemies
        features['counter_strength'] = counter_strength
        features['is_counter_pick'] = int(countered_enemies > 0)
        
        return features
    
    def create_categorical_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Convert features to categorical format suitable for ML"""
        self.logger.info("Creating categorical features for ML...")
        
        # Create a copy for processing
        ml_df = features_df.copy()
        
        # Hero ID categorical features (exclude target_hero from encoding)
        hero_columns = [col for col in ml_df.columns if 'hero_' in col and col.endswith(('_1', '_2', '_3', '_4'))]
        
        # Convert hero IDs to categorical (but keep target_hero as numeric)
        for col in hero_columns:
            if col in ml_df.columns:
                ml_df[col] = ml_df[col].astype('category')
        
        # Pick position and team side
        ml_df['pick_position'] = ml_df['pick_position'].astype('category')
        ml_df['team_side'] = ml_df['team_side'].astype('category')
        ml_df['pick_phase'] = ml_df['pick_phase'].astype('category')
        
        # One-hot encode categorical variables
        categorical_cols = ml_df.select_dtypes(include=['category']).columns
        ml_df = pd.get_dummies(ml_df, columns=categorical_cols, prefix_sep='_')
        
        self.logger.info(f"Created {ml_df.shape[1]} total features")
        return ml_df
    
    def generate_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """Generate summary statistics of extracted features"""
        summary = {
            'total_samples': len(features_df),
            'unique_matches': features_df['match_id'].nunique(),
            'unique_heroes_as_targets': features_df['target_hero'].nunique(),
            'win_rate_distribution': features_df['team_won'].value_counts().to_dict(),
            'pick_position_distribution': features_df['pick_position'].value_counts().to_dict(),
            'team_side_distribution': features_df['team_side'].value_counts().to_dict(),
            'duration_stats': {
                'mean': features_df['duration'].mean(),
                'median': features_df['duration'].median(),
                'std': features_df['duration'].std()
            },
            'feature_columns': list(features_df.columns),
            'top_target_heroes': features_df['target_hero_name'].value_counts().head(10).to_dict()
        }
        
        return summary

def main():
    """CLI interface for feature engineering"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Extract features from Dota 2 match data')
    parser.add_argument('--input', default='data/raw', help='Input directory with match data')
    parser.add_argument('--output', default='data/processed', help='Output directory for features')
    parser.add_argument('--sample', type=int, help='Process only N matches for testing')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Load data
        logger.info("Loading match data...")
        
        # Load heroes
        heroes_file = os.path.join(args.input, 'heroes.json')
        with open(heroes_file, 'r') as f:
            heroes = json.load(f)
        
        # Load matches
        matches_file = os.path.join(args.input, 'detailed_matches.json')
        with open(matches_file, 'r') as f:
            matches = json.load(f)
        
        # Sample if requested
        if args.sample:
            matches = matches[:args.sample]
            logger.info(f"Using sample of {len(matches)} matches")
        
        # Extract features
        extractor = HeroFeatureExtractor(heroes)
        features_df = extractor.extract_match_features(matches)
        
        # Create ML-ready features
        ml_features_df = extractor.create_categorical_features(features_df)
        
        # Generate summary
        summary = extractor.generate_feature_summary(features_df)
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        
        # Save raw features
        features_file = os.path.join(args.output, 'features_raw.csv')
        features_df.to_csv(features_file, index=False)
        logger.info(f"Saved raw features to {features_file}")
        
        # Save ML features  
        ml_features_file = os.path.join(args.output, 'features_ml.csv')
        ml_features_df.to_csv(ml_features_file, index=False)
        logger.info(f"Saved ML features to {ml_features_file}")
        
        # Save summary
        summary_file = os.path.join(args.output, 'feature_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved feature summary to {summary_file}")
        
        # Print summary
        logger.info("Feature Engineering Summary:")
        logger.info(f"  Total samples: {summary['total_samples']}")
        logger.info(f"  From matches: {summary['unique_matches']}")
        logger.info(f"  Unique target heroes: {summary['unique_heroes_as_targets']}")
        logger.info(f"  Win rate: {summary['win_rate_distribution'].get(1, 0) / summary['total_samples'] * 100:.1f}%")
        
        print(f"\n✅ Feature engineering completed!")
        print(f"📊 Generated {summary['total_samples']} training samples")
        print(f"📁 Files saved to {args.output}/")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())