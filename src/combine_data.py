#!/usr/bin/env python3
"""
Combine All Data Collection Files
Merges multiple JSON files and removes duplicates
"""

import json
import os
import glob
from typing import List, Dict, Set
from collections import defaultdict

def find_all_data_files() -> List[str]:
    """Find all data collection files"""
    data_files = []
    
    # Common patterns for data files
    patterns = [
        'data/raw/detailed_matches*.json',
        'data/raw/matches_temp*.json', 
        'detailed_matches*.json',
        'matches_temp*.json',
        '*matches*.json'
    ]
    
    print("🔍 Searching for data files...")
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            if os.path.getsize(file) > 1024:  # Skip tiny files
                data_files.append(file)
                size_mb = os.path.getsize(file) / 1024 / 1024
                print(f"   📄 {file} ({size_mb:.1f} MB)")
    
    # Remove duplicates and sort
    data_files = sorted(list(set(data_files)))
    print(f"📊 Found {len(data_files)} data files")
    
    return data_files

def load_matches_from_file(file_path: str) -> List[Dict]:
    """Load matches from a single file"""
    try:
        print(f"📥 Loading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different file formats
        if isinstance(data, list):
            matches = data
        elif isinstance(data, dict) and 'matches' in data:
            matches = data['matches']
        else:
            print(f"⚠️  Unknown format in {file_path}")
            return []
        
        print(f"   ✅ Loaded {len(matches)} matches")
        return matches
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def normalize_match_format(match: Dict) -> Dict:
    """Normalize match to consistent format"""
    # Extract essential fields only
    normalized = {
        'match_id': match.get('match_id'),
        'duration': match.get('duration'),
        'radiant_win': match.get('radiant_win'),
        'skill': match.get('skill'),
        'skill_name': match.get('skill_name'),
        'avg_mmr': match.get('avg_mmr'),
        'patch': match.get('patch'),
        'lobby_type': match.get('lobby_type'),
        'game_mode': match.get('game_mode'),
        'region': match.get('region'),
        'start_time': match.get('start_time'),
        'players': []
    }
    
    # Normalize players
    players = match.get('players', [])
    for player in players:
        if isinstance(player, dict):
            normalized_player = {
                'hero_id': player.get('hero_id'),
                'player_slot': player.get('player_slot'),
                'rank_tier': player.get('rank_tier')
            }
            normalized['players'].append(normalized_player)
    
    return normalized

def combine_all_data() -> List[Dict]:
    """Combine all data files and remove duplicates"""
    print("🔄 Starting data combination process...")
    
    # Find all data files
    data_files = find_all_data_files()
    
    if not data_files:
        print("❌ No data files found!")
        return []
    
    # Load all matches
    all_matches = []
    seen_match_ids: Set[int] = set()
    
    for file_path in data_files:
        matches = load_matches_from_file(file_path)
        
        for match in matches:
            match_id = match.get('match_id')
            
            if match_id and match_id not in seen_match_ids:
                # Normalize format and add
                normalized_match = normalize_match_format(match)
                if normalized_match['match_id']:  # Valid match
                    all_matches.append(normalized_match)
                    seen_match_ids.add(match_id)
    
    print(f"📊 Total unique matches after deduplication: {len(all_matches)}")
    return all_matches

def analyze_combined_data(matches: List[Dict]) -> Dict:
    """Analyze the combined dataset"""
    print("📈 Analyzing combined dataset...")
    
    # Skill distribution
    skill_dist = defaultdict(int)
    patch_dist = defaultdict(int)
    duration_stats = []
    hero_usage = defaultdict(int)
    
    for match in matches:
        # Skill distribution
        skill = match.get('skill_name', 'Unknown')
        skill_dist[skill] += 1
        
        # Patch distribution
        patch = match.get('patch')
        if patch:
            patch_dist[str(patch)] += 1
        
        # Duration stats
        duration = match.get('duration', 0)
        if duration > 0:
            duration_stats.append(duration)
        
        # Hero usage
        for player in match.get('players', []):
            hero_id = player.get('hero_id')
            if hero_id:
                hero_usage[hero_id] += 1
    
    # Calculate stats
    analysis = {
        'total_matches': len(matches),
        'skill_distribution': dict(skill_dist),
        'patch_distribution': dict(sorted(patch_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
        'unique_heroes': len(hero_usage),
        'avg_duration': sum(duration_stats) / len(duration_stats) if duration_stats else 0,
        'min_duration': min(duration_stats) if duration_stats else 0,
        'max_duration': max(duration_stats) if duration_stats else 0
    }
    
    return analysis

def save_combined_data(matches: List[Dict], analysis: Dict):
    """Save combined data and analysis"""
    print("💾 Saving combined data...")
    
    # Create output directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Save combined matches (compact format)
    output_file = 'data/raw/detailed_matches_combined.json'
    with open(output_file, 'w') as f:
        json.dump(matches, f)  # No indentation = compact
    
    # Save analysis
    analysis_file = 'data/processed/combined_data_analysis.json'
    os.makedirs('data/processed', exist_ok=True)
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # File size info
    file_size = os.path.getsize(output_file) / 1024 / 1024
    
    print(f"✅ Combined data saved!")
    print(f"📄 Output file: {output_file}")
    print(f"📊 File size: {file_size:.1f} MB")
    print(f"📈 Analysis saved: {analysis_file}")
    
    return output_file

def main():
    """Main combination process"""
    print("🚀 Starting data combination process...")
    
    try:
        # Combine all data
        combined_matches = combine_all_data()
        
        if not combined_matches:
            print("❌ No matches to combine!")
            return 1
        
        # Analyze combined data
        analysis = analyze_combined_data(combined_matches)
        
        # Save results
        output_file = save_combined_data(combined_matches, analysis)
        
        # Print summary
        print(f"\n🎉 Data combination completed!")
        print(f"📊 Summary:")
        print(f"   Total matches: {analysis['total_matches']:,}")
        print(f"   Unique heroes: {analysis['unique_heroes']}")
        print(f"   Average duration: {analysis['avg_duration']/60:.1f} minutes")
        print(f"   Skill distribution:")
        for skill, count in analysis['skill_distribution'].items():
            percentage = count / analysis['total_matches'] * 100
            print(f"     {skill}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n📁 Combined file ready for ML training:")
        print(f"   {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Combination failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())