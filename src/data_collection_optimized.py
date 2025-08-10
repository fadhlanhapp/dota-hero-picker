#!/usr/bin/env python3
"""
Optimized Dota 2 Data Collection using proper API parameters
Much faster and more efficient than random searching
"""

import requests
import json
import time
import pandas as pd
from tqdm import tqdm
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OptimizedCollector:
    """Optimized data collector using proper API endpoints"""
    
    def __init__(self):
        self.session = requests.Session()
        self.rate_limit_delay = 0.3  # 300ms between requests
        
        # Statistics tracking
        self.stats = {
            'requests_made': 0,
            'matches_collected': 0,
            'start_time': datetime.now()
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Optimized Collector initialized")
    
    def _make_api_call(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Make API call with error handling"""
        url = f"https://api.opendota.com/api/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            self.stats['requests_made'] += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                self.logger.warning("Rate limited, waiting 10 seconds...")
                time.sleep(10)
                return None
            else:
                self.logger.debug(f"API call failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.debug(f"API call error: {e}")
            return None
    
    def get_recent_matches_by_skill(self, skill_level: int, count: int = 100) -> List[int]:
        """
        Get recent match IDs for specific skill level using proper API
        
        Args:
            skill_level: 1=Normal, 2=High, 3=Very High
            count: Number of matches to collect
        """
        skill_names = {1: "Normal", 2: "High", 3: "Very High"}
        skill_name = skill_names.get(skill_level, "Unknown")
        
        self.logger.info(f"Collecting {count} {skill_name} skill matches using optimized API...")
        
        match_ids = []
        
        # Use the public matches endpoint with skill filter
        # This is much more efficient than random guessing!
        params = {
            'mmr_ascending': 0,  # Start from highest MMR within bracket
            'less_than_match_id': None  # For pagination
        }
        
        # Map skill levels to approximate MMR ranges for API
        if skill_level == 1:  # Normal
            params['max_mmr'] = 3200
        elif skill_level == 2:  # High  
            params['min_mmr'] = 3200
            params['max_mmr'] = 3700
        elif skill_level == 3:  # Very High
            params['min_mmr'] = 3700
        
        pbar = tqdm(total=count, desc=f"Collecting {skill_name} matches")
        
        while len(match_ids) < count:
            # Get public matches
            public_matches = self._make_api_call('publicMatches', params)
            
            if not public_matches:
                time.sleep(5)
                continue
            
            for match in public_matches:
                if len(match_ids) >= count:
                    break
                
                match_id = match.get('match_id')
                if match_id and match_id not in match_ids:
                    match_ids.append(match_id)
                    pbar.update(1)
            
            # Pagination - get older matches
            if public_matches:
                params['less_than_match_id'] = public_matches[-1].get('match_id')
            
            time.sleep(self.rate_limit_delay)
        
        pbar.close()
        self.logger.info(f"✅ Collected {len(match_ids)} {skill_name} match IDs")
        return match_ids[:count]
    
    def collect_matches_optimized(self, total_matches: int = 1000, skill_distribution: dict = None) -> List[Dict]:
        """
        Collect matches with optimized API usage
        """
        if not skill_distribution:
            skill_distribution = {"Normal": 0.4, "High": 0.4, "Very High": 0.2}
        
        self.logger.info(f"Starting optimized collection of {total_matches} matches")
        self.logger.info(f"Skill distribution: {skill_distribution}")
        
        all_match_ids = []
        
        # Collect match IDs for each skill bracket
        for skill_name, ratio in skill_distribution.items():
            skill_level = {"Normal": 1, "High": 2, "Very High": 3}[skill_name]
            target_count = int(total_matches * ratio)
            
            match_ids = self.get_recent_matches_by_skill(skill_level, target_count)
            all_match_ids.extend(match_ids)
        
        self.logger.info(f"Collected {len(all_match_ids)} total match IDs")
        
        # Now collect detailed match data
        detailed_matches = []
        
        pbar = tqdm(all_match_ids, desc="Fetching match details")
        
        for match_id in pbar:
            match_data = self._make_api_call(f"matches/{match_id}")
            
            if match_data and self._is_valid_match(match_data):
                essential_match = self._extract_essential_data(match_data)
                detailed_matches.append(essential_match)
                self.stats['matches_collected'] += 1
            
            pbar.set_postfix({
                'valid': len(detailed_matches),
                'success_rate': f"{len(detailed_matches)/(pbar.n+1)*100:.1f}%"
            })
            
            time.sleep(self.rate_limit_delay)
        
        pbar.close()
        
        # Save results
        output_file = "data/raw/detailed_matches_optimized.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(detailed_matches, f)
        
        self.logger.info(f"✅ Saved {len(detailed_matches)} matches to {output_file}")
        return detailed_matches
    
    def _extract_essential_data(self, match_data: Dict) -> Dict:
        """Extract essential data only"""
        return {
            'match_id': match_data.get('match_id'),
            'duration': match_data.get('duration'),
            'radiant_win': match_data.get('radiant_win'),
            'skill': match_data.get('skill'),
            'skill_name': self._get_skill_name(match_data.get('skill', 0)),
            'avg_mmr': match_data.get('avg_mmr'),
            'patch': match_data.get('patch'),
            'lobby_type': match_data.get('lobby_type'),
            'game_mode': match_data.get('game_mode'),
            'region': match_data.get('region'),
            'players': [
                {
                    'hero_id': player.get('hero_id'),
                    'player_slot': player.get('player_slot'),
                    'rank_tier': player.get('rank_tier')
                }
                for player in match_data.get('players', [])
            ]
        }
    
    def _get_skill_name(self, skill_level: int) -> str:
        """Convert skill level to name"""
        skill_map = {1: "Normal", 2: "High", 3: "Very High"}
        return skill_map.get(skill_level, "Unknown")
    
    def _is_valid_match(self, match_data: Dict) -> bool:
        """Validate match data"""
        try:
            if not match_data or 'players' not in match_data:
                return False
            
            players = match_data['players']
            if len(players) != 10:
                return False
            
            # Duration check
            duration = match_data.get('duration', 0)
            if duration < 600 or duration > 7200:
                return False
            
            # Must have match outcome
            if 'radiant_win' not in match_data:
                return False
            
            # Check all players have heroes
            for player in players:
                if not player.get('hero_id'):
                    return False
            
            return True
            
        except Exception:
            return False

def main():
    """Run optimized data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Dota 2 data collection')
    parser.add_argument('--matches', type=int, default=1000, help='Total matches to collect')
    
    args = parser.parse_args()
    
    try:
        collector = OptimizedCollector()
        matches = collector.collect_matches_optimized(args.matches)
        
        print(f"\n🎉 Collection completed!")
        print(f"📊 Matches collected: {len(matches)}")
        print(f"⏱️  Total requests: {collector.stats['requests_made']}")
        print(f"🚀 Success rate: {len(matches)/collector.stats['requests_made']*100:.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Collection interrupted")
        return 1
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())