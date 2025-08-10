#!/usr/bin/env python3
"""
Dota 2 Hero Picker - Enhanced Data Collection with Skill Brackets
Collects match data from OpenDota API including skill levels for better ML training
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
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class SkillAwareCollector:
    """Enhanced OpenDota API data collector with skill bracket awareness"""
    
    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        
        # Enhanced rate limiting for skill data
        self.rate_limit_delay = 0.2  # 200ms between requests
        self.batch_size = 100
        
        # Statistics tracking
        self.stats = {
            'requests_made': 0,
            'matches_collected': 0,
            'matches_filtered': 0,
            'skill_brackets_found': {},
            'start_time': datetime.now()
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Skill-Aware Collector initialized")
    
    def _make_api_call(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Make API call with enhanced error handling and skill data focus"""
        url = f"https://api.opendota.com/api/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            self.stats['requests_made'] += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait longer
                self.logger.warning("Rate limited, waiting 5 seconds...")
                time.sleep(5)
                return None
            else:
                self.logger.debug(f"API call failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.debug(f"API call error: {e}")
            return None
    
    def get_skill_bracket_name(self, skill_level: int) -> str:
        """Convert skill level integer to readable name"""
        skill_map = {
            1: "Normal",      # 0-3200 MMR approx
            2: "High",        # 3200-3700 MMR approx  
            3: "Very High"    # 3700+ MMR approx
        }
        return skill_map.get(skill_level, "Unknown")
    
    def get_match_ids_by_skill(self, target_matches: int, skill_distribution: dict = None) -> List[int]:
        """
        Get match IDs with balanced skill distribution
        
        Args:
            target_matches: Total matches to collect
            skill_distribution: Dict like {"Normal": 0.4, "High": 0.4, "Very High": 0.2}
        """
        if not skill_distribution:
            # Default balanced distribution
            skill_distribution = {
                "Normal": 0.4,    # 40% normal skill
                "High": 0.4,      # 40% high skill  
                "Very High": 0.2  # 20% very high skill
            }
        
        self.logger.info(f"Collecting match IDs with skill distribution: {skill_distribution}")
        
        match_ids = []
        skill_targets = {
            skill: int(target_matches * ratio) 
            for skill, ratio in skill_distribution.items()
        }
        
        self.logger.info(f"Target matches per skill: {skill_targets}")
        
        # Start from recent matches and work backwards
        max_match_id = 8500000000  # Approximate recent match ID
        search_range = 1000000     # Search within this range
        
        # Collect match IDs for each skill bracket
        for skill_name, target_count in skill_targets.items():
            skill_level = {"Normal": 1, "High": 2, "Very High": 3}[skill_name]
            collected_for_skill = 0
            attempts = 0
            max_attempts = target_count * 10  # Try 10x more than needed
            
            self.logger.info(f"Collecting {target_count} {skill_name} skill matches...")
            
            pbar = tqdm(total=target_count, desc=f"Collecting {skill_name} matches")
            
            while collected_for_skill < target_count and attempts < max_attempts:
                # Random match ID in recent range
                match_id = random.randint(max_match_id - search_range, max_match_id)
                
                # Quick check if this match has the right skill level
                match_data = self._make_api_call(f"matches/{match_id}")
                attempts += 1
                
                if match_data and self._is_valid_match_with_skill(match_data, skill_level):
                    match_ids.append(match_id)
                    collected_for_skill += 1
                    pbar.update(1)
                    
                    # Update skill bracket stats
                    self.stats['skill_brackets_found'][skill_name] = \
                        self.stats['skill_brackets_found'].get(skill_name, 0) + 1
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                if attempts % 100 == 0:
                    self.logger.info(f"  {skill_name}: {collected_for_skill}/{target_count} found ({attempts} attempts)")
            
            pbar.close()
            self.logger.info(f"✅ Collected {collected_for_skill} {skill_name} skill matches")
        
        self.logger.info(f"🎯 Total match IDs collected: {len(match_ids)}")
        return match_ids
    
    def _is_valid_match_with_skill(self, match_data: Dict, target_skill_level: int) -> bool:
        """Validate match has correct skill level and basic requirements"""
        try:
            # Basic validation
            if not match_data or 'players' not in match_data:
                return False
            
            players = match_data['players']
            if len(players) != 10:
                return False
            
            # Duration check (at least 10 minutes)
            duration = match_data.get('duration', 0)
            if duration < 600:
                return False
            
            # Skill level check
            skill_level = match_data.get('skill')
            if skill_level != target_skill_level:
                return False
            
            # Check all players have hero_id
            for player in players:
                if not player.get('hero_id'):
                    return False
            
            # Must have radiant_win result
            if 'radiant_win' not in match_data:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Match validation error: {e}")
            return False
    
    def collect_detailed_matches_with_skill(self, match_ids: List[int]) -> List[Dict]:
        """
        Collect detailed match data with skill information
        """
        self.logger.info(f"💰 Estimated cost: ${len(match_ids) * 0.0001:.3f}")
        
        detailed_matches = []
        failed_matches = []
        
        # Progress bar
        progress_bar = tqdm(
            match_ids,
            desc="Fetching match details",
            unit="match/s"
        )
        
        for i, match_id in enumerate(progress_bar):
            match_data = self._make_api_call(f"matches/{match_id}")
            
            if match_data and self._validate_detailed_match(match_data):
                # Extract essential data with skill information
                essential_match = self._extract_essential_match_data(match_data)
                detailed_matches.append(essential_match)
                self.stats['matches_collected'] += 1
            else:
                failed_matches.append(match_id)
                self.stats['matches_filtered'] += 1
            
            # Update progress
            progress_bar.set_postfix({
                'valid': len(detailed_matches),
                'failed': len(failed_matches),
                'success_rate': f"{len(detailed_matches)/(i+1)*100:.1f}%"
            })
            
            # Save intermediate results every 100 matches
            if (i + 1) % 100 == 0:
                self._save_intermediate_results(detailed_matches, i + 1)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        # Save final results
        matches_file = f"{self.config.RAW_DIR}/detailed_matches_with_skill.json"
        with open(matches_file, 'w') as f:
            json.dump(detailed_matches, f)  # No indentation for smaller size
        
        # Save failed matches for analysis
        if failed_matches:
            failed_file = f"{self.config.RAW_DIR}/failed_matches.json"
            with open(failed_file, 'w') as f:
                json.dump(failed_matches, f)
        
        self.logger.info(f"✅ Collected {len(detailed_matches)} valid matches")
        self.logger.info(f"❌ {len(failed_matches)} matches filtered out")
        
        return detailed_matches
    
    def _extract_essential_match_data(self, match_data: Dict) -> Dict:
        """Extract only essential data needed for ML training"""
        return {
            'match_id': match_data.get('match_id'),
            'duration': match_data.get('duration'),
            'radiant_win': match_data.get('radiant_win'),
            'skill': match_data.get('skill'),  # Skill bracket (1=Normal, 2=High, 3=Very High)
            'skill_name': self.get_skill_bracket_name(match_data.get('skill', 0)),
            'avg_mmr': match_data.get('avg_mmr'),  # Average MMR if available
            'lobby_type': match_data.get('lobby_type'),  # Game mode
            'game_mode': match_data.get('game_mode'),
            'patch': match_data.get('patch'),  # Game version
            'region': match_data.get('region'),  # Server region
            'start_time': match_data.get('start_time'),
            'players': [
                {
                    'hero_id': player.get('hero_id'),
                    'player_slot': player.get('player_slot'),
                    'rank_tier': player.get('rank_tier'),  # Individual player rank
                    'wins': player.get('wins'),  # Player's total wins
                    'lose': player.get('lose'),   # Player's total losses
                    'level': player.get('level'),  # Account level
                }
                for player in match_data.get('players', [])
            ]
        }
    
    def _validate_detailed_match(self, match_data: Dict) -> bool:
        """Comprehensive validation for detailed match data"""
        try:
            # Basic structure
            if not match_data or 'players' not in match_data:
                return False
            
            players = match_data['players']
            if len(players) != 10:
                return False
            
            # Duration check
            duration = match_data.get('duration', 0)
            if duration < 600 or duration > 7200:  # 10 min to 2 hours
                return False
            
            # Must have skill bracket
            if not match_data.get('skill'):
                return False
            
            # Must have match outcome
            if 'radiant_win' not in match_data:
                return False
            
            # Validate players
            radiant_count = dire_count = 0
            for player in players:
                if not player.get('hero_id'):
                    return False
                
                player_slot = player.get('player_slot', 0)
                if player_slot < 128:
                    radiant_count += 1
                else:
                    dire_count += 1
            
            # Must have 5v5
            if radiant_count != 5 or dire_count != 5:
                return False
            
            # Filter out practice/lobby games
            lobby_type = match_data.get('lobby_type', 0)
            if lobby_type not in [0, 7]:  # 0=Public matchmaking, 7=Ranked
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Detailed match validation error: {e}")
            return False
    
    def _save_intermediate_results(self, matches: List[Dict], batch_num: int):
        """Save intermediate results to prevent data loss"""
        temp_file = f"{self.config.RAW_DIR}/matches_temp_skill_{batch_num}.json"
        with open(temp_file, 'w') as f:
            json.dump(matches, f)
        self.logger.debug(f"Saved intermediate results: {len(matches)} matches")
    
    def generate_collection_summary(self, matches: List[Dict]) -> Dict:
        """Generate comprehensive collection summary with skill statistics"""
        end_time = datetime.now()
        duration = end_time - self.stats['start_time']
        
        # Skill distribution analysis
        skill_distribution = {}
        avg_mmr_by_skill = {}
        rank_distribution = {}
        
        for match in matches:
            skill_name = match.get('skill_name', 'Unknown')
            skill_distribution[skill_name] = skill_distribution.get(skill_name, 0) + 1
            
            # Analyze player ranks in this match
            for player in match.get('players', []):
                rank_tier = player.get('rank_tier')
                if rank_tier:
                    rank_name = self._get_rank_name(rank_tier)
                    rank_distribution[rank_name] = rank_distribution.get(rank_name, 0) + 1
        
        summary = {
            'collection_completed': end_time.isoformat(),
            'duration_formatted': str(duration),
            'total_requests': self.stats['requests_made'],
            'matches_collected': len(matches),
            'matches_filtered': self.stats['matches_filtered'],
            'success_rate': (len(matches) / max(self.stats['requests_made'], 1)) * 100,
            'matches_per_hour': len(matches) / (duration.total_seconds() / 3600),
            'estimated_cost': len(matches) * 0.0001,
            'skill_distribution': skill_distribution,
            'rank_distribution': rank_distribution,
            'unique_heroes': len(set(
                player['hero_id'] 
                for match in matches 
                for player in match.get('players', [])
                if player.get('hero_id')
            )),
            'avg_duration': sum(match.get('duration', 0) for match in matches) / len(matches),
            'radiant_win_rate': sum(1 for match in matches if match.get('radiant_win')) / len(matches)
        }
        
        return summary
    
    def _get_rank_name(self, rank_tier: int) -> str:
        """Convert rank tier to readable name"""
        if not rank_tier:
            return "Unranked"
        
        rank_names = {
            1: "Herald", 2: "Guardian", 3: "Crusader", 4: "Archon",
            5: "Legend", 6: "Ancient", 7: "Divine", 8: "Immortal"
        }
        
        tier = (rank_tier // 10) if rank_tier >= 10 else 1
        return rank_names.get(tier, "Unknown")

def main():
    """CLI interface for skill-aware data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect Dota 2 match data with skill bracket awareness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_collection_with_skill.py --matches 20000                           # Balanced skill distribution
  python data_collection_with_skill.py --matches 10000 --skill-focus high       # Focus on high skill  
  python data_collection_with_skill.py --matches 5000 --custom-distribution     # Custom distribution
        """
    )
    
    parser.add_argument(
        '--matches', 
        type=int, 
        default=10000,
        help='Total number of matches to collect'
    )
    
    parser.add_argument(
        '--skill-focus',
        choices=['normal', 'high', 'very_high', 'balanced'],
        default='balanced',
        help='Skill bracket focus'
    )
    
    parser.add_argument(
        '--custom-distribution',
        action='store_true',
        help='Use custom skill distribution (equal parts all skills)'
    )
    
    args = parser.parse_args()
    
    # Setup skill distribution
    if args.skill_focus == 'normal':
        distribution = {"Normal": 0.8, "High": 0.15, "Very High": 0.05}
    elif args.skill_focus == 'high':
        distribution = {"Normal": 0.2, "High": 0.6, "Very High": 0.2}
    elif args.skill_focus == 'very_high':
        distribution = {"Normal": 0.1, "High": 0.3, "Very High": 0.6}
    elif args.custom_distribution:
        distribution = {"Normal": 0.33, "High": 0.33, "Very High": 0.34}
    else:  # balanced
        distribution = {"Normal": 0.4, "High": 0.4, "Very High": 0.2}
    
    try:
        print(f"🚀 Starting skill-aware data collection...")
        print(f"🎯 Target matches: {args.matches:,}")
        print(f"🏆 Skill distribution: {distribution}")
        print(f"💰 Estimated cost: ${args.matches * 0.0001:.2f}")
        
        # Initialize collector
        collector = SkillAwareCollector()
        
        # Collect match IDs with skill awareness
        match_ids = collector.get_match_ids_by_skill(args.matches, distribution)
        
        if not match_ids:
            print("❌ No match IDs collected. Check your network connection.")
            return 1
        
        print(f"📊 Collected {len(match_ids)} match IDs")
        
        # Collect detailed match data
        detailed_matches = collector.collect_detailed_matches_with_skill(match_ids)
        
        if not detailed_matches:
            print("❌ No detailed matches collected.")
            return 1
        
        # Generate summary
        summary = collector.generate_collection_summary(detailed_matches)
        
        # Save summary
        config = Config()
        summary_file = f"{config.PROCESSED_DIR}/collection_summary_skill.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print results
        print(f"\n🎉 Collection completed successfully!")
        print(f"📊 Matches collected: {summary['matches_collected']:,}")
        print(f"🏆 Skill distribution:")
        for skill, count in summary['skill_distribution'].items():
            percentage = (count / summary['matches_collected']) * 100
            print(f"   {skill}: {count:,} ({percentage:.1f}%)")
        print(f"⚡ Success rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Duration: {summary['duration_formatted']}")
        print(f"💰 Actual cost: ${summary['estimated_cost']:.3f}")
        print(f"📁 Data saved to: data/raw/detailed_matches_with_skill.json")
        print(f"📋 Summary saved to: {summary_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Collection interrupted by user")
        print(f"📊 Partial data may be saved in temp files")
        return 1
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())