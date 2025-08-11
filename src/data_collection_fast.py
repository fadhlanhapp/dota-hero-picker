#!/usr/bin/env python3
"""
High-Speed Dota 2 Data Collection
Maximizes OpenDota API rate limits (1200 calls/minute = 20/second)
"""

import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import threading

class HighSpeedCollector:
    """High-speed collector maximizing API rate limits"""
    
    def __init__(self):
        # Rate limiting: 1200 calls/minute = 20/second
        self.max_calls_per_second = 2  # Slightly under limit for safety
        self.delay_between_calls = 1.0 / self.max_calls_per_second  # ~0.055 seconds
        
        self.session = requests.Session()
        
        # Thread-safe statistics
        self.stats_lock = threading.Lock()
        self.stats = {
            'requests_made': 0,
            'matches_collected': 0,
            'rate_limited': 0,
            'start_time': datetime.now()
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"High-speed collector initialized - {self.max_calls_per_second} calls/second max")
    
    def _make_api_call(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Thread-safe API call with proper rate limiting"""
        url = f"https://api.opendota.com/api/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            with self.stats_lock:
                self.stats['requests_made'] += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                with self.stats_lock:
                    self.stats['rate_limited'] += 1
                self.logger.warning("Rate limited - backing off")
                time.sleep(2)  # Brief backoff
                return None
            else:
                return None
                
        except Exception as e:
            return None
    
    def get_match_ids_fast(self, count: int) -> List[int]:
        """
        Get match IDs fast using publicMatches endpoint
        Much more efficient than random guessing
        """
        self.logger.info(f"Collecting {count} match IDs using fast method...")
        
        match_ids = set()  # Use set to avoid duplicates
        page_size = 100  # publicMatches returns ~100 matches per call
        
        pbar = tqdm(total=count, desc="Collecting match IDs")
        less_than_match_id = None
        
        while len(match_ids) < count:
            # Calculate how many more pages we need
            remaining = count - len(match_ids)
            
            # Get public matches
            params = {}
            if less_than_match_id:
                params['less_than_match_id'] = less_than_match_id
            
            public_matches = self._make_api_call('publicMatches', params)
            
            if not public_matches:
                time.sleep(1)
                continue
            
            # Extract match IDs
            new_ids = 0
            for match in public_matches:
                match_id = match.get('match_id')
                if match_id and match_id not in match_ids:
                    match_ids.add(match_id)
                    new_ids += 1
                    pbar.update(1)
                    
                    if len(match_ids) >= count:
                        break
            
            # Pagination
            if public_matches:
                less_than_match_id = public_matches[-1].get('match_id')
            
            # Rate limiting
            time.sleep(self.delay_between_calls)
            
            if new_ids == 0:  # Avoid infinite loop
                self.logger.warning("No new matches found, ending collection")
                break
        
        pbar.close()
        match_ids_list = list(match_ids)[:count]
        self.logger.info(f"✅ Collected {len(match_ids_list)} unique match IDs")
        
        # Save match IDs to file for future use
        os.makedirs('data/raw', exist_ok=True)
        with open('data/raw/match_ids.json', 'w') as f:
            json.dump(match_ids_list, f)
        self.logger.info(f"💾 Saved match IDs to data/raw/match_ids.json")
        
        return match_ids_list
    
    def collect_match_details_parallel(self, match_ids: List[int], max_workers: int = 20) -> List[Dict]:
        """
        Collect match details using parallel threads
        """
        self.logger.info(f"Collecting details for {len(match_ids)} matches using {max_workers} threads...")
        
        detailed_matches = []
        failed_matches = []
        
        def fetch_match_detail(match_id: int) -> Optional[Dict]:
            """Fetch single match detail"""
            # Rate limiting per thread
            time.sleep(self.delay_between_calls)
            
            match_data = self._make_api_call(f"matches/{match_id}")
            
            if match_data and self._is_valid_match(match_data):
                return self._extract_essential_data(match_data)
            return None
        
        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_match_id = {
                executor.submit(fetch_match_detail, match_id): match_id 
                for match_id in match_ids
            }
            
            # Progress bar
            pbar = tqdm(total=len(match_ids), desc="Fetching match details")
            
            # Process completed requests
            for future in as_completed(future_to_match_id):
                match_id = future_to_match_id[future]
                
                try:
                    result = future.result()
                    if result:
                        detailed_matches.append(result)
                        with self.stats_lock:
                            self.stats['matches_collected'] += 1
                    else:
                        failed_matches.append(match_id)
                except Exception as e:
                    failed_matches.append(match_id)
                
                pbar.update(1)
                pbar.set_postfix({
                    'valid': len(detailed_matches),
                    'failed': len(failed_matches),
                    'success_rate': f"{len(detailed_matches)/(len(detailed_matches)+len(failed_matches))*100:.1f}%"
                })
        
        pbar.close()
        self.logger.info(f"✅ Collected {len(detailed_matches)} valid matches")
        self.logger.info(f"❌ {len(failed_matches)} matches failed")
        
        return detailed_matches
    
    def _extract_essential_data(self, match_data: Dict) -> Dict:
        """Extract minimal essential data"""
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
            'start_time': match_data.get('start_time'),
            'players': [
                {
                    'hero_id': player.get('hero_id'),
                    'player_slot': player.get('player_slot'),
                    'rank_tier': player.get('rank_tier'),
                }
                for player in match_data.get('players', [])
            ]
        }
    
    def _get_skill_name(self, skill_level: int) -> str:
        """Convert skill level to name"""
        skill_map = {1: "Normal", 2: "High", 3: "Very High"}
        return skill_map.get(skill_level, "Unknown")
    
    def _is_valid_match(self, match_data: Dict) -> bool:
        """Quick match validation"""
        try:
            # Basic checks only (fast)
            return (
                match_data and
                'players' in match_data and
                len(match_data['players']) == 10 and
                match_data.get('duration', 0) >= 600 and
                'radiant_win' in match_data and
                all(player.get('hero_id') for player in match_data['players'])
            )
        except Exception:
            return False
    
    def collect_fast(self, total_matches: int) -> List[Dict]:
        """
        Fast collection pipeline
        """
        start_time = datetime.now()
        self.logger.info(f"🚀 Starting high-speed collection of {total_matches} matches")
        
        # Step 1: Get match IDs quickly
        match_ids = self.get_match_ids_fast(total_matches)
        
        if not match_ids:
            self.logger.error("Failed to collect match IDs")
            return []
        
        # Step 2: Collect match details in parallel
        detailed_matches = self.collect_match_details_parallel(match_ids)
        
        # Save results
        output_file = "data/raw/detailed_matches_fast.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(detailed_matches, f)  # No indentation = compact
        
        # Statistics
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info(f"🎉 High-speed collection completed!")
        self.logger.info(f"⏱️  Duration: {duration}")
        self.logger.info(f"📊 Total requests: {self.stats['requests_made']}")
        self.logger.info(f"🚀 Average rate: {self.stats['requests_made']/(duration.total_seconds()/60):.1f} calls/minute")
        self.logger.info(f"📈 Success rate: {len(detailed_matches)/len(match_ids)*100:.1f}%")
        self.logger.info(f"💾 Saved to: {output_file}")
        
        # File size
        file_size = os.path.getsize(output_file) / 1024 / 1024
        self.logger.info(f"💾 File size: {file_size:.1f} MB ({file_size/len(detailed_matches)*1000:.1f} KB per match)")
        
        return detailed_matches

def main():
    """Run high-speed data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='High-speed Dota 2 data collection')
    parser.add_argument('--matches', type=int, default=1000, help='Total matches to collect')
    parser.add_argument('--workers', type=int, default=20, help='Parallel workers')
    
    args = parser.parse_args()
    
    try:
        collector = HighSpeedCollector()
        matches = collector.collect_fast(args.matches)
        
        if matches:
            print(f"\n🎉 Success! Collected {len(matches)} matches")
            
            # Quick analysis
            skill_dist = {}
            for match in matches:
                skill = match.get('skill_name', 'Unknown')
                skill_dist[skill] = skill_dist.get(skill, 0) + 1
            
            print(f"🏆 Skill distribution:")
            for skill, count in skill_dist.items():
                print(f"   {skill}: {count} ({count/len(matches)*100:.1f}%)")
        else:
            print("❌ No matches collected")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Collection interrupted")
        return 1
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())