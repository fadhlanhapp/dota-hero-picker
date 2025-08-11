#!/usr/bin/env python3
"""
Collect Match Details from Existing Match IDs
Skips the match ID collection phase and uses what you already have
"""

import json
import requests
import time
from tqdm import tqdm
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads .env file into environment
except ImportError:
    pass  # dotenv not available, use system env vars only

class MatchDetailCollector:
    """Collect match details from existing match IDs with conservative rate limiting"""
    
    def __init__(self):
        # Get API key from environment
        self.api_key = os.getenv('OPENDOTA_API_KEY')
        
        # High-speed rate limiting with API key  
        self.delay_between_calls = 0.05 if self.api_key else 0.5  # 20 calls/sec with API key
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
        
        if self.api_key:
            self.logger.info(f"🔑 Using API key (ending in ...{self.api_key[-4:]}) - faster collection enabled")
        else:
            self.logger.warning("⚠️ No API key found in OPENDOTA_API_KEY env var - using conservative rate limits")
    
    def load_existing_match_ids(self) -> List[int]:
        """Load match IDs from existing files"""
        self.logger.info("🔍 Looking for existing match IDs...")
        
        all_match_ids = set()
        
        # Look for various file patterns
        patterns = [
            'data/raw/detailed_matches*.json',
            'data/raw/match_ids.json',
            'match_ids*.json',
            '*matches*.json'
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract match IDs based on file structure
                    if isinstance(data, list):
                        # Could be list of match IDs or match objects
                        for item in data:
                            if isinstance(item, int):
                                all_match_ids.add(item)
                            elif isinstance(item, dict) and 'match_id' in item:
                                all_match_ids.add(item['match_id'])
                    elif isinstance(data, dict):
                        # Could be a dict with matches key
                        if 'matches' in data:
                            for match in data['matches']:
                                if 'match_id' in match:
                                    all_match_ids.add(match['match_id'])
                        elif 'match_ids' in data:
                            all_match_ids.update(data['match_ids'])
                    
                    self.logger.info(f"   📄 Found {len(all_match_ids)} unique IDs in {file}")
                    
                except Exception as e:
                    self.logger.debug(f"   ⚠️  Couldn't read {file}: {e}")
        
        match_ids = list(all_match_ids)
        self.logger.info(f"📊 Total unique match IDs found: {len(match_ids)}")
        return match_ids
    
    def save_match_ids(self, match_ids: List[int]):
        """Save match IDs to a separate file for future use"""
        os.makedirs('data/raw', exist_ok=True)
        with open('data/raw/match_ids_backup.json', 'w') as f:
            json.dump(match_ids, f)
        self.logger.info(f"💾 Saved {len(match_ids)} match IDs to match_ids_backup.json")
    
    def _make_api_call(self, endpoint: str) -> Optional[Dict]:
        """Make API call with conservative rate limiting"""
        url = f"https://api.opendota.com/api/{endpoint}"
        
        # Add API key if available
        params = {}
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            with self.stats_lock:
                self.stats['requests_made'] += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                with self.stats_lock:
                    self.stats['rate_limited'] += 1
                self.logger.warning(f"Rate limited (total: {self.stats['rate_limited']}), waiting 10 seconds...")
                time.sleep(10)  # Shorter wait on rate limit
                return None
            elif response.status_code == 404:
                return None  # Match not found
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"API error: {e}")
            return None
    
    def collect_match_details(self, match_ids: List[int], resume_from: int = 0, max_workers: int = 5) -> List[Dict]:
        """
        Collect match details with parallel processing and resume capability
        
        Args:
            match_ids: List of match IDs to collect
            resume_from: Index to resume from (for interrupted collections)
            max_workers: Number of parallel threads
        """
        self.logger.info(f"📥 Collecting details for {len(match_ids)} matches using {max_workers} threads...")
        
        if resume_from > 0:
            self.logger.info(f"📌 Resuming from match index {resume_from}")
            match_ids = match_ids[resume_from:]
        
        detailed_matches = []
        failed_matches = []
        
        # Load any existing collected data
        existing_file = 'data/raw/detailed_matches_incremental.json'
        if os.path.exists(existing_file):
            try:
                with open(existing_file, 'r') as f:
                    detailed_matches = json.load(f)
                self.logger.info(f"📂 Loaded {len(detailed_matches)} existing matches")
                
                # Skip already collected matches
                collected_ids = {m['match_id'] for m in detailed_matches}
                match_ids = [m for m in match_ids if m not in collected_ids]
                self.logger.info(f"📊 Remaining to collect: {len(match_ids)}")
            except:
                pass
        
        def fetch_match_detail(match_id: int) -> Optional[Dict]:
            """Fetch single match detail with rate limiting"""
            time.sleep(self.delay_between_calls)  # Rate limiting per thread
            
            match_data = self._make_api_call(f"matches/{match_id}")
            
            if match_data and self._is_valid_match(match_data):
                with self.stats_lock:
                    self.stats['matches_collected'] += 1
                return self._extract_essential_data(match_data)
            return None
        
        # Use ThreadPoolExecutor for parallel processing
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
                
                # Save periodically (every 100 matches)
                if len(detailed_matches) % 100 == 0:
                    self._save_incremental(detailed_matches)
        
        pbar.close()
        
        # Final save
        self._save_incremental(detailed_matches)
        
        # Save failed matches for retry
        if failed_matches:
            with open('data/raw/failed_match_ids.json', 'w') as f:
                json.dump(failed_matches, f)
            self.logger.info(f"💾 Saved {len(failed_matches)} failed match IDs for retry")
        
        return detailed_matches
    
    def _save_incremental(self, matches: List[Dict]):
        """Save matches incrementally"""
        os.makedirs('data/raw', exist_ok=True)
        with open('data/raw/detailed_matches_incremental.json', 'w') as f:
            json.dump(matches, f)
    
    def _extract_essential_data(self, match_data: Dict) -> Dict:
        """Extract only essential data"""
        return {
            'match_id': match_data.get('match_id'),
            'duration': match_data.get('duration'),
            'radiant_win': match_data.get('radiant_win'),
            'skill': match_data.get('skill'),
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
    
    def _is_valid_match(self, match_data: Dict) -> bool:
        """Quick validation"""
        try:
            return (
                match_data and
                'players' in match_data and
                len(match_data['players']) == 10 and
                match_data.get('duration', 0) >= 600 and
                'radiant_win' in match_data
            )
        except:
            return False

def main():
    """Main collection process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect match details from existing IDs')
    parser.add_argument('--limit', type=int, help='Limit number of matches to collect')
    parser.add_argument('--resume', type=int, default=0, help='Resume from index')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    try:
        collector = MatchDetailCollector()
        
        # Load existing match IDs
        match_ids = collector.load_existing_match_ids()
        
        if not match_ids:
            print("❌ No match IDs found! Please run data collection first.")
            return 1
        
        # Save backup of IDs
        collector.save_match_ids(match_ids)
        
        # Limit if specified
        if args.limit:
            match_ids = match_ids[:args.limit]
            print(f"📊 Limited to {args.limit} matches")
        
        # Collect details
        matches = collector.collect_match_details(match_ids, args.resume, args.workers)
        
        # Summary
        duration = datetime.now() - collector.stats['start_time']
        print(f"\n🎉 Collection completed!")
        print(f"📊 Matches collected: {collector.stats['matches_collected']}")
        print(f"⚠️  Rate limited events: {collector.stats['rate_limited']}")
        print(f"⏱️  Duration: {duration}")
        print(f"📁 Data saved: data/raw/detailed_matches_incremental.json")
        
        # File size
        if os.path.exists('data/raw/detailed_matches_incremental.json'):
            size_mb = os.path.getsize('data/raw/detailed_matches_incremental.json') / 1024 / 1024
            print(f"💾 File size: {size_mb:.1f} MB")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Collection interrupted - progress saved")
        print(f"📌 Resume with: python3 {__file__} --resume {collector.stats['matches_collected']}")
        return 1
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())