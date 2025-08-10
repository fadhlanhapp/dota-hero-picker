# Create the full data_collection.py script
#!/usr/bin/env python3
"""
Dota 2 Hero Picker - Data Collection Module
Collects match data from OpenDota API for ML training
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
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class OpenDotaCollector:
    """OpenDota API data collector with validation and persistence"""
    
    def __init__(self):
        self.config = Config()
        self.config.validate_config()
        
        self.session = requests.Session()
        
        # Setup headers - OpenDota uses API key as parameter, not header
        self.api_key = self.config.OPENDOTA_API_KEY
        
        # Setup logging
        self._setup_logging()
        
        # Create directories
        self._create_directories()
        
        # Statistics
        self.stats = {
            'api_calls_made': 0,
            'matches_collected': 0,
            'matches_filtered': 0,
            'errors': 0,
            'start_time': None
        }
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.RAW_DIR,
            self.config.PROCESSED_DIR,
            self.config.LOGS_DIR,
            self.config.MODELS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{self.config.LOGS_DIR}/collection_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def _make_api_call(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited API call with retry logic"""
        url = f"{self.config.BASE_URL}/{endpoint}"
        
        # Add API key to params if available
        if params is None:
            params = {}
        if self.api_key and self.api_key != 'your_api_key_here':
            params['api_key'] = self.api_key
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                self.logger.debug(f"API call: {url}, attempt {attempt + 1}")
                
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.TIMEOUT
                )
                
                self.stats['api_calls_made'] += 1
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = self.config.RATE_LIMIT_DELAY * (attempt + 1) * 2
                    self.logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"API error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed: {e}")
                
            # Wait before retry
            if attempt < self.config.MAX_RETRIES - 1:
                time.sleep(self.config.RATE_LIMIT_DELAY * (attempt + 1))
        
        self.stats['errors'] += 1
        return None
    
    def collect_heroes_data(self) -> List[Dict]:
        """Step 1: Collect all heroes data"""
        self.logger.info("🦸 Collecting heroes data...")
        
        heroes = self._make_api_call("heroes")
        if heroes:
            # Save heroes data
            heroes_file = f"{self.config.RAW_DIR}/heroes.json"
            with open(heroes_file, 'w') as f:
                json.dump(heroes, f, indent=2)
            
            self.logger.info(f"✅ Collected {len(heroes)} heroes")
            return heroes
        else:
            raise Exception("Failed to collect heroes data")
    
    def collect_match_ids(self, target_count: int) -> List[int]:
        """Step 2: Collect public match IDs"""
        self.logger.info(f"🔍 Collecting {target_count} match IDs...")
        
        all_match_ids = []
        last_match_id = None
        batches_needed = (target_count // self.config.BATCH_SIZE) + 1
        
        progress_bar = tqdm(
            range(batches_needed), 
            desc="Fetching match IDs",
            unit="batch"
        )
        
        for batch in progress_bar:
            params = {}
            if last_match_id:
                params['less_than_match_id'] = last_match_id
            
            matches = self._make_api_call("publicMatches", params)
            
            if not matches:
                self.logger.warning(f"No matches returned in batch {batch}")
                continue
            
            # Extract match IDs
            batch_ids = [match['match_id'] for match in matches]
            all_match_ids.extend(batch_ids)
            
            # Update progress
            last_match_id = matches[-1]['match_id']
            progress_bar.set_postfix({
                'total_ids': len(all_match_ids),
                'last_id': last_match_id
            })
            
            # Stop if we have enough
            if len(all_match_ids) >= target_count:
                break
            
            # Rate limiting
            time.sleep(self.config.RATE_LIMIT_DELAY)
        
        # Trim to exact target
        all_match_ids = all_match_ids[:target_count]
        
        # Save match IDs
        ids_file = f"{self.config.RAW_DIR}/match_ids.json"
        with open(ids_file, 'w') as f:
            json.dump(all_match_ids, f, indent=2)
        
        self.logger.info(f"✅ Collected {len(all_match_ids)} match IDs")
        return all_match_ids
    
    def collect_detailed_matches(self, match_ids: List[int]) -> List[Dict]:
        """Step 3: Collect detailed match data with validation"""
        self.logger.info(f"📊 Collecting detailed data for {len(match_ids)} matches...")
        self.logger.info(f"💰 Estimated cost: ${len(match_ids) * 0.0001:.3f}")
        
        detailed_matches = []
        failed_matches = []
        
        progress_bar = tqdm(
            match_ids, 
            desc="Fetching match details",
            unit="match"
        )
        
        for i, match_id in enumerate(progress_bar):
            match_data = self._make_api_call(f"matches/{match_id}")
            
            if match_data and self._is_valid_match(match_data):
                detailed_matches.append(match_data)
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
            time.sleep(self.config.RATE_LIMIT_DELAY)
        
        # Save final results
        matches_file = f"{self.config.RAW_DIR}/detailed_matches.json"
        with open(matches_file, 'w') as f:
            json.dump(detailed_matches, f, indent=2)
        
        # Save failed matches for analysis
        if failed_matches:
            failed_file = f"{self.config.RAW_DIR}/failed_matches.json"
            with open(failed_file, 'w') as f:
                json.dump(failed_matches, f, indent=2)
        
        self.logger.info(f"✅ Collected {len(detailed_matches)} valid matches")
        self.logger.info(f"❌ {len(failed_matches)} matches filtered out")
        
        return detailed_matches
    
    def _is_valid_match(self, match_data: Dict) -> bool:
        """Comprehensive match validation"""
        try:
            # Basic structure validation
            if not match_data or 'players' not in match_data:
                return False
            
            players = match_data['players']
            
            # Must have exactly 10 players
            if len(players) != 10:
                return False
            
            # Check game mode
            game_mode = match_data.get('game_mode')
            if game_mode not in self.config.VALID_GAME_MODES:
                return False
            
            # Check lobby type
            lobby_type = match_data.get('lobby_type')
            if lobby_type not in self.config.VALID_LOBBY_TYPES:
                return False
            
            # Check duration
            duration = match_data.get('duration', 0)
            if duration < self.config.MIN_DURATION or duration > self.config.MAX_DURATION:
                return False
            
            # Check all players have heroes
            for player in players:
                if not player.get('hero_id') or player.get('hero_id') == 0:
                    return False
            
            # Check match outcome
            if 'radiant_win' not in match_data:
                return False
            
            # Check team balance (5 radiant, 5 dire)
            radiant_count = sum(1 for p in players if p.get('player_slot', 0) < 128)
            dire_count = sum(1 for p in players if p.get('player_slot', 0) >= 128)
            
            if radiant_count != 5 or dire_count != 5:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Validation error: {e}")
            return False
    
    def _save_intermediate_results(self, matches: List[Dict], count: int):
        """Save intermediate results to prevent data loss"""
        temp_file = f"{self.config.RAW_DIR}/matches_temp_{count}.json"
        with open(temp_file, 'w') as f:
            json.dump(matches, f, indent=2)
        self.logger.debug(f"Saved intermediate results: {count} matches")
    
    def run_full_collection(self) -> Dict:
        """Run the complete data collection pipeline"""
        self.stats['start_time'] = datetime.now()
        
        try:
            self.logger.info("🚀 Starting data collection pipeline...")
            self.logger.info(f"Target: {self.config.TARGET_MATCHES} matches")
            
            # Step 1: Heroes data
            heroes = self.collect_heroes_data()
            
            # Step 2: Match IDs
            match_ids = self.collect_match_ids(self.config.TARGET_MATCHES)
            
            # Step 3: Detailed match data
            detailed_matches = self.collect_detailed_matches(match_ids)
            
            # Generate summary
            summary = self._generate_summary(detailed_matches)
            
            self.logger.info("✅ Data collection completed successfully!")
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            raise
    
    def _generate_summary(self, matches: List[Dict]) -> Dict:
        """Generate comprehensive collection summary"""
        end_time = datetime.now()
        duration = end_time - self.stats['start_time']
        
        summary = {
            'collection_stats': self.stats.copy(),
            'duration_seconds': duration.total_seconds(),
            'duration_formatted': str(duration),
            'matches_collected': len(matches),
            'estimated_cost': self.stats['api_calls_made'] * 0.0001,
            'success_rate': len(matches) / max(self.stats['api_calls_made'], 1) * 100,
            'matches_per_hour': len(matches) / (duration.total_seconds() / 3600),
            'config_used': {
                'target_matches': self.config.TARGET_MATCHES,
                'rate_limit_delay': self.config.RATE_LIMIT_DELAY,
                'valid_game_modes': self.config.VALID_GAME_MODES,
                'min_duration': self.config.MIN_DURATION,
                'max_duration': self.config.MAX_DURATION
            }
        }
        
        # Save summary
        summary_file = f"{self.config.PROCESSED_DIR}/collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Log summary
        self.logger.info("📊 Collection Summary:")
        self.logger.info(f"   Total API calls: {summary['collection_stats']['api_calls_made']}")
        self.logger.info(f"   Matches collected: {summary['matches_collected']}")
        self.logger.info(f"   Success rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"   Duration: {summary['duration_formatted']}")
        self.logger.info(f"   Estimated cost: ${summary['estimated_cost']:.3f}")
        self.logger.info(f"   Rate: {summary['matches_per_hour']:.1f} matches/hour")
        
        return summary

def main():
    """CLI interface for data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect Dota 2 match data from OpenDota API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_collection.py --test              # Test with 10 matches
  python data_collection.py --matches 1000      # Collect 1000 matches
  python data_collection.py --matches 5000      # Full collection
        """
    )
    
    parser.add_argument(
        '--matches', 
        type=int, 
        default=None,
        help='Number of matches to collect (default from config)'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true', 
        help='Run test mode with 10 matches'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint (if available)'
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.test:
        print("🧪 Running in test mode with 10 matches...")
        Config.TARGET_MATCHES = 10
    elif args.matches:
        Config.TARGET_MATCHES = args.matches
    
    # Create collector and run
    collector = OpenDotaCollector()
    
    try:
        summary = collector.run_full_collection()
        print(f"\n🎉 Success! Collected {summary['matches_collected']} matches")
        print(f"💰 Cost: ${summary['estimated_cost']:.3f}")
        print(f"📁 Data saved to: {collector.config.RAW_DIR}/")
        
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
        print("📁 Partial data may be available in temp files")
        
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())