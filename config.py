import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Settings
    OPENDOTA_API_KEY = os.getenv('OPENDOTA_API_KEY')
    BASE_URL = "https://api.opendota.com/api"
    
    # Rate Limiting
    RATE_LIMIT_DELAY = float(os.getenv('RATE_LIMIT_DELAY', 1))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    TIMEOUT = 30
    
    # Data Collection Targets
    TARGET_MATCHES = int(os.getenv('TARGET_MATCHES', 5000))
    BATCH_SIZE = 100
    
    # Match Filtering
    VALID_GAME_MODES = [2, 3, 22]  # All Pick variants
    VALID_LOBBY_TYPES = [0, 1, 7]  # Public, Practice, Ranked
    MIN_DURATION = int(os.getenv('MIN_DURATION', 600))  # 10 minutes
    MAX_DURATION = int(os.getenv('MAX_DURATION', 7200))  # 2 hours
    
    # File Paths
    DATA_DIR = "data"
    RAW_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DIR = f"{DATA_DIR}/processed"
    LOGS_DIR = f"{DATA_DIR}/logs"
    MODELS_DIR = "models"
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.OPENDOTA_API_KEY or cls.OPENDOTA_API_KEY == 'your_api_key_here':
            print("⚠️  Warning: No OpenDota API key configured. Rate limits may apply.")
        
        if cls.TARGET_MATCHES <= 0:
            raise ValueError("TARGET_MATCHES must be positive")
        
        print("✅ Configuration validated successfully")
        return True
