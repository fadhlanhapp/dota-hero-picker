import sys
sys.path.append('src')
from model_training import HeroPickPredictor

# Load trained model
predictor = HeroPickPredictor()
predictor.load_model('models/hero_predictor_random_forest.pkl')

# Test prediction
team_heroes = [1, 5, 10]      # Anti-Mage, Crystal Maiden, Shadow Fiend
enemy_heroes = [15, 20, 25]   # Pudge, Drow Ranger, Lina

recommendations = predictor.predict_best_heroes(team_heroes, enemy_heroes, top_k=5)

print("🎯 Hero Recommendations:")
for rec in recommendations:
    print(f"  {rec['hero_name']}: {rec['win_probability']:.3f} win rate")