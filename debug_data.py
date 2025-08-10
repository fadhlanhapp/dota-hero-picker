import pandas as pd

print("🔍 Debugging your data...")

# Load raw features
df = pd.read_csv('data/processed/features_raw.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns[:10])}")  # First 10 columns

if 'target_hero' in df.columns:
    print(f"Target hero column found!")
    print(f"Unique target heroes: {df['target_hero'].nunique()}")
    print(f"Target value range: {df['target_hero'].min()} to {df['target_hero'].max()}")
    print(f"Target distribution (top 10):")
    print(df['target_hero'].value_counts().head(10))
    
    # Check for any NaN or invalid values
    print(f"Missing values in target: {df['target_hero'].isna().sum()}")
    print(f"Zero values in target: {(df['target_hero'] == 0).sum()}")
else:
    print("❌ target_hero column NOT FOUND!")
    print("Available columns:", list(df.columns))

# Check a few sample rows
print(f"\nFirst 3 rows:")
print(df.head(3))