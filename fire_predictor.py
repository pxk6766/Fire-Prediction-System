# Multi-region Forest Fire Prediction System
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🔥 FOREST FIRE PREDICTION SYSTEM")
print("=" * 60)

print("\n📂 Loading regional datasets...")

# Bejaia Region (Northeast)
bejaia = pd.read_csv('Bejaia Region ForestFire Dataset.csv')
bejaia.columns = bejaia.columns.str.strip()
bejaia['Region'] = 'Bejaia'
print(f"✅ Bejaia Region: {len(bejaia)} records")

# Sidi-Bel Abbes Region (Northwest)
sidi = pd.read_csv('Sidi-Bel Abbes Region ForestFire Dataset.csv')
sidi.columns = sidi.columns.str.strip()
sidi['Region'] = 'Sidi-Bel Abbes'
print(f"✅ Sidi-Bel Abbes Region: {len(sidi)} records")

# Combining both regions
data = pd.concat([bejaia, sidi], ignore_index=True)
print(f"\n🔥 COMBINED DATASET: {len(data)} records from 2 regions")

# Creating binary target
data['fire_occurred'] = data['Classes'].apply(
    lambda x: 1 if 'fire' in str(x).lower() and 'not' not in str(x).lower() else 0
)

print("\n📊 Fire Distribution by Region:")
region_summary = data.groupby('Region')['fire_occurred'].agg(['sum', 'count'])
region_summary.columns = ['Fires', 'Total Records']
region_summary['No Fire'] = region_summary['Total Records'] - region_summary['Fires']
region_summary['Fire Rate %'] = (region_summary['Fires'] / region_summary['Total Records'] * 100).round(1)
print(region_summary)

print(f"\n🔥 OVERALL:")
print(f"  Total Fires: {data['fire_occurred'].sum()}")
print(f"  No Fires: {len(data) - data['fire_occurred'].sum()}")
print(f"  Fire Rate: {(data['fire_occurred'].sum() / len(data) * 100):.1f}%")

# Selecting features
feature_columns = ['Temperature', 'RH', 'Ws', 'Rain']
data_clean = data[feature_columns + ['fire_occurred', 'Region']].dropna()

print(f"\n📊 Using {len(data_clean)} complete records for training")

X = data_clean[feature_columns]
y = data_clean['fire_occurred']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n🔧 Training set: {len(X_train)} samples")
print(f"🔧 Testing set: {len(X_test)} samples")

# Training model
print("\n🤖 Training Random Forest AI model on COMBINED data...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
print("🔮 Making predictions...")
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ MODEL ACCURACY: {accuracy * 100:.2f}%")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Fire', 'Fire']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🎯 Confusion Matrix:")
print(f"                  Predicted No Fire | Predicted Fire")
print(f"Actually No Fire:        {cm[0][0]:2d}                  {cm[0][1]:2d}")
print(f"Actually Fire:           {cm[1][0]:2d}                  {cm[1][1]:2d}")

# Feature Importance
importances = model.feature_importances_
feature_imp = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n📊 FEATURE IMPORTANCE (What matters most?):")
print(feature_imp.to_string(index=False))

# VISUALIZATION 1: Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_imp['Feature'], feature_imp['Importance'], color='orangered')
plt.xlabel('Importance Score', fontsize=12)
plt.title('🔥 Which Weather Conditions Predict Forest Fires?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: 'feature_importance.png'")

# VISUALIZATION 2: Weather Patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🌡️ Weather Conditions: Fire vs No Fire (Both Regions Combined)', 
             fontsize=16, fontweight='bold')

for idx, feature in enumerate(feature_columns):
    ax = axes[idx//2, idx%2]
    
    fire_data = data_clean[data_clean['fire_occurred'] == 1][feature]
    no_fire_data = data_clean[data_clean['fire_occurred'] == 0][feature]
    
    ax.hist([no_fire_data, fire_data], label=['No Fire', 'Fire'], 
            bins=20, color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('weather_patterns.png', dpi=300, bbox_inches='tight')
print("💾 Saved: 'weather_patterns.png'")

# VISUALIZATION 3: Regional Comparison (NEW!)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🗺️ Regional Weather Comparison: Bejaia vs Sidi-Bel Abbes', 
             fontsize=16, fontweight='bold')

for idx, feature in enumerate(feature_columns):
    ax = axes[idx//2, idx%2]
    
    bejaia_data = data_clean[data_clean['Region'] == 'Bejaia'][feature]
    sidi_data = data_clean[data_clean['Region'] == 'Sidi-Bel Abbes'][feature]
    
    ax.hist([bejaia_data, sidi_data], label=['Bejaia (NE)', 'Sidi-Bel Abbes (NW)'], 
            bins=20, color=['blue', 'purple'], alpha=0.6, edgecolor='black')
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{feature} by Region', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('regional_comparison.png', dpi=300, bbox_inches='tight')
print("💾 Saved: 'regional_comparison.png'")

# TESTING SCENARIOS
print("\n" + "="*60)
print("🧪 TESTING REAL-WORLD SCENARIOS")
print("="*60)

scenarios = [
    {
        'name': '🔥 EXTREME DANGER',
        'temp': 40, 'humidity': 15, 'wind': 30, 'rain': 0,
        'desc': 'Scorching hot, bone dry, strong winds'
    },
    {
        'name': '🌧️ RAINY DAY',
        'temp': 18, 'humidity': 80, 'wind': 10, 'rain': 5,
        'desc': 'Cool, humid, steady rain'
    },
    {
        'name': '☀️ MODERATE SUMMER',
        'temp': 28, 'humidity': 45, 'wind': 15, 'rain': 0,
        'desc': 'Warm, moderate humidity, breezy'
    }
]

for scenario in scenarios:
    test = pd.DataFrame({
        'Temperature': [scenario['temp']],
        'RH': [scenario['humidity']],
        'Ws': [scenario['wind']],
        'Rain': [scenario['rain']]
    })
    
    pred = model.predict(test)[0]
    prob = model.predict_proba(test)[0][1] * 100
    
    print(f"\n{scenario['name']}")
    print(f"   {scenario['desc']}")
    print(f"   📊 Temp: {scenario['temp']}°C | Humidity: {scenario['humidity']}% | "
          f"Wind: {scenario['wind']} km/h | Rain: {scenario['rain']}mm")
    print(f"   {'⚠️  FIRE RISK!' if pred == 1 else '✅ Low Risk'} (Confidence: {prob:.1f}%)")

# Interactive prediction
print("\n" + "="*60)
print("🎮 TEST YOUR OWN CONDITIONS!")
print("="*60)

try:
    temp = float(input("\n🌡️  Enter Temperature (°C): "))
    humidity = float(input("💧 Enter Humidity (%): "))
    wind = float(input("💨 Enter Wind Speed (km/h): "))
    rain = float(input("🌧️  Enter Rain (mm): "))
    
    custom_test = pd.DataFrame({
        'Temperature': [temp],
        'RH': [humidity],
        'Ws': [wind],
        'Rain': [rain]
    })
    
    custom_pred = model.predict(custom_test)[0]
    custom_prob = model.predict_proba(custom_test)[0][1] * 100
    
    print("\n" + "="*60)
    print("🔮 YOUR PREDICTION:")
    print("="*60)
    if custom_pred == 1:
        print("⚠️  🔥 FIRE RISK DETECTED!")
        print(f"The model is {custom_prob:.1f}% confident there's fire risk.")
        print("⚠️  Recommended actions:")
        print("   - Issue fire weather warning")
        print("   - Increase firefighter readiness")
        print("   - Restrict outdoor burning activities")
    else:
        print("✅ LOW FIRE RISK")
        print(f"The model is {100-custom_prob:.1f}% confident conditions are safe.")
        print("✅ Conditions are favorable - low fire danger")
    print("="*60)
    
except:
    print("\n⏭️  Skipping interactive mode")

print("\n" + "="*60)
print("🎉 ENHANCED FOREST FIRE PREDICTION SYSTEM COMPLETE!")
print("="*60)
print("\n📊 Generated visualizations:")
print("   1. feature_importance.png - Which factors matter most")
print("   2. weather_patterns.png - Fire vs No-Fire conditions")
print("   3. regional_comparison.png - Bejaia vs Sidi-Bel Abbes analysis")