python3 fire_predictor.py

Output:

🔥 FOREST FIRE PREDICTION SYSTEM
============================================================

📂 Loading regional datasets...
✅ Bejaia Region: 122 records
✅ Sidi-Bel Abbes Region: 122 records

🔥 COMBINED DATASET: 244 records from 2 regions

📊 Fire Distribution by Region:
                Fires  Total Records  No Fire  Fire Rate %
Region
Bejaia             59            122       63         48.4
Sidi-Bel Abbes     78            122       44         63.9

🔥 OVERALL:
  Total Fires: 137
  No Fires: 107
  Fire Rate: 56.1%

📊 Using 244 complete records for training

🔧 Training set: 170 samples
🔧 Testing set: 74 samples

🤖 Training Random Forest AI model on COMBINED data...
🔮 Making predictions...

✅ MODEL ACCURACY: 83.78%

📋 Detailed Classification Report:
              precision    recall  f1-score   support

     No Fire       0.86      0.75      0.80        32
        Fire       0.83      0.90      0.86        42

    accuracy                           0.84        74
   macro avg       0.84      0.83      0.83        74
weighted avg       0.84      0.84      0.84        74


🎯 Confusion Matrix:
                  Predicted No Fire | Predicted Fire
Actually No Fire:        24                   8
Actually Fire:            4                  38

📊 FEATURE IMPORTANCE (What matters most?):
    Feature  Importance
       Rain    0.413956
Temperature    0.302070
         RH    0.199354
         Ws    0.084620

💾 Saved: 'feature_importance.png'
💾 Saved: 'weather_patterns.png'
💾 Saved: 'regional_comparison.png'

============================================================
🧪 TESTING REAL-WORLD SCENARIOS
============================================================

🔥 EXTREME DANGER
   Scorching hot, bone dry, strong winds
   📊 Temp: 40°C | Humidity: 15% | Wind: 30 km/h | Rain: 0mm
   ⚠️  FIRE RISK! (Confidence: 98.0%)

🌧️ RAINY DAY
   Cool, humid, steady rain
   📊 Temp: 18°C | Humidity: 80% | Wind: 10 km/h | Rain: 5mm
   ✅ Low Risk (Confidence: 4.0%)

☀️ MODERATE SUMMER
   Warm, moderate humidity, breezy
   📊 Temp: 28°C | Humidity: 45% | Wind: 15 km/h | Rain: 0mm
   ⚠️  FIRE RISK! (Confidence: 74.0%)

============================================================
🎮 TEST YOUR OWN CONDITIONS!
============================================================

🌡️  Enter Temperature (°C):  10
💧 Enter Humidity (%): 10
💨 Enter Wind Speed (km/h): 10
🌧️  Enter Rain (mm):  10

============================================================
🔮 YOUR PREDICTION:
============================================================
✅ LOW FIRE RISK
The model is 67.0% confident conditions are safe.
✅ Conditions are favorable - low fire danger
============================================================

============================================================
🎉 ENHANCED FOREST FIRE PREDICTION SYSTEM COMPLETE!
============================================================

📊 Generated visualizations:
   1. feature_importance.png - Which factors matter most
   2. weather_patterns.png - Fire vs No-Fire conditions
   3. regional_comparison.png - Bejaia vs Sidi-Bel Abbes analysis
