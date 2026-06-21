"""
Parking Violation Hotspot Prediction - ML Training Pipeline
TEAM-X | R V Sai Charan | Flipkart Grid Lock 2.0 2026

Trains directly on the REAL HackerEarth dataset:
  columns: id, latitude, longitude, location, vehicle_number, vehicle_type,
           description, violation_type, offence_code, created_datetime,
           closed_datetime, modified_datetime, device_id, created_by_id,
           center_code, police_station, data_sent_to_scita, junction_name,
           action_taken_timestamp, data_sent_to_scita_timestamp,
           updated_vehicle_number, updated_vehicle_type, validation_status,
           validation_timestamp

This script:
1. Loads the real violation CSV (any size — tested up to 2,98,450 rows)
2. Engineers time + location + vehicle features
3. Trains a RandomForest classifier to predict HIGH-RISK violation windows
4. Trains a KMeans clustering model to identify geographic hotspots
5. Computes a Congestion Impact Score per zone
6. Saves trained models + exports predictions to JSON for the dashboard
"""

import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "violations.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*70)
print("PARKING VIOLATION HOTSPOT INTELLIGENCE - ML TRAINING PIPELINE")
print("TEAM-X | R V Sai Charan | Flipkart Grid Lock 2.0 2026")
print("="*70)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print(f"\n[1/7] Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"   Loaded {len(df):,} violation records")
print(f"   Columns found: {list(df.columns)}")

# ─────────────────────────────────────────────
# 2. COLUMN AUTO-DETECTION (works with real HackerEarth CSV or sample CSV)
# ─────────────────────────────────────────────
print("\n[2/7] Auto-detecting column mapping...")

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

col_station   = find_col(df, ["police_station", "station"])
col_vehicle   = find_col(df, ["vehicle_type", "updated_vehicle_type"])
col_violation = find_col(df, ["violation_type", "description"])
col_lat       = find_col(df, ["latitude", "location_lat"])
col_lon       = find_col(df, ["longitude", "location_long", "location_lon"])
col_datetime  = find_col(df, ["created_datetime", "created_at", "date", "fine_time"])
col_status    = find_col(df, ["validation_status", "status"])

required = {
    "police_station": col_station, "vehicle_type": col_vehicle,
    "violation_type": col_violation, "latitude": col_lat,
    "longitude": col_lon, "datetime": col_datetime
}
missing = [k for k, v in required.items() if v is None]
if missing:
    raise ValueError(f"Could not auto-detect required columns: {missing}. "
                      f"Available columns: {list(df.columns)}")

print(f"   police_station    -> '{col_station}'")
print(f"   vehicle_type      -> '{col_vehicle}'")
print(f"   violation_type    -> '{col_violation}'")
print(f"   latitude          -> '{col_lat}'")
print(f"   longitude         -> '{col_lon}'")
print(f"   datetime          -> '{col_datetime}'")
print(f"   validation_status -> '{col_status}'" if col_status else "   validation_status -> NOT FOUND (will skip)")

work = pd.DataFrame()
work["police_station"] = df[col_station].astype(str).str.strip()
work["vehicle_type"]   = df[col_vehicle].astype(str).str.strip().str.upper()
work["violation_type"] = df[col_violation].astype(str).str.strip()
work["latitude"]  = pd.to_numeric(df[col_lat], errors="coerce")
work["longitude"] = pd.to_numeric(df[col_lon], errors="coerce")
work["validation_status"] = df[col_status].astype(str).str.strip().str.lower() if col_status else "unknown"

dt_raw = df[col_datetime].astype(str).str.replace(r"\+00$", "", regex=True)
work["datetime"] = pd.to_datetime(dt_raw, errors="coerce")

before = len(work)
work = work.dropna(subset=["police_station", "vehicle_type", "violation_type",
                             "latitude", "longitude", "datetime"])
after = len(work)
print(f"\n   Rows before cleaning: {before:,}")
print(f"   Rows after cleaning:  {after:,}  ({after/before*100:.1f}% retained)")

work["hour"] = work["datetime"].dt.hour
work["day_of_week"] = work["datetime"].dt.dayofweek
work["month"] = work["datetime"].dt.month
work["date"] = work["datetime"].dt.strftime("%Y-%m-%d")

df = work
print(f"\n   FINAL TRAINING SET: {len(df):,} rows")
print(f"   Police stations: {df['police_station'].nunique()}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3/7] Engineering features...")

le_station = LabelEncoder()
le_vehicle = LabelEncoder()
le_violation = LabelEncoder()

df['station_enc'] = le_station.fit_transform(df['police_station'])
df['vehicle_enc'] = le_vehicle.fit_transform(df['vehicle_type'])
df['violation_enc'] = le_violation.fit_transform(df['violation_type'])

df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_peak_morning'] = df['hour'].between(8, 11).astype(int)
df['is_peak_evening'] = df['hour'].between(17, 20).astype(int)
df['is_peak'] = ((df['is_peak_morning'] == 1) | (df['is_peak_evening'] == 1)).astype(int)

hourly_density = df.groupby(['police_station', 'hour']).size().reset_index(name='violation_count')
density_threshold = hourly_density['violation_count'].quantile(0.70)
hourly_density['is_hotspot_window'] = (hourly_density['violation_count'] >= density_threshold).astype(int)

df = df.merge(hourly_density[['police_station', 'hour', 'is_hotspot_window']],
               on=['police_station', 'hour'], how='left')

print(f"   Created {df.shape[1]} features")
print(f"   Hotspot window threshold: {density_threshold:.0f} violations/hour")

# ─────────────────────────────────────────────
# 4. TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
print("\n[4/7] Training Random Forest classifier (hotspot window prediction)...")

feature_cols = ['station_enc', 'vehicle_enc', 'violation_enc', 'hour',
                 'day_of_week', 'month', 'is_weekend', 'is_peak_morning',
                 'is_peak_evening', 'is_peak']
X = df[feature_cols]
y = df['is_hotspot_window']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=150, max_depth=12, min_samples_split=10,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"   Trained on:   {len(X_train):,} rows")
print(f"   Tested on:    {len(X_test):,} rows")
print(f"   Accuracy:     {acc:.4f}")
print(f"   Precision:    {prec:.4f}")
print(f"   Recall:       {rec:.4f}")
print(f"   F1-Score:     {f1:.4f}")

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n   Top feature importances:")
for feat, imp in importances.head(5).items():
    print(f"     {feat}: {imp:.4f}")

# ─────────────────────────────────────────────
# 5. KMEANS CLUSTERING
# ─────────────────────────────────────────────
print("\n[5/7] Running KMeans clustering on GPS coordinates...")

coords = df[['latitude', 'longitude']].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

N_CLUSTERS = min(15, df['police_station'].nunique())
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(coords_scaled)

cluster_summary = df.groupby('cluster').agg(
    violation_count=('police_station', 'count'),
    lat_center=('latitude', 'mean'),
    lon_center=('longitude', 'mean'),
    top_station=('police_station', lambda x: x.mode()[0]),
    top_vehicle=('vehicle_type', lambda x: x.mode()[0]),
).reset_index().sort_values('violation_count', ascending=False)

print(f"   Identified {N_CLUSTERS} geographic clusters from {len(df):,} GPS points")
print(f"   Top cluster: {cluster_summary.iloc[0]['top_station']} ({cluster_summary.iloc[0]['violation_count']:,} violations)")

# ─────────────────────────────────────────────
# 6. CONGESTION IMPACT SCORE
# ─────────────────────────────────────────────
print("\n[6/7] Computing Congestion Impact Scores per station...")

main_road_kw = df['violation_type'].str.contains("MAIN ROAD", case=False, na=False)
footpath_kw = df['violation_type'].str.contains("FOOTPATH", case=False, na=False)

station_stats = df.groupby('police_station').agg(
    total_violations=('police_station', 'count'),
    peak_violations=('is_peak', 'sum'),
    two_wheeler_pct=('vehicle_type', lambda x: x.isin(['SCOOTER', 'MOTOR CYCLE']).mean()),
).reset_index()

road_block = df.assign(road_block=(main_road_kw | footpath_kw).astype(int)) \
                .groupby('police_station')['road_block'].sum().reset_index()
station_stats = station_stats.merge(road_block, on='police_station', how='left')

max_v = station_stats['total_violations'].max()
station_stats['volume_score'] = (station_stats['total_violations'] / max_v) * 40
station_stats['peak_ratio'] = station_stats['peak_violations'] / station_stats['total_violations']
station_stats['peak_score'] = station_stats['peak_ratio'] * 25
station_stats['road_block_score'] = (station_stats['road_block'] / station_stats['total_violations']) * 20
station_stats['vehicle_density_score'] = station_stats['two_wheeler_pct'] * 15

station_stats['congestion_impact_score'] = (
    station_stats['volume_score'] + station_stats['peak_score'] +
    station_stats['road_block_score'] + station_stats['vehicle_density_score']
).round(1)

station_stats = station_stats.sort_values('congestion_impact_score', ascending=False)

print("\n   Congestion Impact Score (top 5):")
for _, row in station_stats.head(5).iterrows():
    print(f"     {row['police_station']}: {row['congestion_impact_score']:.1f}/100  ({row['total_violations']:,} violations)")

# ─────────────────────────────────────────────
# 7. SAVE MODELS + EXPORT JSON
# ─────────────────────────────────────────────
print("\n[7/7] Saving models and exporting predictions...")

with open(os.path.join(MODEL_DIR, "hotspot_classifier.pkl"), "wb") as f:
    pickle.dump(clf, f)
with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(kmeans, f)
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "wb") as f:
    pickle.dump({'station': le_station, 'vehicle': le_vehicle,
                 'violation': le_violation, 'scaler': scaler}, f)

output = {
    "dataset_info": {
        "total_rows_in_source_file": int(before),
        "rows_used_for_training": int(after),
        "rows_retained_pct": round(after/before*100, 2),
        "stations": int(df['police_station'].nunique()),
        "date_range": f"{df['date'].min()} to {df['date'].max()}",
    },
    "model_metrics": {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1_score": round(f1, 4),
        "n_estimators": 150,
        "training_samples": len(X_train), "test_samples": len(X_test),
    },
    "feature_importance": importances.round(4).to_dict(),
    "station_scores": station_stats[['police_station', 'total_violations',
                                       'congestion_impact_score', 'peak_ratio',
                                       'two_wheeler_pct']].round(3).to_dict(orient='records'),
    "clusters": cluster_summary.round(4).to_dict(orient='records'),
}

with open(os.path.join(MODEL_DIR, "predictions.json"), "w") as f:
    json.dump(output, f, indent=2)

df.to_csv(os.path.join(BASE_DIR, "data", "violations_cleaned.csv"), index=False)

print(f"   Saved hotspot_classifier.pkl")
print(f"   Saved kmeans_model.pkl")
print(f"   Saved encoders.pkl")
print(f"   Saved violations_cleaned.csv ({len(df):,} rows)")
print(f"   Exported predictions.json")
print("\n" + "="*70)
print(f"TRAINING COMPLETE — Model trained on {after:,} of {before:,} source rows")
print("="*70)
