import pandas as pd
import numpy as np
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent          
PROJECT_ROOT = SCRIPT_DIR.parent           
os.chdir(PROJECT_ROOT)  

#  CONFIG 
VITAL_COLS = ['heart_rate', 'temperature', 'oxygen_saturation', 'respiratory_rate', 'blood_pressure']
LAB_COLS = ['white_cell_count', 'crp', 'lactate', 'creatinine', 'platelet_count']
LOOKBACK_HOUR = 6
LAB_LOOKBACK_HOURS = 24

#  LOAD DATA 
def load_data():
    patients = pd.read_csv('data/processed/patients_clean.csv', parse_dates=['registration_date'])
    vitals = pd.read_csv('data/processed/vital_signs_clean.csv', parse_dates=['timestamp'])
    labs = pd.read_csv('data/processed/laboratory_results_clean.csv', parse_dates=['timestamp'])
    outcomes = pd.read_csv('data/processed/sepsis_outcomes_clean.csv', parse_dates=['diagnosis_time'])
    return patients, vitals, labs, outcomes

# VITAL FEATURES 
def get_vital_features(pid, cutoff, vitals):
    window = vitals[(vitals['patient_id'] == pid) & 
                    (vitals['timestamp'] <= cutoff) & 
                    (vitals['timestamp'] >= cutoff - pd.Timedelta(hours=LOOKBACK_HOUR))]
    if window.empty:
        window = vitals[(vitals['patient_id'] == pid) & (vitals['timestamp'] <= cutoff)].tail(1)
    
    feats = {}
    for col in VITAL_COLS:
        vals = window[col]
        feats[f'{col}_mean'] = vals.mean()
        feats[f'{col}_min'] = vals.min()
        feats[f'{col}_max'] = vals.max()
        feats[f'{col}_std'] = vals.std() if len(vals) > 1 else 0.0
        feats[f'{col}_last'] = vals.iloc[-1] if len(vals) > 0 else np.nan
        
        if len(window) > 1:
            hours = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds() / 3600
            feats[f'{col}_rate_per_hr'] = (vals.iloc[-1] - vals.iloc[0]) / hours if hours > 0 else 0.0
        else:
            feats[f'{col}_rate_per_hr'] = 0.0
    return feats

# LAB FEATURES 
def get_lab_features(pid, cutoff, labs):
    window = labs[(labs['patient_id'] == pid) & 
                  (labs['timestamp'] <= cutoff) & 
                  (labs['timestamp'] >= cutoff - pd.Timedelta(hours=LAB_LOOKBACK_HOURS))]
    if window.empty:
        window = labs[(labs['patient_id'] == pid) & (labs['timestamp'] <= cutoff)].tail(1)
    
    feats = {}
    for col in LAB_COLS:
        vals = window[col]
        feats[f'{col}_mean'] = vals.mean()
        feats[f'{col}_last'] = vals.iloc[-1] if len(vals) > 0 else np.nan
    return feats

# STATIC FEATURES
def get_static_features(patients):
    static = patients[['patient_id', 'age', 'gender']].copy()
    static['comorbidity_count'] = patients['medical_conditions'].apply(
        lambda x: 0 if pd.isna(x) or x == 'None reported' else len(x.split(','))
    )
    static['gender'] = static['gender'].astype(str)
    static = pd.get_dummies(static, columns=['gender'], prefix='gender', drop_first=True)
    return static

# MAIN PIPELINE 
def build_features():
    # Load
    patients, vitals, labs, outcomes = load_data()
    
    # Create prediction times
    outcomes['prediction_time'] = outcomes['diagnosis_time'] - pd.Timedelta(hours=9)
    
    # Vital features
    vital_rows = [{'patient_id': pid, **get_vital_features(pid, cutoff, vitals)} 
                  for pid, cutoff in zip(outcomes['patient_id'], outcomes['prediction_time'])]
    vital_df = pd.DataFrame(vital_rows)
    
    # Lab features
    lab_rows = [{'patient_id': pid, **get_lab_features(pid, cutoff, labs)} 
                for pid, cutoff in zip(outcomes['patient_id'], outcomes['prediction_time'])]
    lab_df = pd.DataFrame(lab_rows)
    
    # Static features
    static_df = get_static_features(patients)
    
    # Merge all
    final_df = (static_df
                .merge(vital_df, on='patient_id')
                .merge(lab_df, on='patient_id')
                .merge(outcomes[['patient_id', 'sepsis_event']], on='patient_id'))
    
    # Fill missing
    feature_cols = [c for c in final_df.columns if c not in ('patient_id', 'sepsis_event')]
    numeric_cols = final_df[feature_cols].select_dtypes(include='number').columns
    final_df[numeric_cols] = final_df[numeric_cols].fillna(final_df[numeric_cols].median())
    
    # Convert target
    final_df['sepsis_event'] = final_df['sepsis_event'].astype(int)
    
    # Save
    final_df.to_csv('data/processed/sepsis_features.csv', index=False)
    print(f"Saved {final_df.shape[0]} rows, {final_df.shape[1]} columns")
    
    return final_df

# ===== RUN =====
if __name__ == "__main__":
    build_features()