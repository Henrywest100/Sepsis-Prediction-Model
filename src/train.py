from pathlib import Path
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(
    roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert-0