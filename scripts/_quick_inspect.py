import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_MATRIX_PATH = PROJECT_ROOT / 'data' / 'processed' / 'feature_matrix.csv'

print('FEATURE_MATRIX_PATH ->', FEATURE_MATRIX_PATH)

df = pd.read_csv(FEATURE_MATRIX_PATH)
print('shape:', df.shape)
print('columns:', len(df.columns))

label_counts = df['is_buggy'].value_counts().to_dict()
print('label_counts:', label_counts)
print(f"buggy_rate: {100 * label_counts.get(1,0)/len(df):.2f}%")

numeric_df = df.select_dtypes(include=[np.number])
missing = numeric_df.isnull().mean().sort_values(ascending=False)
missing_nonzero = missing[missing > 0]
print('n_missing_numeric_cols:', len(missing_nonzero))
if not missing_nonzero.empty:
    print(missing_nonzero.to_dict())

feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'is_buggy']
X = df[feature_cols].fillna(0)
y = df['is_buggy'].astype(int)
mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
import pandas as pd
mi_df = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
print('top3_mi:', mi_df.head(3).index.tolist())
print('top15_mi:', mi_df.head(15).to_dict())
