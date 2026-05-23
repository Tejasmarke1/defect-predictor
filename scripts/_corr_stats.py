import pandas as pd
import numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'feature_matrix.csv')
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c!='is_buggy']
spearman = df[feature_cols].corr(method='spearman')
triu_idx = np.triu_indices_from(spearman.values, k=1)
all_corrs = spearman.values[triu_idx]
print('total_pairs:', len(all_corrs))
for t in [0.70,0.85,0.90]:
    print(f'pairs_>={t}:', np.sum(np.abs(all_corrs) >= t))
print('max_abs_rho:', float(np.nanmax(np.abs(all_corrs))))
