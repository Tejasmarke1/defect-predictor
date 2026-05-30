import time, traceback, sys
sys.path.insert(0, '.')

print('=== Step 1: Import pipeline modules ===')
try:
    from src.mining.git_miner import GitMiner
    print('✓ GitMiner')
    from src.features.feature_pipeline import FeaturePipeline
    print('✓ FeaturePipeline')
    from src.models.train import DefectXGBoost
    print('✓ DefectXGBoost')
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

print()
print('=== Step 2: Mine flask (180 days) ===')
from datetime import datetime, timezone, timedelta
since = datetime.now(tz=timezone.utc) - timedelta(days=180)
t0 = time.perf_counter()
try:
    miner = GitMiner(repo_url='https://github.com/pallets/flask', since=since)
    commits = miner.mine()
    print(f'✓ Mined {len(commits)} commits in {time.perf_counter()-t0:.1f}s')
    print(f'  columns: {list(commits.columns)}')
    print(f'  sample:\n{commits.head(2).to_string()}')
except Exception as e:
    print(f'✗ Mining failed after {time.perf_counter()-t0:.1f}s: {e}')
    traceback.print_exc()
    sys.exit(1)

print()
print('=== Step 3: Feature pipeline ===')
t0 = time.perf_counter()
try:
    pipeline = FeaturePipeline()
    features = pipeline.run(commits)
    print(f'✓ Features in {time.perf_counter()-t0:.1f}s — shape: {features.shape}')
except Exception as e:
    print(f'✗ Feature pipeline failed after {time.perf_counter()-t0:.1f}s: {e}')
    traceback.print_exc()
    sys.exit(1)

print()
print('=== Step 4: Predict ===')
from src.models.xgboost_model import DefectXGBoost
xgb = DefectXGBoost()
xgb.load('models/xgboost_defect_predictor.json')
meta_cols = {'file_path','is_buggy','repo_name','commit_date','window_start','window_end'}
feat_cols = [c for c in features.columns if c not in meta_cols]
X = features[feat_cols].fillna(0).values
t0 = time.perf_counter()
scores = xgb.predict_proba(X)
print(f'✓ Predicted {len(scores)} files in {time.perf_counter()-t0:.3f}s')

print()
print('ALL STEPS OK — bottleneck identified above by timing')
