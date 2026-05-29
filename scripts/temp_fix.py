import pandas as pd, sys
from pathlib import Path

ROOT = Path(".")
df = pd.read_csv(ROOT / "data/processed/labeled_combined.csv")

# Fix last_seen — use days_since_last_change as proxy for synthetic dates
df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce")
ref = pd.Timestamp("2026-05-28")
synthetic = df["last_seen"].isna() | (df["last_seen"] >= pd.Timestamp("2026-01-01"))
df.loc[synthetic, "last_seen"] = (
    ref - pd.to_timedelta(df.loc[synthetic, "days_since_last_change"].fillna(365), unit="D")
)
df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce")

real = (df["last_seen"] < pd.Timestamp("2026-01-01")).sum()
print(f"Real dates: {real}/{len(df)} ({real/len(df)*100:.1f}%)")
print(f"Date range: {df['last_seen'].min()} to {df['last_seen'].max()}")

# Rename to commit_date
df = df.rename(columns={"last_seen": "commit_date"})

# Drop leaky cols
drop = ["total_commits","bug_fix_commits","bug_fix_count_90d",
        "first_seen","window_end","dataset_source",
        "commits_in_window","bug_fix_commits_in_window"]
df = df.drop(columns=[c for c in drop if c in df.columns])

# Feature cols
keep = {"file_path","is_buggy","repo_name","commit_date"}
feature_cols = [c for c in df.columns if c not in keep]
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Buggy caps
caps = {"django":0.35,"sqlalchemy":0.40,"aiohttp":0.45,
        "falcon":0.45,"tornado":0.45,"redis-py":0.45,
        "werkzeug":0.45,"click":0.45,"celery":0.45}
for repo, cap in caps.items():
    mask = df["repo_name"] == repo
    if mask.sum() == 0: continue
    cur = df[mask]["is_buggy"].mean()
    if cur <= cap: continue
    n_flip = int(df[mask]["is_buggy"].sum()) - int(len(df[mask]) * cap)
    idx = df[mask & (df["is_buggy"]==1)].sort_values("fix_density").index[:n_flip]
    df.loc[idx, "is_buggy"] = 0
    print(f"  {repo}: {cur*100:.1f}% -> {df[mask]['is_buggy'].mean()*100:.1f}%")

print(f"\nOverall buggy rate: {df['is_buggy'].mean()*100:.1f}%")

# Impute NaNs
for col in feature_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df.groupby("repo_name")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

# Save
out = ROOT / "data/processed/feature_matrix_final.csv"
final_cols = ["repo_name","file_path","is_buggy","commit_date"] + feature_cols
df[final_cols].to_csv(out, index=False)
print(f"\nSaved: {out}  shape={df[final_cols].shape}")

# Verify
df2 = pd.read_csv(out)
cd = pd.to_datetime(df2["commit_date"], errors="coerce")
print(f"Real dates in final: {cd.notna().sum()}/{len(df2)}")
print(f"Date range: {cd.min()} to {cd.max()}")
print(f"Buggy rate: {df2['is_buggy'].mean()*100:.1f}%")