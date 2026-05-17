"""
SZZ Labeler — File-Level Bug Label Generation
===============================================
Takes the raw commit-level data and generates file-level labels.

The key insight: if commit C fixes a bug in file F,
then the PREVIOUS version of file F was buggy.

This module creates the ground truth dataset:
  file_path → is_buggy (True/False) for each time window

Usage:
    labeler = SZZLabeler(df_commits)
    df_labeled = labeler.create_file_labels()
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PROCESSED_DIR


class SZZLabeler:
    """
    Converts commit-level mining data into file-level defect labels
    using temporal windowing.

    Strategy:
    - Divide repo history into time windows (e.g., 3-month windows)
    - For each window, label a file as BUGGY if it was modified
      in a bug-fix commit in the NEXT window
    - This way we predict future bugs from current state
    """

    def __init__(
        self,
        df_commits: pd.DataFrame,
        window_days: int = 90,
        prediction_horizon_days: int = 90,
    ):
        """
        Args:
            df_commits: Raw commit DataFrame from GitMiner
            window_days: Size of each observation window (features computed here)
            prediction_horizon_days: How far ahead we predict bugs
        """
        self.df = df_commits.copy()
        self.window_days = window_days
        self.horizon_days = prediction_horizon_days
        self._validate_input()

    # ── Public API ─────────────────────────────────────────────────────────────

    def create_file_labels(self) -> pd.DataFrame:
        """
        Main method. Returns a DataFrame where each row is a
        (file, time_window) pair with a bug label.

        Returns:
            DataFrame with columns:
                file_path, window_start, window_end,
                is_buggy, bug_fix_commits, total_commits
        """
        logger.info("Creating file-level bug labels using SZZ...")

        # Get all unique files
        all_files = self.df["file_path"].unique()
        logger.info(f"Labeling {len(all_files)} unique files")

        # Create time windows
        windows = self._create_time_windows()
        logger.info(f"Created {len(windows)} time windows of {self.window_days} days each")

        records = []
        for window_start, window_end in windows:
            horizon_end = window_end + pd.Timedelta(days=self.horizon_days)

            # Files modified in this observation window
            window_files = self.df[
                (self.df["commit_date"] >= window_start) &
                (self.df["commit_date"] < window_end)
            ]["file_path"].unique()

            if len(window_files) == 0:
                continue

            # Bug-fix commits in the PREDICTION HORIZON (future)
            future_bugfix_files = set(
                self.df[
                    (self.df["commit_date"] >= window_end) &
                    (self.df["commit_date"] < horizon_end) &
                    (self.df["is_bug_fix"] == True)
                ]["file_path"].unique()
            )

            for file_path in window_files:
                # Count commits in this window for this file
                file_window_commits = self.df[
                    (self.df["commit_date"] >= window_start) &
                    (self.df["commit_date"] < window_end) &
                    (self.df["file_path"] == file_path)
                ]

                records.append({
                    "file_path": file_path,
                    "window_start": window_start,
                    "window_end": window_end,
                    "is_buggy": file_path in future_bugfix_files,
                    "commits_in_window": len(file_window_commits),
                    "bug_fix_commits_in_window": int(file_window_commits["is_bug_fix"].sum()),
                    "repo_name": file_window_commits["repo_name"].iloc[0] if len(file_window_commits) > 0 else "unknown",
                })

        df_labeled = pd.DataFrame(records)
        df_labeled = self._post_process(df_labeled)

        logger.success(
            f"Labeled {len(df_labeled)} (file, window) pairs. "
            f"Bug ratio: {df_labeled['is_buggy'].mean():.1%}"
        )

        return df_labeled

    def create_simple_file_labels(self) -> pd.DataFrame:
        """
        Simpler alternative: one label per file across entire history.
        Use this if time windows produce too few samples.

        A file is BUGGY if it appears in any bug-fix commit.
        """
        logger.info("Creating simple (non-windowed) file labels...")

        buggy_files = set(
            self.df[self.df["is_bug_fix"] == True]["file_path"].unique()
        )
        all_files = self.df["file_path"].unique()

        records = []
        for file_path in all_files:
            file_commits = self.df[self.df["file_path"] == file_path]
            records.append({
                "file_path": file_path,
                "is_buggy": file_path in buggy_files,
                "total_commits": len(file_commits),
                "bug_fix_commits": int(file_commits["is_bug_fix"].sum()),
                "first_seen": file_commits["commit_date"].min(),
                "last_seen": file_commits["commit_date"].max(),
                "repo_name": file_commits["repo_name"].iloc[0],
            })

        df_labeled = pd.DataFrame(records)
        bug_ratio = df_labeled["is_buggy"].mean()

        logger.success(
            f"Simple labels: {len(df_labeled)} files, "
            f"bug ratio: {bug_ratio:.1%}"
        )
        return df_labeled

    def get_label_statistics(self, df_labeled: pd.DataFrame) -> dict:
        """Analyze label distribution and quality."""
        stats = {
            "total_samples": len(df_labeled),
            "buggy_samples": int(df_labeled["is_buggy"].sum()),
            "clean_samples": int((~df_labeled["is_buggy"]).sum()),
            "bug_ratio": round(df_labeled["is_buggy"].mean(), 4),
            "unique_files": df_labeled["file_path"].nunique(),
        }

        # Class balance assessment
        if stats["bug_ratio"] < 0.1:
            stats["balance_warning"] = "HIGH IMBALANCE (<10% buggy) — use class weights"
        elif stats["bug_ratio"] > 0.4:
            stats["balance_warning"] = "BALANCED — good dataset"
        else:
            stats["balance_warning"] = "MODERATE IMBALANCE — use class weights"

        logger.info("Label Statistics:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")

        return stats

    # ── Private Methods ────────────────────────────────────────────────────────

    def _create_time_windows(self) -> list[tuple]:
        """Create non-overlapping time windows across the repo history."""
        start_date = self.df["commit_date"].min()
        end_date = self.df["commit_date"].max()

        windows = []
        current = start_date
        while current + pd.Timedelta(days=self.window_days) < end_date:
            window_end = current + pd.Timedelta(days=self.window_days)
            windows.append((current, window_end))
            current = window_end

        return windows

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean labeled data."""
        # Remove windows with too few commits (unreliable)
        df = df[df["commits_in_window"] >= 2].copy()

        # Sort chronologically
        df = df.sort_values(["file_path", "window_start"]).reset_index(drop=True)

        # Save
        output_path = PROCESSED_DIR / f"labeled_{df['repo_name'].iloc[0] if 'repo_name' in df.columns else 'data'}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Labels saved to {output_path}")

        return df

    def _validate_input(self):
        """Validate the input DataFrame."""
        required_cols = ["commit_date", "file_path", "is_bug_fix", "repo_name"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not pd.api.types.is_datetime64_any_dtype(self.df["commit_date"]):
            self.df["commit_date"] = pd.to_datetime(self.df["commit_date"])

        logger.info(f"Input validated: {len(self.df)} commits, "
                   f"date range: {self.df['commit_date'].min().date()} → "
                   f"{self.df['commit_date'].max().date()}")


if __name__ == "__main__":
    # Test with cached mining data
    from pathlib import Path

    raw_path = Path("data/raw/flask_commits.csv")
    if not raw_path.exists():
        logger.error("Run git_miner.py first to generate raw data")
        exit(1)

    df_commits = pd.read_csv(raw_path, parse_dates=["commit_date"])

    labeler = SZZLabeler(df_commits, window_days=90, prediction_horizon_days=90)

    # Try windowed labels first
    df_labeled = labeler.create_file_labels()
    stats = labeler.get_label_statistics(df_labeled)

    # If too few samples, fall back to simple labels
    if stats["total_samples"] < 100:
        logger.warning("Too few windowed samples, using simple labels")
        df_labeled = labeler.create_simple_file_labels()
        stats = labeler.get_label_statistics(df_labeled)

    print("\nSample labeled data:")
    print(df_labeled.head(10))