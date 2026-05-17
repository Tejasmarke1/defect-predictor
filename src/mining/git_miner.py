"""
Git Mining Pipeline — Day 1 Core
=================================
Uses PyDriller to mine git history and extract:
- Commit metadata (author, date, message, files changed)
- File-level change history
- Bug-fix commit labels via SZZ algorithm

Usage:
    miner = GitMiner("https://github.com/pallets/flask")
    df = miner.mine()
    df.to_csv("data/raw/flask_commits.csv", index=False)
"""

import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from pydriller import Repository
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import MINING, RAW_DIR


class GitMiner:
    """
    Mines a Git repository and extracts file-level commit history
    with bug-fix labels using the SZZ algorithm (keyword-based).
    """

    def __init__(
        self,
        repo_url: str,
        repo_name: Optional[str] = None,
        since: Optional[datetime] = None,
        to: Optional[datetime] = None,
    ):
        self.repo_url = repo_url
        self.repo_name = repo_name or self._extract_repo_name(repo_url)
        self.since = since
        self.to = to
        self.output_path = RAW_DIR / f"{self.repo_name}_commits.csv"

    # ── Public API ─────────────────────────────────────────────────────────────

    def mine(self, force_remine: bool = False) -> pd.DataFrame:
        """
        Main entry point. Mines the repository and returns a DataFrame.
        Caches results — won't re-mine unless force_remine=True.
        """
        if self.output_path.exists() and not force_remine:
            logger.info(f"Loading cached data from {self.output_path}")
            return pd.read_csv(self.output_path, parse_dates=["commit_date"])

        logger.info(f"Mining repository: {self.repo_url}")
        records = self._extract_commit_records()

        if not records:
            raise ValueError(f"No Python commits found in {self.repo_url}")

        df = pd.DataFrame(records)
        df = self._post_process(df)

        df.to_csv(self.output_path, index=False)
        logger.success(f"Mined {len(df)} file-commit records → {self.output_path}")
        return df

    def get_summary(self, df: pd.DataFrame) -> dict:
        """Print a summary of what was mined."""
        summary = {
            "total_commits": df["commit_hash"].nunique(),
            "total_files": df["file_path"].nunique(),
            "bug_fix_commits": df[df["is_bug_fix"]]["commit_hash"].nunique(),
            "buggy_files": df[df["is_bug_fix"]]["file_path"].nunique(),
            "date_range": f"{df['commit_date'].min().date()} → {df['commit_date'].max().date()}",
            "bug_ratio": round(df[df["is_bug_fix"]]["file_path"].nunique() / df["file_path"].nunique(), 3),
        }

        logger.info("=" * 50)
        logger.info(f"Repository: {self.repo_name}")
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 50)

        return summary

    # ── Private Methods ────────────────────────────────────────────────────────

    def _extract_commit_records(self) -> list[dict]:
        """
        Iterate over all commits and extract file-level records.
        Each record = one file modified in one commit.

        Robustness notes:
        - Merge commits can cause git diff-tree exit code 128 — skipped gracefully
        - Commits with None message are handled safely
        - Individual bad commits never abort the whole mining run
        """
        records = []
        skipped = 0
        kwargs = {"url": self.repo_url}
        if self.since:
            kwargs["since"] = self.since
        if self.to:
            kwargs["to"] = self.to

        try:
            repo = Repository(**kwargs)
            commits = list(repo.traverse_commits())
            logger.info(f"Found {len(commits)} total commits to process")

            for commit in tqdm(commits, desc="Mining commits"):
                try:
                    # commit.msg can be None on malformed/empty commits
                    msg = commit.msg or ""
                    is_fix = self._is_bug_fix_commit(msg)

                    # accessing modified_files triggers git diff — can fail on
                    # merge commits or shallow clones (exit code 128)
                    modified = commit.modified_files

                    for modified_file in modified:
                        # Skip non-Python files
                        if not self._is_valid_file(modified_file.filename):
                            continue

                        # Skip files that are too large
                        if modified_file.source_code and len(
                            modified_file.source_code.encode()
                        ) > MINING["max_file_size"]:
                            continue

                        record = {
                            "commit_hash": commit.hash,
                            "commit_date": commit.author_date,
                            "commit_message": msg[:200],
                            "author_email": commit.author.email if commit.author else "unknown",
                            "author_name": commit.author.name if commit.author else "unknown",
                            "is_bug_fix": is_fix,
                            "file_path": modified_file.new_path or modified_file.old_path,
                            "filename": modified_file.filename,
                            "lines_added": modified_file.added_lines or 0,
                            "lines_deleted": modified_file.deleted_lines or 0,
                            "lines_changed": (modified_file.added_lines or 0) + (modified_file.deleted_lines or 0),
                            "complexity": modified_file.complexity or 0,
                            "nloc": modified_file.nloc or 0,
                            "source_code": modified_file.source_code or "",
                        }
                        records.append(record)

                except Exception as commit_err:
                    # Log at WARNING not ERROR — these are expected on merge commits
                    logger.warning(f"Skipping commit {commit.hash}: {commit_err}")
                    skipped += 1
                    continue

        except Exception as e:
            logger.error(f"Mining failed fatally: {e}")
            raise

        logger.info(f"Mining complete — {len(records)} records extracted, {skipped} commits skipped")
        return records

    def _is_bug_fix_commit(self, commit_message: str) -> bool:
        """
        SZZ Algorithm — keyword-based bug fix detection.
        Also checks for issue references like 'fixes #123'.
        """
        msg = commit_message.lower()

        # Direct keyword match
        for keyword in MINING["bug_keywords"]:
            if keyword in msg:
                return True

        # Issue reference patterns: "fixes #123", "closes #456"
        issue_patterns = [
            r"fix(?:es|ed)?\s+#\d+",
            r"close(?:s|d)?\s+#\d+",
            r"resolve(?:s|d)?\s+#\d+",
            r"bug\s+#\d+",
        ]
        for pattern in issue_patterns:
            if re.search(pattern, msg):
                return True

        return False

    def _is_valid_file(self, filename: str) -> bool:
        """Check if file should be included in analysis."""
        # Must be Python
        if not any(filename.endswith(ext) for ext in MINING["file_extensions"]):
            return False

        # Skip test files
        filename_lower = filename.lower()
        if any(ignore in filename_lower for ignore in MINING["ignore_paths"]):
            return False

        return True

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enrich the raw mined data."""
        # Parse datetime — utc=True handles any tz-aware strings
        df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")
        # Convert to tz-naive (simpler for all downstream pandas operations)
        df["commit_date"] = df["commit_date"].dt.tz_convert(None)
        # Drop any rows where date parsing failed entirely
        df = df.dropna(subset=["commit_date"])

        # Sort by date
        df = df.sort_values("commit_date").reset_index(drop=True)

        # Add repo name
        df["repo_name"] = self.repo_name

        # Normalize file paths
        df["file_path"] = df["file_path"].fillna("unknown").str.replace("\\", "/")

        # Drop duplicates (same file, same commit)
        df = df.drop_duplicates(subset=["commit_hash", "file_path"])

        logger.info(f"After cleaning: {len(df)} records, {df['file_path'].nunique()} unique files")
        return df

    @staticmethod
    def _extract_repo_name(url: str) -> str:
        """Extract 'flask' from 'https://github.com/pallets/flask'."""
        return url.rstrip("/").split("/")[-1].replace(".git", "")


class MultiRepoMiner:
    """
    Mine multiple repositories and combine into one dataset.
    Useful for getting more training data.
    """

    RECOMMENDED_REPOS = [
        "https://github.com/pallets/flask",
        "https://github.com/psf/requests",
        "https://github.com/celery/celery",
        "https://github.com/scrapy/scrapy",
    ]

    def __init__(self, repo_urls: list[str]):
        self.repo_urls = repo_urls

    def mine_all(self) -> pd.DataFrame:
        """Mine all repos and return combined DataFrame."""
        dfs = []
        for url in self.repo_urls:
            try:
                miner = GitMiner(url)
                df = miner.mine()
                dfs.append(df)
                logger.success(f"✓ Mined {url}")
            except Exception as e:
                logger.error(f"✗ Failed to mine {url}: {e}")
                continue

        if not dfs:
            raise ValueError("No repos were successfully mined")

        combined = pd.concat(dfs, ignore_index=True)
        output_path = RAW_DIR / "combined_commits.csv"
        combined.to_csv(output_path, index=False)
        logger.success(f"Combined dataset: {len(combined)} records from {len(dfs)} repos")
        return combined


if __name__ == "__main__":
    # Quick test — mine Flask (small, fast)
    logger.info("Starting Day 1 mining test with Flask repository...")

    miner = GitMiner(
        repo_url="https://github.com/pallets/flask",
        since=datetime(2020, 1, 1),  # Last 4 years is plenty
    )

    df = miner.mine()
    summary = miner.get_summary(df)

    print("\nSample records:")
    print(df[["commit_date", "file_path", "is_bug_fix", "lines_changed"]].head(10))
    print(f"\nBug fix ratio: {summary['bug_ratio']:.1%}")
    print(f"\nData saved to: {miner.output_path}")