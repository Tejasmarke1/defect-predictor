"""
PROMISE Dataset Loader
=======================
Downloads and loads pre-labeled defect datasets from the PROMISE repository.
Use this to validate your pipeline quickly while mining runs in background.

Datasets included:
    - KC1: NASA flight software (C++) - 2109 files
    - JM1: NASA real-time system (C) - 10885 files
    - CM1: NASA spacecraft instruments - 498 files
    - PC1: NASA flight system - 1109 files

Usage:
    loader = PromiseLoader()
    df = loader.load("KC1")
"""

import io
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import DATASETS_DIR


class PromiseLoader:
    """
    Loads PROMISE Software Engineering Repository datasets.
    These are the gold-standard benchmark datasets for defect prediction research.
    """

    # Public PROMISE dataset URLs (ARFF format from GitHub mirrors)
    DATASETS = {
        "KC1": {
            "url": "https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/refs/heads/main/kc1.csv",
            "description": "NASA KC1 - 2109 Java modules",
        },
        "JM1": {
            "url": "https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/refs/heads/main/jm1.csv",
            "description": "NASA JM1 - 10885 C modules",
        },
        "CM1": {
            "url": "https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/refs/heads/main/cm1.csv",
            "description": "NASA CM1 - 498 C modules",
        },
        "PC1": {
            "url": "https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/refs/heads/main/pc1.csv",
            "description": "NASA PC1 - 1109 C modules",
        },
    }

    # Standard PROMISE feature names → our names
    FEATURE_MAP = {
        "loc": "lines_of_code",
        "v(g)": "cyclomatic_complexity",
        "ev(g)": "essential_complexity",
        "iv(g)": "design_complexity",
        "n": "halstead_length",
        "v": "halstead_volume",
        "l": "halstead_level",
        "d": "halstead_difficulty",
        "i": "halstead_intelligence",
        "e": "halstead_effort",
        "b": "halstead_bugs",
        "t": "halstead_time",
        "lOCode": "lines_of_code_clean",
        "lOComment": "lines_of_comments",
        "lOBlank": "lines_blank",
        "lOCodeAndComment": "lines_code_and_comment",
        "uniq_Op": "unique_operators",
        "uniq_Opnd": "unique_operands",
        "total_Op": "total_operators",
        "total_Opnd": "total_operands",
        "branchCount": "branch_count",
        "defects": "is_buggy",
    }

    def __init__(self):
        self.cache_dir = DATASETS_DIR

    def load(self, dataset_name: str = "KC1") -> pd.DataFrame:
        """
        Load a PROMISE dataset. Downloads if not cached.

        Args:
            dataset_name: One of KC1, JM1, CM1, PC1

        Returns:
            Cleaned DataFrame ready for feature engineering
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(self.DATASETS.keys())}")

        cache_path = self.cache_dir / f"{dataset_name}.csv"

        if cache_path.exists():
            logger.info(f"Loading cached {dataset_name} from {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info(f"Downloading {dataset_name} dataset...")
            df = self._download(dataset_name)
            df.to_csv(cache_path, index=False)
            logger.success(f"Saved to {cache_path}")

        df = self._clean(df, dataset_name)
        logger.info(f"{dataset_name}: {len(df)} samples, bug ratio: {df['is_buggy'].mean():.1%}")
        return df

    def load_multiple(self, dataset_names: list[str]) -> pd.DataFrame:
        """Load and combine multiple datasets."""
        dfs = []
        for name in dataset_names:
            try:
                df = self.load(name)
                df["dataset_source"] = name
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")

        if not dfs:
            raise ValueError("No datasets loaded successfully")

        combined = pd.concat(dfs, ignore_index=True)
        logger.success(f"Combined: {len(combined)} samples from {len(dfs)} datasets")
        return combined

    def list_datasets(self):
        """Print available datasets."""
        print("\nAvailable PROMISE Datasets:")
        print("-" * 50)
        for name, info in self.DATASETS.items():
            print(f"  {name}: {info['description']}")
        print()

    # ── Private Methods ────────────────────────────────────────────────────────

    def _download(self, dataset_name: str) -> pd.DataFrame:
        """Download dataset from URL."""
        url = self.DATASETS[dataset_name]["url"]

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            return df
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            # Return synthetic data for testing
            logger.warning("Generating synthetic data for testing...")
            return self._generate_synthetic_data(dataset_name)

    def _clean(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Standardize column names and handle data quality issues."""
        # Rename columns
        df = df.rename(columns={
            col: self.FEATURE_MAP.get(col, col.lower().replace(" ", "_"))
            for col in df.columns
        })

        # Ensure is_buggy is boolean
        if "is_buggy" in df.columns:
            if df["is_buggy"].dtype == object:
                df["is_buggy"] = df["is_buggy"].str.lower().map(
                    {"true": True, "false": False, "yes": True, "no": False}
                )
            df["is_buggy"] = df["is_buggy"].astype(bool)

        # Add metadata
        df["file_path"] = [f"{dataset_name}_module_{i}" for i in range(len(df))]
        df["repo_name"] = dataset_name
        df["dataset_source"] = "PROMISE"

        # Drop rows with too many nulls
        df = df.dropna(thresh=len(df.columns) * 0.7)

        # Fill remaining nulls with median
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        return df

    def _generate_synthetic_data(self, dataset_name: str, n_samples: int = 500) -> pd.DataFrame:
        """
        Generate synthetic data with realistic distributions.
        Used as fallback when download fails.
        """
        import numpy as np
        rng = np.random.default_rng(42)

        data = {
            "loc": rng.integers(10, 500, n_samples),
            "v(g)": rng.integers(1, 50, n_samples),
            "ev(g)": rng.integers(1, 30, n_samples),
            "iv(g)": rng.integers(1, 20, n_samples),
            "n": rng.integers(20, 2000, n_samples),
            "v": rng.uniform(100, 5000, n_samples),
            "l": rng.uniform(0, 1, n_samples),
            "d": rng.uniform(1, 100, n_samples),
            "i": rng.uniform(0, 1000, n_samples),
            "e": rng.uniform(100, 100000, n_samples),
            "b": rng.uniform(0, 5, n_samples),
            "t": rng.uniform(0, 1000, n_samples),
            "lOCode": rng.integers(5, 400, n_samples),
            "lOComment": rng.integers(0, 100, n_samples),
            "lOBlank": rng.integers(0, 50, n_samples),
            "lOCodeAndComment": rng.integers(0, 50, n_samples),
            "uniq_Op": rng.integers(5, 30, n_samples),
            "uniq_Opnd": rng.integers(5, 100, n_samples),
            "total_Op": rng.integers(10, 500, n_samples),
            "total_Opnd": rng.integers(10, 500, n_samples),
            "branchCount": rng.integers(0, 30, n_samples),
        }

        df = pd.DataFrame(data)

        # Realistic bug ratio: ~20%
        # More complex files are more likely to be buggy
        complexity_score = (
            df["v(g)"] / df["v(g)"].max() +
            df["loc"] / df["loc"].max()
        ) / 2
        bug_prob = 0.1 + 0.3 * complexity_score
        df["defects"] = rng.random(n_samples) < bug_prob

        logger.warning(f"Using synthetic data ({n_samples} samples)")
        return df


if __name__ == "__main__":
    loader = PromiseLoader()
    loader.list_datasets()

    # Load KC1 as primary test dataset
    df = loader.load("KC1")
    print("\nDataset Info:")
    print(df.describe())
    print(f"\nBug ratio: {df['is_buggy'].mean():.1%}")
    print(f"Columns: {list(df.columns)}")