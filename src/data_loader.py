"""Data Ingestion Module for Buy or Wait financial decision agent."""

import logging
from pathlib import Path
from typing import Dict, Union
import pandas as pd

logger = logging.getLogger(__name__)

DATASET_FILES = [
    "requests.csv",
    "financial_profiles.csv",
    "financial_events.csv",
    "exchange_rates.csv",
    "request_payment_options.csv",
    "messages.csv",
    "images.csv",
]


def load_all_datasets(data_dir: Union[str, Path] = "dataset") -> Dict[str, pd.DataFrame]:
    """
    Load all challenge CSV datasets into a dictionary of DataFrames.

    Args:
        data_dir: Path to directory containing the dataset CSV files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping dataset stem name (e.g. 'requests')
                                 to its loaded pandas DataFrame.
    """
    base_path = Path(data_dir)
    datasets: Dict[str, pd.DataFrame] = {}

    for filename in DATASET_FILES:
        key = Path(filename).stem
        file_path = base_path / filename

        if not file_path.exists():
            logger.warning(f"Warning: File {file_path} does not exist.")
            print(f"Warning: File missing: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            datasets[key] = df
            logger.info(f"Loaded {filename} with shape {df.shape}")
        except FileNotFoundError:
            logger.error(f"Error: File not found: {file_path}")
            print(f"Error: File not found: {file_path}")
        except pd.errors.EmptyDataError:
            logger.warning(f"Warning: File {file_path} is empty.")
            print(f"Warning: File is empty: {file_path}")
            datasets[key] = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            print(f"Error loading {file_path}: {e}")

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_all_datasets()
    print("\nSummary of loaded datasets:")
    for name, df in data.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
