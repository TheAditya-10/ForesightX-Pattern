"""
Data Preprocessing Module for ForesightX
========================================
Production-ready module for cleaning and preparing stock market data.

Features:
- Load raw data from data ingestion stage
- Handle missing values (forward/backward fill)
- Remove duplicates
- Sort by date chronologically
- Handle outliers (detection and optional removal)
- Train/test split with chronological order
- Comprehensive logging and exception handling

Author: Aditya Pratap Singh Tomar
Date: 2025-12-17
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Third-party imports
import pandas as pd
import numpy as np
import yaml

# Local imports
from src.services.logger import get_logger, log_function_call


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


class DataPreprocessor:
    """
    Handles data preprocessing and cleaning for stock market data.
    
    This class provides robust preprocessing capabilities with:
    - Missing value handling (forward/backward fill)
    - Duplicate removal
    - Chronological sorting
    - Outlier detection and handling
    - Train/test splitting
    - Data validation
    - Structured logging
    
    Attributes:
        config (dict): Configuration parameters from params.yaml
        logger: Configured logger instance
        dagshub_service: DagsHub service for cloud storage (optional)
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config_path: Path to configuration YAML file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            PreprocessingError: If initialization fails
        """
        self.logger = get_logger("DataPreprocessing")
        self.logger.info("=" * 70)
        self.logger.info("Initializing Data Preprocessing Pipeline")
        self.logger.info("=" * 70)
        
        try:
            # Load configuration
            self.config = self._load_config(config_path)
            self.logger.info(f"Configuration loaded from: {config_path}")
            
            # Setup directories
            self._setup_directories()
            
            self.logger.info("Preprocessing pipeline initialized successfully")
            self.logger.info("Note: Files saved locally. Use 'dvc push' to upload to DagsHub storage")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize preprocessing: {e}")
            raise PreprocessingError(f"Initialization failed: {e}")
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        try:
            directories = [
                Path("data/interim"),
                Path("data/processed"),
                Path("logs"),
                Path("metadata")
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Directory verified: {directory}")
            
        except Exception as e:
            self.logger.error(f"Error setting up directories: {e}")
            raise PreprocessingError(f"Directory setup failed: {e}")
    
    @log_function_call
    def load_data(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw data from the data ingestion stage.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with raw data
            
        Raises:
            PreprocessingError: If data loading fails
        """
        try:
            symbol = symbol or self.config['data_ingestion']['stock_symbol']
            data_path = Path(f"data/raw/stock_data_raw_{symbol}.csv")
            
            self.logger.info(f"Loading raw data for {symbol}")
            self.logger.info(f"  Path: {data_path}")
            
            if not data_path.exists():
                raise PreprocessingError(f"Raw data file not found: {data_path}")
            
            df = pd.read_csv(data_path)
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            self.logger.info(f"✓ Data loaded successfully")
            self.logger.info(f"  Shape: {df.shape}")
            self.logger.info(f"  Columns: {list(df.columns)}")
            self.logger.info(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise PreprocessingError(f"Data loading failed: {e}")
    
    @log_function_call
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Strategy:
        - For price columns: Forward fill then backward fill
        - Drop rows with remaining NaN in critical columns (Close)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
            
        Raises:
            PreprocessingError: If handling fails
        """
        try:
            self.logger.info("Handling missing values...")
            
            initial_missing = df.isnull().sum().sum()
            self.logger.info(f"  Initial missing values: {initial_missing}")
            
            if initial_missing == 0:
                self.logger.info("  ✓ No missing values found!")
                return df
            
            # Get preprocessing config
            handle_method = self.config['preprocessing'].get('handle_missing', 'ffill')
            
            # Define price columns
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Handle missing values for each price column
            for col in price_columns:
                if col in df.columns:
                    before = df[col].isnull().sum()
                    if before > 0:
                        # Forward fill
                        df[col] = df[col].fillna(method='ffill')
                        # Backward fill for any remaining
                        df[col] = df[col].fillna(method='bfill')
                        
                        after = df[col].isnull().sum()
                        filled = before - after
                        if filled > 0:
                            self.logger.info(f"  ✓ Filled {filled} missing values in {col}")
            
            # Drop rows with remaining NaN in critical columns
            critical_cols = ['Close']
            rows_before = len(df)
            df = df.dropna(subset=critical_cols)
            rows_dropped = rows_before - len(df)
            
            if rows_dropped > 0:
                self.logger.warning(f"  Dropped {rows_dropped} rows with missing critical values")
            
            final_missing = df.isnull().sum().sum()
            self.logger.info(f"  Final missing values: {final_missing}")
            self.logger.info("  ✓ Missing values handled")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {e}")
            raise PreprocessingError(f"Missing value handling failed: {e}")
    
    @log_function_call
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows based on Date column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
            
        Raises:
            PreprocessingError: If removal fails
        """
        try:
            self.logger.info("Checking for duplicates...")
            
            initial_rows = len(df)
            duplicates = df.duplicated(subset=['Date']).sum()
            
            if duplicates > 0:
                self.logger.warning(f"  Found {duplicates} duplicate dates")
                df = df.drop_duplicates(subset=['Date'], keep='first')
                removed = initial_rows - len(df)
                self.logger.info(f"  ✓ Removed {removed} duplicate rows")
            else:
                self.logger.info("  ✓ No duplicates found!")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error removing duplicates: {e}")
            raise PreprocessingError(f"Duplicate removal failed: {e}")
    
    @log_function_call
    def sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort DataFrame by Date column in chronological order.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Sorted DataFrame
            
        Raises:
            PreprocessingError: If sorting fails
        """
        try:
            self.logger.info("Sorting data chronologically...")
            
            df = df.sort_values('Date')
            df = df.reset_index(drop=True)
            
            self.logger.info("  ✓ Data sorted by date")
            self.logger.info(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error sorting data: {e}")
            raise PreprocessingError(f"Sorting failed: {e}")
    
    @log_function_call
    def handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and optionally handle outliers in price data.
        
        Uses IQR method (3*IQR threshold for extreme outliers only).
        Note: Price spikes can be real market events, so we detect but may not remove.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (processed DataFrame, outlier statistics)
            
        Raises:
            PreprocessingError: If handling fails
        """
        try:
            self.logger.info("Checking for extreme outliers...")
            
            remove_outliers = self.config['preprocessing'].get('remove_outliers', False)
            outlier_method = self.config['preprocessing'].get('outlier_method', 'iqr')
            outlier_threshold = self.config['preprocessing'].get('outlier_threshold', 3.0)
            
            # Calculate IQR for Close price
            Q1 = df['Close'].quantile(0.25)
            Q3 = df['Close'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            
            # Find outliers
            outlier_mask = (df['Close'] < lower_bound) | (df['Close'] > upper_bound)
            extreme_outliers = outlier_mask.sum()
            
            outlier_stats = {
                'method': outlier_method,
                'threshold': outlier_threshold,
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outliers_found': int(extreme_outliers),
                'outliers_removed': 0
            }
            
            if extreme_outliers > 0:
                outlier_dates = df[outlier_mask]['Date'].tolist()
                self.logger.warning(
                    f"  Found {extreme_outliers} extreme price outliers "
                    f"({extreme_outliers/len(df)*100:.2f}%)"
                )
                self.logger.info(f"  Dates with extreme values: {[d.date() for d in outlier_dates[:5]]}")
                
                if remove_outliers:
                    initial_rows = len(df)
                    df = df[~outlier_mask]
                    removed = initial_rows - len(df)
                    outlier_stats['outliers_removed'] = removed
                    self.logger.warning(f"  ⚠ Removed {removed} outlier rows")
                else:
                    self.logger.info("  ℹ Not removing outliers (could be real market events)")
            else:
                self.logger.info("  ✓ No extreme outliers detected")
            
            # Check for invalid prices (≤0)
            invalid_prices = (df['Close'] <= 0).sum()
            if invalid_prices > 0:
                self.logger.warning(f"  Found {invalid_prices} invalid prices (≤0)")
                initial_rows = len(df)
                df = df[df['Close'] > 0]
                removed = initial_rows - len(df)
                self.logger.info(f"  ✓ Removed {removed} rows with invalid prices")
                outlier_stats['invalid_prices_removed'] = removed
            
            return df, outlier_stats
            
        except Exception as e:
            self.logger.error(f"Error handling outliers: {e}")
            raise PreprocessingError(f"Outlier handling failed: {e}")
    
    @log_function_call
    def train_test_split(
        self, 
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split data into train and test sets chronologically.
        
        Args:
            df: Input DataFrame
            symbol: Stock symbol for file naming
            
        Returns:
            Tuple of (train_df, test_df, split_info)
            
        Raises:
            PreprocessingError: If splitting fails
        """
        try:
            self.logger.info("Splitting data into train and test sets...")
            
            test_size = self.config['training'].get('test_size', 0.2)
            
            # Calculate split point
            split_idx = int(len(df) * (1 - test_size))
            
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            split_info = {
                'total_samples': len(df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'test_size': test_size,
                'train_date_range': {
                    'start': train_df['Date'].min().isoformat(),
                    'end': train_df['Date'].max().isoformat()
                },
                'test_date_range': {
                    'start': test_df['Date'].min().isoformat(),
                    'end': test_df['Date'].max().isoformat()
                }
            }
            
            self.logger.info(f"  ✓ Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
            self.logger.info(f"    Date range: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
            self.logger.info(f"  ✓ Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
            self.logger.info(f"    Date range: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
            
            return train_df, test_df, split_info
            
        except Exception as e:
            self.logger.error(f"Error in train/test split: {e}")
            raise PreprocessingError(f"Train/test split failed: {e}")
    
    @log_function_call
    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        symbol: str,
        preprocessing_stats: Dict[str, Any],
        save_to_s3: bool = False
    ) -> Dict[str, str]:
        """
        Save processed train and test data to disk.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            symbol: Stock symbol
            preprocessing_stats: Statistics from preprocessing
            save_to_s3: Whether to upload to S3
            
        Returns:
            Dictionary with file paths
            
        Raises:
            PreprocessingError: If saving fails
        """
        try:
            self.logger.info("Saving processed data...")
            
            # Generate file paths
            train_filename = f"train_data_{symbol}.csv"
            test_filename = f"test_data_{symbol}.csv"
            stats_filename = f"preprocessing_stats_{symbol}.json"
            
            train_path = Path("data/processed") / train_filename
            test_path = Path("data/processed") / test_filename
            stats_path = Path("metadata") / stats_filename
            
            # Save train data
            train_df.to_csv(train_path, index=False)
            train_size_kb = train_path.stat().st_size / 1024
            self.logger.info(f"  ✓ Train data saved to: {train_path}")
            self.logger.info(f"    Size: {train_size_kb:.2f} KB, Records: {len(train_df)}")
            
            # Save test data
            test_df.to_csv(test_path, index=False)
            test_size_kb = test_path.stat().st_size / 1024
            self.logger.info(f"  ✓ Test data saved to: {test_path}")
            self.logger.info(f"    Size: {test_size_kb:.2f} KB, Records: {len(test_df)}")
            
            # Save preprocessing statistics
            preprocessing_stats['saved_timestamp'] = datetime.now().isoformat()
            preprocessing_stats['files'] = {
                'train': str(train_path),
                'test': str(test_path),
                'train_size_kb': round(train_size_kb, 2),
                'test_size_kb': round(test_size_kb, 2)
            }
            
            with open(stats_path, 'w') as f:
                json.dump(preprocessing_stats, f, indent=4)
            self.logger.info(f"  ✓ Statistics saved to: {stats_path}")
            
            result = {
                'train_path': str(train_path),
                'test_path': str(test_path),
                'stats_path': str(stats_path)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
            raise PreprocessingError(f"Save failed: {e}")
    
    @log_function_call
    def run_preprocessing_pipeline(
        self,
        symbol: Optional[str] = None,
        save_to_s3: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        This orchestrates:
        1. Load raw data
        2. Handle missing values
        3. Remove duplicates
        4. Sort by date
        5. Handle outliers
        6. Train/test split
        7. Save processed data
        
        Args:
            symbol: Stock ticker symbol
            save_to_s3: Whether to upload to S3
            
        Returns:
            Dictionary with pipeline results
            
        Raises:
            PreprocessingError: If pipeline fails
        """
        import time
        pipeline_start = time.time()
        
        try:
            self.logger.info("=" * 70)
            self.logger.info("STARTING PREPROCESSING PIPELINE")
            self.logger.info("=" * 70)
            
            symbol = symbol or self.config['data_ingestion']['stock_symbol']
            
            # Step 1: Load data
            df = self.load_data(symbol)
            initial_shape = df.shape
            
            # Step 2: Handle missing values
            df = self.handle_missing_values(df)
            
            # Step 3: Remove duplicates
            df = self.remove_duplicates(df)
            
            # Step 4: Sort by date
            df = self.sort_by_date(df)
            
            # Step 5: Handle outliers
            df, outlier_stats = self.handle_outliers(df)
            
            final_shape = df.shape
            
            # Step 6: Train/test split
            train_df, test_df, split_info = self.train_test_split(df, symbol)
            
            # Prepare preprocessing statistics
            preprocessing_stats = {
                'symbol': symbol,
                'pipeline_timestamp': datetime.now().isoformat(),
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'rows_removed': initial_shape[0] - final_shape[0],
                'outlier_stats': outlier_stats,
                'split_info': split_info,
                'config': {
                    'handle_missing': self.config['preprocessing']['handle_missing'],
                    'remove_outliers': self.config['preprocessing']['remove_outliers'],
                    'test_size': self.config['training']['test_size']
                }
            }
            
            # Step 7: Save processed data
            file_paths = self.save_processed_data(
                train_df, test_df, symbol, preprocessing_stats, save_to_s3
            )
            
            # Calculate pipeline metrics
            pipeline_duration = time.time() - pipeline_start
            
            result = {
                'status': 'success',
                'symbol': symbol,
                'initial_records': initial_shape[0],
                'final_records': final_shape[0],
                'train_records': len(train_df),
                'test_records': len(test_df),
                'file_paths': file_paths,
                'preprocessing_stats': preprocessing_stats,
                'pipeline_duration_seconds': round(pipeline_duration, 2)
            }
            
            self.logger.info("=" * 70)
            self.logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"  Duration: {pipeline_duration:.2f} seconds")
            self.logger.info(f"  Records processed: {initial_shape[0]} → {final_shape[0]}")
            self.logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}")
            self.logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise PreprocessingError(f"Pipeline execution failed: {e}")


def main():
    """Main function to run preprocessing."""
    try:
        # Initialize preprocessing pipeline
        preprocessor = DataPreprocessor()
        
        # Run pipeline
        result = preprocessor.run_preprocessing_pipeline()
        
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        print(f"Status: {result['status']}")
        print(f"Symbol: {result['symbol']}")
        print(f"Records: {result['initial_records']} → {result['final_records']}")
        print(f"Train: {result['train_records']}, Test: {result['test_records']}")
        print(f"Duration: {result['pipeline_duration_seconds']}s")
        print(f"Train data: {result['file_paths']['train_path']}")
        print(f"Test data: {result['file_paths']['test_path']}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
