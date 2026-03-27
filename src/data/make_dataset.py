"""
Data Ingestion Module for ForesightX
====================================
Production-ready module for fetching and saving historical stock market data.

Features:
- Yahoo Finance data source (via yfinance)
- Comprehensive exception handling
- Data validation and quality checks
- Structured logging with rotation
- Retry mechanism for API failures
- Metadata collection and storage

Author: Aditya Pratap Singh Tomar
Date: 2025-12-17
"""

import os
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Third-party imports
import pandas as pd
import numpy as np
import yfinance as yf
import yaml

# Local imports
from src.services.logger import get_logger, log_function_call


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class StockDataIngestion:
    """
    Handles stock market data ingestion from Yahoo Finance.
    
    This class provides robust data fetching capabilities with:
    - Yahoo Finance integration via yfinance library
    - Automatic retry logic for API failures
    - Data validation and quality checks
    - Structured logging
    - Local and cloud storage options
    
    Attributes:
        config (dict): Configuration parameters from params.yaml
        logger: Configured logger instance
        dagshub_service: DagsHubService instance for cloud storage
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            config_path: Path to configuration YAML file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            DataIngestionError: If initialization fails
        """
        self.logger = get_logger("DataIngestion")
        self.logger.info("=" * 70)
        self.logger.info("Initializing Stock Data Ingestion Pipeline")
        self.logger.info("=" * 70)
        
        try:
            # Load configuration
            self.config = self._load_config(config_path)
            self.logger.info(f"Configuration loaded from: {config_path}")
            
            # Setup directories
            self._setup_directories()
            
            self.logger.info("Data ingestion pipeline initialized successfully")
            self.logger.info("Note: Files saved locally. Use 'dvc push' to upload to DagsHub storage")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data ingestion: {e}")
            raise DataIngestionError(f"Initialization failed: {e}")
    
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
                Path("data/raw"),
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
            raise DataIngestionError(f"Directory setup failed: {e}")
    
    @log_function_call
    def fetch_stock_data(
        self, 
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_retries: int = 3
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (DataFrame with OHLCV data, metadata dictionary)
            
        Raises:
            DataIngestionError: If data fetching fails after retries
        """
        # Use config values if parameters not provided
        symbol = symbol or self.config['data_ingestion']['stock_symbol']
        start_date = start_date or self.config['data_ingestion']['start_date']
        end_date = datetime.now() or self.config['data_ingestion']['end_date']
        
        self.logger.info(f"Fetching data for {symbol}")
        self.logger.info(f"  Source: Yahoo Finance (yfinance)")
        self.logger.info(f"  Date Range: {start_date} to {end_date}")
        
        # Validate inputs
        self._validate_inputs(symbol, start_date, end_date)
        
        # Retry logic
        for attempt in range(1, max_retries + 1):
            try:
                df, metadata = self._fetch_from_yfinance(symbol, start_date, end_date)
                
                # Validate fetched data
                self._validate_data(df, symbol)
                
                self.logger.info(f"✓ Data fetched successfully on attempt {attempt}")
                return df, metadata
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_retries} attempts failed for {symbol}")
                    raise DataIngestionError(f"Failed to fetch data after {max_retries} attempts: {e}")
    
    def _validate_inputs(self, symbol: str, start_date: str, end_date: str):
        """
        Validate input parameters.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            
        Raises:
            DataIngestionError: If validation fails
        """
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                raise DataIngestionError("Invalid stock symbol")
            
            # Allow alphanumeric, dots, hyphens, underscores (for exchange suffixes like .BO, .NS, etc.)
            import re
            if not re.match(r'^[A-Za-z0-9._-]+$', symbol):
                raise DataIngestionError("Symbol must contain only alphanumeric characters, dots, hyphens, or underscores")
            
            # Validate dates
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
            
            if start >= end:
                raise DataIngestionError("Start date must be before end date")
            
            if end > datetime.now():
                self.logger.warning("End date is in the future, adjusting to today")
            
            date_range = (end - start).days
            if date_range < 30:
                self.logger.warning(f"Date range is only {date_range} days (minimum 30 recommended)")
            
            self.logger.debug("Input validation passed")
            
        except ValueError as e:
            raise DataIngestionError(f"Invalid date format: {e}")
        except Exception as e:
            raise DataIngestionError(f"Validation error: {e}")
    
    def _fetch_from_yfinance(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch data from Yahoo Finance using yfinance library.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Tuple of (DataFrame, metadata dictionary)
            
        Raises:
            DataIngestionError: If fetching fails
        """
        try:
            self.logger.info("Connecting to Yahoo Finance API...")
            
            # Create ticker object
            stock = yf.Ticker(symbol)
            
            # Download historical data
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise DataIngestionError(f"No data returned for {symbol}")
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Get stock metadata
            try:
                info = stock.info
                metadata = {
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', None),
                    'currency': info.get('currency', 'USD'),
                    'exchange': info.get('exchange', 'N/A'),
                    'country': info.get('country', 'N/A'),
                    'fetch_timestamp': datetime.now().isoformat(),
                    'data_source': 'yfinance',
                    'total_records': len(df),
                    'date_range': {
                        'start': df['Date'].min().isoformat(),
                        'end': df['Date'].max().isoformat()
                    },
                    'columns': list(df.columns)
                }
            except Exception as e:
                self.logger.warning(f"Could not fetch full metadata: {e}")
                metadata = {
                    'symbol': symbol,
                    'fetch_timestamp': datetime.now().isoformat(),
                    'data_source': 'yfinance',
                    'total_records': len(df),
                    'date_range': {
                        'start': df['Date'].min().isoformat(),
                        'end': df['Date'].max().isoformat()
                    }
                }
            
            self.logger.info(f"  Company: {metadata.get('company_name', symbol)}")
            self.logger.info(f"  Sector: {metadata.get('sector', 'N/A')}")
            self.logger.info(f"  Total Records: {len(df)}")
            self.logger.info(f"  Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            
            return df, metadata
            
        except Exception as e:
            self.logger.error(f"Error fetching from yfinance: {e}")
            raise DataIngestionError(f"yfinance fetch failed: {e}")
    
    def _validate_data(self, df: pd.DataFrame, symbol: str):
        """
        Validate fetched data quality.
        
        Args:
            df: DataFrame to validate
            symbol: Stock symbol for logging
            
        Raises:
            DataIngestionError: If validation fails
        """
        try:
            self.logger.info("Validating data quality...")
            
            # Check if DataFrame is empty
            if df.empty:
                raise DataIngestionError("DataFrame is empty")
            
            # Check required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataIngestionError(f"Missing required columns: {missing_columns}")
            
            # Check for all NaN columns
            all_nan_cols = df.columns[df.isna().all()].tolist()
            if all_nan_cols:
                self.logger.warning(f"Columns with all NaN values: {all_nan_cols}")
            
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                self.logger.warning("Date column is not datetime type, may need conversion")
            
            # Check for negative values in price columns
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if (df[col] < 0).any():
                    raise DataIngestionError(f"Negative values found in {col}")
            
            # Check for zero volume days
            zero_volume_days = (df['Volume'] == 0).sum()
            if zero_volume_days > 0:
                self.logger.warning(f"Found {zero_volume_days} days with zero volume")
            
            # Check minimum trading days
            min_trading_days = self.config['preprocessing'].get('min_trading_days', 252)
            if len(df) < min_trading_days:
                self.logger.warning(
                    f"Only {len(df)} trading days available (minimum {min_trading_days} recommended)"
                )
            
            # Check for duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate rows")
            
            self.logger.info("✓ Data validation passed")
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise DataIngestionError(f"Validation failed: {e}")
    
    @log_function_call
    def save_data(
        self, 
        df: pd.DataFrame, 
        metadata: Dict[str, Any],
        symbol: str,
        save_to_dagshub: bool = False
    ) -> Dict[str, str]:
        """
        Save data to local filesystem and optionally to DagsHub storage.
        
        Args:
            df: DataFrame to save
            metadata: Metadata dictionary
            symbol: Stock symbol
            save_to_dagshub: Whether to upload to DagsHub
            
        Returns:
            Dictionary with file paths
            
        Raises:
            DataIngestionError: If saving fails
        """
        try:
            self.logger.info("Saving data to disk...")
            
            # Generate file paths
            data_filename = f"stock_data_raw_{symbol}.csv"
            metadata_filename = f"stock_metadata_{symbol}.json"
            
            data_path = Path("data/raw") / data_filename
            metadata_path = Path("metadata") / metadata_filename
            
            # Save DataFrame
            df.to_csv(data_path, index=False)
            file_size_kb = data_path.stat().st_size / 1024
            self.logger.info(f"✓ Data saved to: {data_path}")
            self.logger.info(f"  File size: {file_size_kb:.2f} KB")
            self.logger.info(f"  Records: {len(df)}")
            
            # Add file info to metadata
            metadata['file_info'] = {
                'filename': data_filename,
                'path': str(data_path),
                'size_kb': round(file_size_kb, 2),
                'saved_timestamp': datetime.now().isoformat()
            }
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            self.logger.info(f"✓ Metadata saved to: {metadata_path}")
            
            result = {
                'data_path': str(data_path),
                'metadata_path': str(metadata_path)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise DataIngestionError(f"Save failed: {e}")
    
    @log_function_call
    def run_ingestion_pipeline(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_to_dagshub: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete data ingestion pipeline.
        
        This is the main entry point that orchestrates:
        1. Data fetching from source
        2. Data validation
        3. Saving to local and DagsHub storage
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            save_to_dagshub: Whether to upload to DagsHub
            
        Returns:
            Dictionary with pipeline results
            
        Raises:
            DataIngestionError: If pipeline fails
        """
        pipeline_start = time.time()
        
        try:
            self.logger.info("=" * 70)
            self.logger.info("STARTING DATA INGESTION PIPELINE")
            self.logger.info("=" * 70)
            
            # Step 1: Fetch data
            df, metadata = self.fetch_stock_data(symbol, start_date, end_date)
            
            # Step 2: Save data
            symbol = symbol or self.config['data_ingestion']['stock_symbol']
            
            # Auto-enable DagsHub upload if configured
            if not save_to_dagshub:
                dagshub_config = self.config.get('dagshub_storage', {})
                save_to_dagshub = dagshub_config.get('save_to_dagshub', False)
            
            file_paths = self.save_data(df, metadata, symbol, save_to_dagshub)
            
            # Calculate pipeline metrics
            pipeline_duration = time.time() - pipeline_start
            
            result = {
                'status': 'success',
                'symbol': symbol,
                'records_fetched': len(df),
                'file_paths': file_paths,
                'metadata': metadata,
                'pipeline_duration_seconds': round(pipeline_duration, 2)
            }
            
            self.logger.info("=" * 70)
            self.logger.info("DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Duration: {pipeline_duration:.2f} seconds")
            self.logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise DataIngestionError(f"Pipeline execution failed: {e}")


def main():
    """Main function to run data ingestion."""
    try:
        # Initialize ingestion pipeline
        ingestion = StockDataIngestion()
        
        # Run pipeline
        result = ingestion.run_ingestion_pipeline()
        
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        print(f"Status: {result['status']}")
        print(f"Symbol: {result['symbol']}")
        print(f"Records: {result['records_fetched']}")
        print(f"Duration: {result['pipeline_duration_seconds']}s")
        print(f"Data saved to: {result['file_paths']['data_path']}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
