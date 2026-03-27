"""
Feature Engineering Module for ForesightX
=========================================

This module creates technical indicators and derived features from processed stock data.
It implements production-ready feature engineering with:
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Lag features and price changes
- Volatility metrics
- Volume-based indicators
- Calendar features
- Price pattern features
- Target variables for model training

Author: Aditya Pratap Singh Tomar
Date: December 2025
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.services.logger import get_logger, log_function_call


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class FeatureEngineeringError(Exception):
    """Custom exception for feature engineering errors"""
    pass


# =====================================================================
# FEATURE ENGINEERING CLASS
# =====================================================================

class FeatureEngineer:
    """
    Feature engineering pipeline for stock market data.
    
    Creates comprehensive feature set including:
    - Lag features
    - Technical indicators
    - Volatility metrics
    - Volume indicators
    - Calendar features
    - Price patterns
    - Target variables
    """
    
    def __init__(self, config_path='params.yaml'):
        """
        Initialize Feature Engineer with configuration.
        
        Parameters:
        -----------
        config_path : str
            Path to YAML configuration file
        """
        self.logger = get_logger('FeatureEngineer')
        self.config_path = config_path
        self.config = self._load_config()
        
        # Extract configuration
        self.feature_config = self.config.get('feature_engineering', {})
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.paths_config = self.config.get('paths', {})
        
        self.logger.info("FeatureEngineer initialized successfully")
        self.logger.info("Note: Files saved locally. Use 'dvc push' to upload to DagsHub storage")
    
    @log_function_call
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise FeatureEngineeringError(f"Config load failed: {str(e)}")
    
    @log_function_call
    def load_data(self, symbol):
        """
        Load preprocessed data for feature engineering.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed stock data
        """
        try:
            # Load from train data (already preprocessed and split)
            train_file = os.path.join('data', 'processed', f'train_data_{symbol}.csv')
            
            if not os.path.exists(train_file):
                raise FeatureEngineeringError(f"Training data not found: {train_file}")
            
            df = pd.read_csv(train_file)
            
            # Ensure Date is datetime and remove timezone if present
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                # Remove timezone to avoid issues with dt accessor
                df['Date'] = df['Date'].dt.tz_localize(None)
            
            self.logger.info(f"Loaded {len(df)} rows from {train_file}")
            self.logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise FeatureEngineeringError(f"Data loading failed: {str(e)}")
    
    @log_function_call
    def create_lag_features(self, df):
        """
        Create lag features for price, returns, and volume.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with lag features added
        """
        try:
            self.logger.info("Creating lag features...")
            
            # Calculate returns if not present
            if 'Return' not in df.columns:
                df['Return'] = df['Close'].pct_change()
            
            # Lag periods from config
            lag_periods = self.feature_config.get('lag_periods', [1, 2, 3, 5, 10])
            
            for lag in lag_periods:
                df[f'Price_Lag_{lag}'] = df['Close'].shift(lag)
                df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
                df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            
            # Price changes
            df['Price_Change_1d'] = df['Close'] - df['Close'].shift(1)
            df['Price_Change_5d'] = df['Close'] - df['Close'].shift(5)
            
            features_created = len(lag_periods) * 3 + 2
            self.logger.info(f"Created {features_created} lag features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create lag features: {str(e)}")
            raise FeatureEngineeringError(f"Lag feature creation failed: {str(e)}")
    
    @log_function_call
    def create_moving_averages(self, df):
        """
        Create Simple Moving Average (SMA) and Exponential Moving Average (EMA) features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with moving average features
        """
        try:
            self.logger.info("Creating moving average features...")
            
            # SMA windows from config
            sma_windows = self.feature_config.get('sma_windows', [5, 10, 20, 50, 200])
            
            for window in sma_windows:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'Price_to_SMA_{window}'] = df['Close'] / df[f'SMA_{window}'] - 1
            
            # EMA windows from config
            ema_windows = self.feature_config.get('ema_windows', [12, 26, 50])
            
            for window in ema_windows:
                df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
                df[f'Price_to_EMA_{window}'] = df['Close'] / df[f'EMA_{window}'] - 1
            
            # Moving average crossovers
            df['SMA_Cross_50_200'] = (df['SMA_50'] > df['SMA_200']).astype(int)
            df['EMA_Cross_12_26'] = (df['EMA_12'] > df['EMA_26']).astype(int)
            
            features_created = len(sma_windows) * 2 + len(ema_windows) * 2 + 2
            self.logger.info(f"Created {features_created} moving average features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create moving averages: {str(e)}")
            raise FeatureEngineeringError(f"Moving average creation failed: {str(e)}")
    
    @log_function_call
    def create_rsi(self, df):
        """
        Create Relative Strength Index (RSI) features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with RSI features
        """
        try:
            self.logger.info("Creating RSI features...")
            
            def calculate_rsi(data, window=14):
                """Calculate RSI indicator"""
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            # RSI windows from config
            rsi_windows = self.feature_config.get('rsi_windows', [14, 21])
            
            for window in rsi_windows:
                df[f'RSI_{window}'] = calculate_rsi(df['Close'], window)
                df[f'RSI_{window}_Oversold'] = (df[f'RSI_{window}'] < 30).astype(int)
                df[f'RSI_{window}_Overbought'] = (df[f'RSI_{window}'] > 70).astype(int)
            
            features_created = len(rsi_windows) * 3
            self.logger.info(f"Created {features_created} RSI features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create RSI: {str(e)}")
            raise FeatureEngineeringError(f"RSI creation failed: {str(e)}")
    
    @log_function_call
    def create_macd(self, df):
        """
        Create MACD (Moving Average Convergence Divergence) features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with MACD features
        """
        try:
            self.logger.info("Creating MACD features...")
            
            # Ensure EMA_12 and EMA_26 exist
            if 'EMA_12' not in df.columns:
                df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            if 'EMA_26' not in df.columns:
                df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD components
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # MACD signals
            df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
            df['MACD_Crossover'] = ((df['MACD'] > df['MACD_Signal']) & 
                                     (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
            
            self.logger.info("Created 5 MACD features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create MACD: {str(e)}")
            raise FeatureEngineeringError(f"MACD creation failed: {str(e)}")
    
    @log_function_call
    def create_bollinger_bands(self, df):
        """
        Create Bollinger Bands features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with Bollinger Bands features
        """
        try:
            self.logger.info("Creating Bollinger Bands features...")
            
            # Bollinger Bands parameters from config
            bb_window = self.feature_config.get('bollinger_window', 20)
            bb_std = self.feature_config.get('bollinger_std', 2)
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
            df['BB_Std'] = df['Close'].rolling(window=bb_window).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * df['BB_Std'])
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * df['BB_Std'])
            
            # Bollinger Band metrics
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Bollinger Band signals
            df['BB_Above_Upper'] = (df['Close'] > df['BB_Upper']).astype(int)
            df['BB_Below_Lower'] = (df['Close'] < df['BB_Lower']).astype(int)
            
            self.logger.info("Created 8 Bollinger Bands features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create Bollinger Bands: {str(e)}")
            raise FeatureEngineeringError(f"Bollinger Bands creation failed: {str(e)}")
    
    @log_function_call
    def create_atr(self, df):
        """
        Create Average True Range (ATR) features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with ATR features
        """
        try:
            self.logger.info("Creating ATR features...")
            
            def calculate_atr(high, low, close, window=14):
                """Calculate Average True Range"""
                high_low = high - low
                high_close = np.abs(high - close.shift())
                low_close = np.abs(low - close.shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=window).mean()
                return atr
            
            # ATR windows from config
            atr_windows = self.feature_config.get('atr_windows', [14, 21])
            
            for window in atr_windows:
                df[f'ATR_{window}'] = calculate_atr(df['High'], df['Low'], df['Close'], window)
                df[f'ATR_{window}_Pct'] = (df[f'ATR_{window}'] / df['Close']) * 100
            
            features_created = len(atr_windows) * 2
            self.logger.info(f"Created {features_created} ATR features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create ATR: {str(e)}")
            raise FeatureEngineeringError(f"ATR creation failed: {str(e)}")
    
    @log_function_call
    def create_volatility_features(self, df):
        """
        Create volatility-based features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with volatility features
        """
        try:
            self.logger.info("Creating volatility features...")
            
            # Ensure Return exists
            if 'Return' not in df.columns:
                df['Return'] = df['Close'].pct_change()
            
            # Rolling standard deviation of returns
            vol_windows = self.feature_config.get('volatility_windows', [5, 10, 20, 30])
            
            for window in vol_windows:
                df[f'Volatility_{window}d'] = df['Return'].rolling(window=window).std()
                df[f'Volatility_{window}d_Ann'] = df[f'Volatility_{window}d'] * np.sqrt(252)
            
            # Realized volatility (intraday range)
            df['Realized_Vol'] = (df['High'] - df['Low']) / df['Close']
            
            # Parkinson's volatility
            df['Parkinson_Vol'] = np.sqrt((1 / (4 * np.log(2))) * 
                                          ((np.log(df['High'] / df['Low'])) ** 2))
            
            # Volatility change
            df['Volatility_Change'] = df['Volatility_20d'].pct_change()
            
            features_created = len(vol_windows) * 2 + 3
            self.logger.info(f"Created {features_created} volatility features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create volatility features: {str(e)}")
            raise FeatureEngineeringError(f"Volatility feature creation failed: {str(e)}")
    
    @log_function_call
    def create_volume_features(self, df):
        """
        Create volume-based features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with volume features
        """
        try:
            self.logger.info("Creating volume features...")
            
            # Ensure Return exists
            if 'Return' not in df.columns:
                df['Return'] = df['Close'].pct_change()
            
            # Volume change
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_Change_5d'] = df['Volume'].pct_change(periods=5)
            
            # Volume moving averages
            vol_ma_windows = self.feature_config.get('volume_ma_windows', [5, 20, 50])
            
            for window in vol_ma_windows:
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
                df[f'Volume_Ratio_{window}'] = df['Volume'] / df[f'Volume_MA_{window}']
            
            # VWAP (Volume Weighted Average Price)
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['VWAP_20'] = (df['Typical_Price'] * df['Volume']).rolling(window=20).sum() / \
                            df['Volume'].rolling(window=20).sum()
            df['Price_to_VWAP'] = df['Close'] / df['VWAP_20'] - 1
            
            # On-Balance Volume (OBV)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            # Volume-Price Trend
            df['VPT'] = (df['Volume'] * df['Return']).cumsum()
            
            features_created = 2 + len(vol_ma_windows) * 2 + 5
            self.logger.info(f"Created {features_created} volume features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create volume features: {str(e)}")
            raise FeatureEngineeringError(f"Volume feature creation failed: {str(e)}")
    
    @log_function_call
    def create_calendar_features(self, df):
        """
        Create calendar/time-based features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with calendar features
        """
        try:
            self.logger.info("Creating calendar features...")
            
            # Ensure Date is datetime without timezone
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            
            # Day of week (0=Monday, 4=Friday)
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Is_Monday'] = (df['Day_of_Week'] == 0).astype(int)
            df['Is_Friday'] = (df['Day_of_Week'] == 4).astype(int)
            
            # Month and quarter
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            
            # Day of month
            df['Day_of_Month'] = df['Date'].dt.day
            df['Is_Month_Start'] = (df['Day_of_Month'] <= 5).astype(int)
            df['Is_Month_End'] = (df['Day_of_Month'] >= 25).astype(int)
            
            # Week of year
            df['Week_of_Year'] = df['Date'].dt.isocalendar().week
            
            # Seasonal indicators
            df['Is_January'] = (df['Month'] == 1).astype(int)
            df['Is_December'] = (df['Month'] == 12).astype(int)
            
            # Days since start of year
            year_starts = pd.to_datetime(df['Date'].dt.year.astype(str) + '-01-01')
            df['Days_Since_Year_Start'] = (df['Date'] - year_starts).dt.days
            
            self.logger.info("Created 14 calendar features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create calendar features: {str(e)}")
            raise FeatureEngineeringError(f"Calendar feature creation failed: {str(e)}")
    
    @log_function_call
    def create_price_pattern_features(self, df):
        """
        Create price pattern and momentum features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with price pattern features
        """
        try:
            self.logger.info("Creating price pattern features...")
            
            # Rate of Change (ROC)
            roc_periods = self.feature_config.get('roc_periods', [5, 10, 20])
            for period in roc_periods:
                df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / 
                                        df['Close'].shift(period)) * 100
            
            # Momentum
            df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
            
            # Price range features
            df['High_Low_Range'] = df['High'] - df['Low']
            df['High_Low_Range_Pct'] = (df['High_Low_Range'] / df['Close']) * 100
            
            # Body size (Open-Close range)
            df['Body_Size'] = abs(df['Close'] - df['Open'])
            df['Body_Size_Pct'] = (df['Body_Size'] / df['Close']) * 100
            
            # Upper/Lower shadows
            df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            
            # Bullish/Bearish candle
            df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
            
            # Gap up/down
            df['Gap'] = df['Open'] - df['Close'].shift(1)
            df['Gap_Pct'] = (df['Gap'] / df['Close'].shift(1)) * 100
            
            # Consecutive up/down days
            df['Price_Direction'] = np.sign(df['Close'] - df['Close'].shift(1))
            df['Consecutive_Up'] = df.groupby((df['Price_Direction'] != 
                                  df['Price_Direction'].shift()).cumsum())['Price_Direction'].cumsum()
            
            features_created = len(roc_periods) + 11
            self.logger.info(f"Created {features_created} price pattern features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create price pattern features: {str(e)}")
            raise FeatureEngineeringError(f"Price pattern feature creation failed: {str(e)}")
    
    @log_function_call
    def create_target_variables(self, df):
        """
        Create target variables for prediction.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with target variables
        """
        try:
            self.logger.info("Creating target variables...")
            
            # Ensure Return exists
            if 'Return' not in df.columns:
                df['Return'] = df['Close'].pct_change()
            
            # Next day's return (regression target)
            df['Target_Return'] = df['Return'].shift(-1)
            
            # Classification target (Up/Down)
            df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
            
            # Multi-class target (Strong Down, Down, Flat, Up, Strong Up)
            df['Target_Class'] = pd.cut(df['Target_Return'], 
                                        bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                                        labels=[0, 1, 2, 3, 4])
            
            self.logger.info("Created 3 target variables")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create target variables: {str(e)}")
            raise FeatureEngineeringError(f"Target variable creation failed: {str(e)}")
    
    @log_function_call
    def save_features(self, df, symbol):
        """
        Save feature-engineered data locally and optionally to S3.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature-engineered dataframe
        symbol : str
            Stock ticker symbol
            
        Returns:
        --------
        dict
            Result information
        """
        try:
            self.logger.info("Saving feature-engineered data...")
            
            # Create directories
            features_dir = os.path.join('data', 'features')
            os.makedirs(features_dir, exist_ok=True)
            
            # Save locally
            features_file = os.path.join(features_dir, f'features_{symbol}.csv')
            df.to_csv(features_file, index=False)
            
            self.logger.info(f"Features saved locally: {features_file}")
            self.logger.info(f"Shape: {df.shape}, Size: {os.path.getsize(features_file) / 1024:.2f} KB")
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'shape': df.shape,
                'features': list(df.columns),
                'date_range': {
                    'start': str(df['Date'].min()),
                    'end': str(df['Date'].max())
                },
                'missing_values': int(df.isnull().sum().sum()),
                'file_path': features_file
            }
            
            metadata_dir = 'metadata'
            os.makedirs(metadata_dir, exist_ok=True)
            metadata_file = os.path.join(metadata_dir, f'features_stats_{symbol}.json')
            
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Metadata saved: {metadata_file}")
            
            return {
                'success': True,
                'local_file': features_file,
                'metadata_file': metadata_file,
                'shape': df.shape,
                'features_count': len(df.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save features: {str(e)}")
            raise FeatureEngineeringError(f"Feature saving failed: {str(e)}")
    
    @log_function_call
    def run_feature_engineering_pipeline(self, symbol):
        """
        Run complete feature engineering pipeline.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
            
        Returns:
        --------
        dict
            Pipeline execution results
        """
        try:
            self.logger.info("="*70)
            self.logger.info("STARTING FEATURE ENGINEERING PIPELINE")
            self.logger.info("="*70)
            
            start_time = datetime.now()
            
            # Step 1: Load preprocessed data
            self.logger.info("\n[1/11] Loading preprocessed data...")
            df = self.load_data(symbol)
            initial_shape = df.shape
            
            # Step 2: Create lag features
            self.logger.info("\n[2/11] Creating lag features...")
            df = self.create_lag_features(df)
            
            # Step 3: Create moving averages
            self.logger.info("\n[3/11] Creating moving averages...")
            df = self.create_moving_averages(df)
            
            # Step 4: Create RSI
            self.logger.info("\n[4/11] Creating RSI indicators...")
            df = self.create_rsi(df)
            
            # Step 5: Create MACD
            self.logger.info("\n[5/11] Creating MACD indicators...")
            df = self.create_macd(df)
            
            # Step 6: Create Bollinger Bands
            self.logger.info("\n[6/11] Creating Bollinger Bands...")
            df = self.create_bollinger_bands(df)
            
            # Step 7: Create ATR
            self.logger.info("\n[7/11] Creating ATR indicators...")
            df = self.create_atr(df)
            
            # Step 8: Create volatility features
            self.logger.info("\n[8/11] Creating volatility features...")
            df = self.create_volatility_features(df)
            
            # Step 9: Create volume features
            self.logger.info("\n[9/11] Creating volume features...")
            df = self.create_volume_features(df)
            
            # Step 10: Create calendar features
            self.logger.info("\n[10/11] Creating calendar features...")
            df = self.create_calendar_features(df)
            
            # Step 11: Create price pattern features
            self.logger.info("\n[11/11] Creating price pattern features...")
            df = self.create_price_pattern_features(df)
            
            # Step 12: Create target variables
            self.logger.info("\n[12/12] Creating target variables...")
            df = self.create_target_variables(df)
            
            # Remove rows with NaN (from rolling calculations)
            self.logger.info("\nCleaning up NaN values...")
            before_cleanup = len(df)
            df = df.dropna(subset=[col for col in df.columns if 'Target' not in col])
            after_cleanup = len(df)
            
            self.logger.info(f"Removed {before_cleanup - after_cleanup} rows with NaN")
            self.logger.info(f"Data retained: {after_cleanup / before_cleanup * 100:.1f}%")
            
            # Save features
            self.logger.info("\nSaving feature-engineered data...")
            save_result = self.save_features(df, symbol)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Summary
            result = {
                'success': True,
                'symbol': symbol,
                'initial_shape': initial_shape,
                'final_shape': df.shape,
                'features_created': df.shape[1] - initial_shape[1],
                'rows_retained': after_cleanup,
                'rows_removed': before_cleanup - after_cleanup,
                'retention_rate': f"{after_cleanup / before_cleanup * 100:.1f}%",
                'duration_seconds': duration,
                'local_file': save_result['local_file'],
                'metadata_file': save_result['metadata_file']
            }
            
            self.logger.info("="*70)
            self.logger.info("FEATURE ENGINEERING PIPELINE COMPLETED")
            self.logger.info("="*70)
            self.logger.info(f"\n✓ Initial shape: {initial_shape}")
            self.logger.info(f"✓ Final shape: {df.shape}")
            self.logger.info(f"✓ Features created: {result['features_created']}")
            self.logger.info(f"✓ Duration: {duration:.2f} seconds")
            self.logger.info(f"✓ Output: {save_result['local_file']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Feature engineering pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main execution function"""
    try:
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Get symbol from config
        symbol = engineer.config['data_ingestion']['stock_symbol']
        
        # Run feature engineering pipeline
        result = engineer.run_feature_engineering_pipeline(symbol)
        
        if result['success']:
            print(f"\n✅ Feature engineering completed successfully!")
            print(f"   Output file: {result['local_file']}")
            print(f"   Features created: {result['features_created']}")
            print(f"   Final shape: {result['final_shape']}")
        else:
            print(f"\n❌ Feature engineering failed: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
