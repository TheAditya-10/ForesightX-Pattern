"""
Model Training Module for ForesightX
====================================

This module trains ML models (MLP) for stock price prediction.
It implements production-ready training with:
- Data loading and preprocessing
- Model initialization and training
- Early stopping and model checkpointing
- Performance metrics and evaluation
- Model saving (local + S3)

Author: Aditya Pratap Singh Tomar
Date: December 2025
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.services.logger import get_logger, log_function_call


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class ModelTrainingError(Exception):
    """Custom exception for model training errors"""
    pass


# =====================================================================
# MODEL TRAINER CLASS
# =====================================================================

class MLPModelTrainer:
    """
    MLP Model Trainer for stock price prediction.
    
    Trains scikit-learn MLP (Multi-Layer Perceptron) for regression tasks.
    Includes data loading, feature scaling, model training, evaluation, and saving.
    """
    
    def __init__(self, config_path='params.yaml'):
        """
        Initialize MLP Model Trainer with configuration.
        
        Parameters:
        -----------
        config_path : str
            Path to YAML configuration file
        """
        self.logger = get_logger('MLPModelTrainer')
        self.config_path = config_path
        self.config = self._load_config()
        
        # Extract configuration
        self.training_config = self.config.get('training', {})
        self.model_config = self.config.get('models', {}).get('mlp', {})
        self.data_config = self.config.get('data_ingestion', {})
        
# Model and scaler
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        self.logger.info("MLPModelTrainer initialized successfully")
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
            raise ModelTrainingError(f"Config load failed: {str(e)}")
    
    @log_function_call
    def load_features(self, symbol):
        """
        Load feature-engineered data.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
            
        Returns:
        --------
        pd.DataFrame
            Feature-engineered dataframe
        """
        try:
            self.logger.info(f"Loading feature-engineered data for {symbol}...")
            
            # Load features file
            features_dir = os.path.join('data', 'features')
            features_file = os.path.join(features_dir, f'features_{symbol}.csv')
            
            if not os.path.exists(features_file):
                raise ModelTrainingError(f"No features file found for {symbol}: {features_file}")
            
            df = pd.read_csv(features_file)
            
            # Convert Date if present
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            self.logger.info(f"Loaded {len(df)} rows from {features_file}")
            self.logger.info(f"Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load features: {str(e)}")
            raise ModelTrainingError(f"Feature loading failed: {str(e)}")
    
    @log_function_call
    def prepare_data(self, df):
        """
        Prepare data for MLP training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature-engineered dataframe
            
        Returns:
        --------
        tuple
            (X_train, X_val, y_train, y_val)
        """
        try:
            self.logger.info("Preparing data for training...")
            
            # Remove non-feature columns
            exclude_cols = ['Date', 'Target_Return', 'Target_Direction', 'Target_Class']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Remove rows with NaN in target
            df_clean = df.dropna(subset=['Target_Return']).copy()
            
            # Features and target
            X = df_clean[feature_cols].values
            y = df_clean['Target_Return'].values
            
            self.feature_names = feature_cols
            
            # Train/validation split (chronological)
            val_size = self.training_config.get('validation_size', 0.2)
            
            n = len(df_clean)
            train_end = int(n * (1 - val_size))
            
            X_train = X[:train_end]
            X_val = X[train_end:]
            
            y_train = y[:train_end]
            y_val = y[train_end:]
            
            self.logger.info(f"Data split complete:")
            self.logger.info(f"  Train: {X_train.shape}")
            self.logger.info(f"  Val: {X_val.shape}")
            self.logger.info(f"  Features: {len(feature_cols)}")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {str(e)}")
            raise ModelTrainingError(f"Data preparation failed: {str(e)}")
    
    @log_function_call
    def scale_data(self, X_train, X_val):
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        X_train, X_val : numpy.ndarray
            Feature arrays
            
        Returns:
        --------
        tuple
            (X_train_scaled, X_val_scaled)
        """
        try:
            self.logger.info("Scaling features...")
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            # Fit on training data only
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            self.logger.info("Feature scaling complete")
            
            return X_train_scaled, X_val_scaled
            
        except Exception as e:
            self.logger.error(f"Failed to scale data: {str(e)}")
            raise ModelTrainingError(f"Data scaling failed: {str(e)}")
    
    @log_function_call
    def build_model(self):
        """
        Build MLP model with configured architecture.
        
        Returns:
        --------
        MLPRegressor
            Initialized MLP model
        """
        try:
            self.logger.info("Building MLP model...")
            
            # Model parameters from config
            hidden_layers = tuple(self.model_config.get('hidden_layer_sizes', [128, 64, 32]))
            activation = self.model_config.get('activation', 'relu')
            solver = self.model_config.get('solver', 'adam')
            learning_rate = self.model_config.get('learning_rate', 'adaptive')
            learning_rate_init = self.model_config.get('learning_rate_init', 0.001)
            max_iter = self.model_config.get('max_iter', 200)
            early_stopping = self.model_config.get('early_stopping', True)
            validation_fraction = self.model_config.get('validation_fraction', 0.1)
            n_iter_no_change = self.model_config.get('n_iter_no_change', 15)
            random_state = self.training_config.get('random_state', 42)
            
            # Initialize MLP
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver=solver,
                learning_rate=learning_rate,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                random_state=random_state,
                verbose=True
            )
            
            self.logger.info(f"MLP architecture: {hidden_layers}")
            self.logger.info(f"Activation: {activation}")
            self.logger.info(f"Solver: {solver}")
            self.logger.info(f"Learning rate: {learning_rate_init}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to build model: {str(e)}")
            raise ModelTrainingError(f"Model building failed: {str(e)}")
    
    @log_function_call
    def train_model(self, model, X_train, y_train, X_val, y_val):
        """
        Train MLP model.
        
        Parameters:
        -----------
        model : MLPRegressor
            Initialized model
        X_train, y_train : numpy.ndarray
            Training data
        X_val, y_val : numpy.ndarray
            Validation data
            
        Returns:
        --------
        MLPRegressor
            Trained model
        """
        try:
            self.logger.info("="*70)
            self.logger.info("STARTING MODEL TRAINING")
            self.logger.info("="*70)
            
            start_time = datetime.now()
            
            # Train model
            model.fit(X_train, y_train)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Training complete in {duration:.2f} seconds")
            self.logger.info(f"Iterations: {model.n_iter_}")
            
            # Validation predictions
            val_pred = model.predict(X_val)
            
            # Calculate metrics
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            
            self.logger.info("="*70)
            self.logger.info("VALIDATION METRICS")
            self.logger.info("="*70)
            self.logger.info(f"RMSE: {val_rmse:.6f}")
            self.logger.info(f"MAE: {val_mae:.6f}")
            
            return model, {
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'iterations': model.n_iter_,
                'training_time': duration
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            raise ModelTrainingError(f"Model training failed: {str(e)}")
    
    @log_function_call
    def save_model(self, model, scaler, symbol, metrics):
        """
        Save trained model, scaler, and metadata.
        
        Parameters:
        -----------
        model : MLPRegressor
            Trained model
        scaler : StandardScaler
            Fitted scaler
        symbol : str
            Stock ticker symbol
        metrics : dict
            Training and evaluation metrics
            
        Returns:
        --------
        dict
            Save results
        """
        try:
            self.logger.info("Saving model artifacts...")
            
            # Create directories
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model
            model_file = os.path.join(models_dir, f'mlp_model_{symbol}.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Save scaler
            scaler_file = os.path.join(models_dir, f'mlp_scaler_{symbol}.pkl')
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save metadata
            timestamp = datetime.now().isoformat()
            metadata = {
                'symbol': symbol,
                'timestamp': timestamp,
                'model_type': 'MLP',
                'architecture': self.model_config.get('hidden_layer_sizes', [128, 64, 32]),
                'features_count': len(self.feature_names),
                'feature_names': self.feature_names,
                'metrics': metrics,
                'model_file': model_file,
                'scaler_file': scaler_file,
                'config': {
                    'activation': self.model_config.get('activation', 'relu'),
                    'solver': self.model_config.get('solver', 'adam'),
                    'learning_rate_init': self.model_config.get('learning_rate_init', 0.001),
                    'iterations': model.n_iter_
                }
            }
            
            metadata_dir = 'metadata'
            os.makedirs(metadata_dir, exist_ok=True)
            metadata_file = os.path.join(metadata_dir, f'mlp_model_stats_{symbol}.json')
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved: {model_file}")
            self.logger.info(f"Scaler saved: {scaler_file}")
            self.logger.info(f"Metadata saved: {metadata_file}")
            
            return {
                'success': True,
                'model_file': model_file,
                'scaler_file': scaler_file,
                'metadata_file': metadata_file
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise ModelTrainingError(f"Model saving failed: {str(e)}")
    
    @log_function_call
    def run_training_pipeline(self, symbol):
        """
        Run complete MLP training pipeline.
        
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
            self.logger.info("STARTING MLP TRAINING PIPELINE")
            self.logger.info("="*70)
            
            start_time = datetime.now()
            
            # Step 1: Load features
            self.logger.info("\n[1/5] Loading feature-engineered data...")
            df = self.load_features(symbol)
            
            # Step 2: Prepare data
            self.logger.info("\n[2/5] Preparing data for training...")
            X_train, X_val, y_train, y_val = self.prepare_data(df)
            
            # Step 3: Scale features
            self.logger.info("\n[3/5] Scaling features...")
            X_train_scaled, X_val_scaled = self.scale_data(X_train, X_val)
            
            # Step 4: Build model
            self.logger.info("\n[4/5] Building MLP model...")
            model = self.build_model()
            
            # Step 5: Train model
            self.logger.info("\n[5/5] Training MLP model...")
            model, train_metrics = self.train_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Save model
            self.logger.info("\nSaving model artifacts...")
            save_result = self.save_model(model, self.scaler, symbol, train_metrics)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Summary
            result = {
                'success': True,
                'symbol': symbol,
                'model_type': 'MLP',
                'architecture': self.model_config.get('hidden_layer_sizes', [128, 64, 32]),
                'features_count': len(self.feature_names),
                'metrics': train_metrics,
                'duration_seconds': duration,
                'model_file': save_result['model_file'],
                'scaler_file': save_result['scaler_file'],
                'metadata_file': save_result['metadata_file']
            }
            
            self.logger.info("="*70)
            self.logger.info("MLP TRAINING PIPELINE COMPLETED")
            self.logger.info("="*70)
            self.logger.info(f"\n✓ Model: MLP {self.model_config.get('hidden_layer_sizes', [128, 64, 32])}")
            self.logger.info(f"✓ Features: {len(self.feature_names)}")
            self.logger.info(f"✓ Val RMSE: {train_metrics['val_rmse']:.6f}")
            self.logger.info(f"✓ Val MAE: {train_metrics['val_mae']:.6f}")
            self.logger.info(f"✓ Iterations: {train_metrics['iterations']}")
            self.logger.info(f"✓ Duration: {duration:.2f} seconds")
            self.logger.info(f"✓ Model saved: {save_result['model_file']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
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
        # Initialize trainer
        trainer = MLPModelTrainer()
        
        # Get symbol from config
        symbol = trainer.data_config['stock_symbol']
        
        # Run training pipeline
        result = trainer.run_training_pipeline(symbol)
        
        if result['success']:
            print(f"\n✅ MLP model training completed successfully!")
            print(f"   Model file: {result['model_file']}")
            print(f"   Validation RMSE: {result['metrics']['val_rmse']:.6f}")
            print(f"   Validation MAE: {result['metrics']['val_mae']:.6f}")
        else:
            print(f"\n❌ Training failed: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
