"""
Model Evaluation Module for ForesightX
======================================

This module evaluates trained ML models on test data with MLflow tracking.
It implements production-ready evaluation with:
- Model and scaler loading
- Test set preparation
- Comprehensive metrics calculation
- MLflow experiment tracking (DagsHub integration)
- Metrics persistence (local + DagsHub)

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
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Add project root to path
sys.path.insert(0, str(project_root))

from src.services.logger import get_logger, log_function_call


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class ModelEvaluationError(Exception):
    """Custom exception for model evaluation errors"""
    pass


# =====================================================================
# MODEL EVALUATOR CLASS
# =====================================================================

class MLPModelEvaluator:
    """
    MLP Model Evaluator with MLflow tracking.
    
    Evaluates trained models on test data and logs results to MLflow/DagsHub.
    """
    
    def __init__(self, config_path='params.yaml'):
        """
        Initialize Model Evaluator with configuration.
        
        Parameters:
        -----------
        config_path : str
            Path to YAML configuration file
        """
        self.logger = get_logger('MLPModelEvaluator')
        self.config_path = config_path
        self.config = self._load_config()
        
        # Extract configuration
        self.training_config = self.config.get('training', {})
        self.model_config = self.config.get('models', {}).get('mlp', {})
        self.data_config = self.config.get('data_ingestion', {})
        self.mlflow_config = self.config.get('mlflow', {})
        
# Initialize MLflow
        self._setup_mlflow()
        
        # Model artifacts
        self.model = None
        self.scaler = None
        self.metadata = None
        
        self.logger.info("MLPModelEvaluator initialized successfully")
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
            raise ModelEvaluationError(f"Config load failed: {str(e)}")
    
    @log_function_call
    def _setup_mlflow(self):
        """Setup MLflow with DagsHub integration"""
        try:
            # Get DagsHub token from environment
            dagshub_token = os.environ.get('DAGSHUB_TOKEN')
            
            if not dagshub_token:
                self.logger.warning("DAGSHUB_TOKEN not found in environment variables")
                self.logger.warning("MLflow tracking will use local storage only")
                self.mlflow_enabled = False
                return
            
            # DagsHub configuration
            dagshub_username = self.mlflow_config.get('dagshub_username', 'your-username')
            dagshub_repo = self.mlflow_config.get('dagshub_repo', 'ForesightX')
            
            # Set MLflow tracking URI
            tracking_uri = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set credentials
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            
            # Set experiment
            experiment_name = self.mlflow_config.get('experiment_name', 'MLP_Stock_Prediction')
            mlflow.set_experiment(experiment_name)
            
            self.mlflow_enabled = True
            self.logger.info(f"MLflow tracking configured: {tracking_uri}")
            self.logger.info(f"Experiment: {experiment_name}")
            
        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {str(e)} - using local tracking only")
            self.mlflow_enabled = False
    
    @log_function_call
    def load_model(self, model_path, scaler_path, metadata_path):
        """
        Load trained model, scaler, and metadata.
        
        Parameters:
        -----------
        model_path : str
            Path to pickled model file
        scaler_path : str
            Path to pickled scaler file
        metadata_path : str
            Path to metadata JSON file
            
        Returns:
        --------
        tuple
            (model, scaler, metadata)
        """
        try:
            self.logger.info("Loading model artifacts...")
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded from: {model_path}")
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info(f"Scaler loaded from: {scaler_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.logger.info(f"Metadata loaded from: {metadata_path}")
            
            # Log model info
            self.logger.info(f"Model type: {self.metadata.get('model_type', 'Unknown')}")
            self.logger.info(f"Architecture: {self.metadata.get('architecture', 'Unknown')}")
            self.logger.info(f"Features: {self.metadata.get('features_count', 0)}")
            
            return self.model, self.scaler, self.metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise ModelEvaluationError(f"Model loading failed: {str(e)}")
    
    @log_function_call
    def load_test_data(self, symbol):
        """
        Load and prepare test data.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
            
        Returns:
        --------
        tuple
            (X_test, y_test, test_dates)
        """
        try:
            self.logger.info(f"Loading test data for {symbol}...")
            
            # Load features file
            features_dir = os.path.join('data', 'features')
            features_file = os.path.join(features_dir, f'features_{symbol}.csv')
            
            if not os.path.exists(features_file):
                raise ModelEvaluationError(f"No features file found for {symbol}: {features_file}")
            
            df = pd.read_csv(features_file)
            
            # Convert Date if present
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Remove non-feature columns
            exclude_cols = ['Date', 'Target_Return', 'Target_Direction', 'Target_Class']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Get feature names from metadata
            if self.metadata and 'feature_names' in self.metadata:
                feature_cols = self.metadata['feature_names']
            
            # Remove rows with NaN in target
            df_clean = df.dropna(subset=['Target_Return']).copy()
            
            # Test set split (chronological)
            test_size = self.training_config.get('test_size', 0.2)
            n = len(df_clean)
            test_start = int(n * (1 - test_size))
            
            test_df = df_clean.iloc[test_start:].copy()
            
            # Extract features and target
            X_test = test_df[feature_cols].values
            y_test = test_df['Target_Return'].values
            test_dates = test_df['Date'].values if 'Date' in test_df.columns else None
            
            self.logger.info(f"Test data loaded: {X_test.shape}")
            self.logger.info(f"Test samples: {len(y_test)}")
            
            return X_test, y_test, test_dates
            
        except Exception as e:
            self.logger.error(f"Failed to load test data: {str(e)}")
            raise ModelEvaluationError(f"Test data loading failed: {str(e)}")
    
    @log_function_call
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model on test set with comprehensive metrics.
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test targets
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        try:
            self.logger.info("Evaluating model on test set...")
            
            # Scale test features
            X_test_scaled = self.scaler.transform(X_test)
            
            # Predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to percentage
            
            # Directional accuracy (did we predict the correct direction?)
            y_test_sign = np.sign(y_test)
            y_pred_sign = np.sign(y_pred)
            direction_accuracy = np.mean(y_test_sign == y_pred_sign)
            
            # Calculate up/down movement accuracy separately
            up_moves = y_test > 0
            down_moves = y_test < 0
            
            up_accuracy = np.mean(y_test_sign[up_moves] == y_pred_sign[up_moves]) if np.any(up_moves) else 0
            down_accuracy = np.mean(y_test_sign[down_moves] == y_pred_sign[down_moves]) if np.any(down_moves) else 0
            
            # Prediction statistics
            pred_mean = np.mean(y_pred)
            pred_std = np.std(y_pred)
            actual_mean = np.mean(y_test)
            actual_std = np.std(y_test)
            
            # Residual analysis
            residuals = y_test - y_pred
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            
            metrics = {
                'test_rmse': float(rmse),
                'test_mae': float(mae),
                'test_mape': float(mape),
                'direction_accuracy': float(direction_accuracy),
                'up_movement_accuracy': float(up_accuracy),
                'down_movement_accuracy': float(down_accuracy),
                'prediction_mean': float(pred_mean),
                'prediction_std': float(pred_std),
                'actual_mean': float(actual_mean),
                'actual_std': float(actual_std),
                'residual_mean': float(residual_mean),
                'residual_std': float(residual_std),
                'test_samples': int(len(y_test)),
                'up_movements': int(np.sum(up_moves)),
                'down_movements': int(np.sum(down_moves))
            }
            
            self.logger.info("="*70)
            self.logger.info("TEST SET EVALUATION METRICS")
            self.logger.info("="*70)
            self.logger.info(f"RMSE: {rmse:.6f}")
            self.logger.info(f"MAE: {mae:.6f}")
            self.logger.info(f"MAPE: {mape:.2f}%")
            self.logger.info(f"Direction Accuracy: {direction_accuracy:.4f} ({direction_accuracy*100:.2f}%)")
            self.logger.info(f"Up Movement Accuracy: {up_accuracy:.4f} ({up_accuracy*100:.2f}%)")
            self.logger.info(f"Down Movement Accuracy: {down_accuracy:.4f} ({down_accuracy*100:.2f}%)")
            self.logger.info(f"Test Samples: {len(y_test)}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate model: {str(e)}")
            raise ModelEvaluationError(f"Model evaluation failed: {str(e)}")
    
    @log_function_call
    def save_evaluation_results(self, symbol, metrics, predictions=None):
        """
        Save evaluation metrics to local file and optionally DagsHub.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        metrics : dict
            Evaluation metrics
        predictions : dict, optional
            Predictions data (y_test, y_pred, dates)
            
        Returns:
        --------
        dict
            Save results
        """
        try:
            self.logger.info("Saving evaluation results...")
            
            # Create directories
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            
            # Combine with model metadata
            full_results = {
                'symbol': symbol,
                'evaluation_timestamp': datetime.now().isoformat(),
                'model_info': {
                    'model_type': self.metadata.get('model_type', 'Unknown'),
                    'architecture': self.metadata.get('architecture', []),
                    'features_count': self.metadata.get('features_count', 0),
                    'training_timestamp': self.metadata.get('timestamp', 'Unknown')
                },
                'test_metrics': metrics
            }
            
            # Add training metrics if available
            if self.metadata and 'metrics' in self.metadata:
                full_results['training_metrics'] = self.metadata['metrics']
            
            # Save metrics
            metrics_file = os.path.join(results_dir, f'evaluation_metrics_{symbol}.json')
            with open(metrics_file, 'w') as f:
                json.dump(full_results, f, indent=2)
            
            self.logger.info(f"Metrics saved: {metrics_file}")
            
            # Save predictions if provided
            predictions_file = None
            if predictions is not None:
                predictions_file = os.path.join(results_dir, f'predictions_{symbol}.csv')
                pred_df = pd.DataFrame({
                    'Date': predictions.get('dates', range(len(predictions['y_test']))),
                    'Actual': predictions['y_test'],
                    'Predicted': predictions['y_pred'],
                    'Error': predictions['y_test'] - predictions['y_pred'],
                    'Abs_Error': np.abs(predictions['y_test'] - predictions['y_pred']),
                    'Direction_Correct': np.sign(predictions['y_test']) == np.sign(predictions['y_pred'])
                })
                pred_df.to_csv(predictions_file, index=False)
                self.logger.info(f"Predictions saved: {predictions_file}")
            
            return {
                'success': True,
                'metrics_file': metrics_file,
                'predictions_file': predictions_file
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise ModelEvaluationError(f"Results saving failed: {str(e)}")
    
    @log_function_call
    def log_to_mlflow(self, symbol, metrics, model_path, scaler_path):
        """
        Log metrics, parameters, and artifacts to MLflow.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        metrics : dict
            Evaluation metrics
        model_path : str
            Path to model file
        scaler_path : str
            Path to scaler file
        """
        if not self.mlflow_enabled:
            self.logger.warning("MLflow not enabled - skipping MLflow logging")
            return
        
        try:
            self.logger.info("Logging to MLflow/DagsHub...")
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"MLP_Eval_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log parameters
                mlflow.log_param("symbol", symbol)
                mlflow.log_param("model_type", self.metadata.get('model_type', 'MLP'))
                mlflow.log_param("architecture", str(self.metadata.get('architecture', [])))
                mlflow.log_param("features_count", self.metadata.get('features_count', 0))
                mlflow.log_param("activation", self.metadata.get('config', {}).get('activation', 'relu'))
                mlflow.log_param("solver", self.metadata.get('config', {}).get('solver', 'adam'))
                mlflow.log_param("learning_rate", self.metadata.get('config', {}).get('learning_rate_init', 0.001))
                
                # Log training metrics if available
                if self.metadata and 'metrics' in self.metadata:
                    train_metrics = self.metadata['metrics']
                    if 'val_rmse' in train_metrics:
                        mlflow.log_metric("val_rmse", train_metrics['val_rmse'])
                    if 'val_mae' in train_metrics:
                        mlflow.log_metric("val_mae", train_metrics['val_mae'])
                    if 'iterations' in train_metrics:
                        mlflow.log_metric("training_iterations", train_metrics['iterations'])
                    if 'training_time' in train_metrics:
                        mlflow.log_metric("training_time_seconds", train_metrics['training_time'])
                
                # Log test metrics
                mlflow.log_metric("test_rmse", metrics['test_rmse'])
                mlflow.log_metric("test_mae", metrics['test_mae'])
                mlflow.log_metric("test_mape", metrics['test_mape'])
                mlflow.log_metric("direction_accuracy", metrics['direction_accuracy'])
                mlflow.log_metric("up_movement_accuracy", metrics['up_movement_accuracy'])
                mlflow.log_metric("down_movement_accuracy", metrics['down_movement_accuracy'])
                mlflow.log_metric("test_samples", metrics['test_samples'])
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                # Log artifacts
                mlflow.log_artifact(model_path, "model_artifacts")
                mlflow.log_artifact(scaler_path, "model_artifacts")
                
                # Log metadata file if exists
                if self.metadata:
                    metadata_path = self.metadata.get('model_file', '').replace('.pkl', '_stats.json')
                    metadata_path = metadata_path.replace('mlp_model_', 'mlp_model_stats_')
                    metadata_path = metadata_path.replace('models/', 'metadata/')
                    if os.path.exists(metadata_path):
                        mlflow.log_artifact(metadata_path, "metadata")
                
                # Set tags
                mlflow.set_tag("stock_symbol", symbol)
                mlflow.set_tag("model_family", "MLP")
                mlflow.set_tag("task", "regression")
                mlflow.set_tag("target", "stock_returns")
                
                run_id = mlflow.active_run().info.run_id
                self.logger.info(f"MLflow run completed: {run_id}")
                self.logger.info("Metrics and artifacts logged to DagsHub")
            
        except Exception as e:
            self.logger.error(f"Failed to log to MLflow: {str(e)}")
            self.logger.warning("Continuing without MLflow logging")
    
    @log_function_call
    def run_evaluation_pipeline(self, symbol, model_path=None, scaler_path=None, metadata_path=None):
        """
        Run complete evaluation pipeline.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        model_path : str, optional
            Path to model file (auto-detected if None)
        scaler_path : str, optional
            Path to scaler file (auto-detected if None)
        metadata_path : str, optional
            Path to metadata file (auto-detected if None)
            
        Returns:
        --------
        dict
            Evaluation results
        """
        try:
            self.logger.info("="*70)
            self.logger.info("STARTING MODEL EVALUATION PIPELINE")
            self.logger.info("="*70)
            
            start_time = datetime.now()
            
            # Step 1: Auto-detect model files if not provided
            if not model_path or not scaler_path or not metadata_path:
                self.logger.info("\n[1/5] Auto-detecting latest model files...")
                models_dir = 'models'
                metadata_dir = 'metadata'
                
                # Use fixed filenames
                model_path = os.path.join(models_dir, f'mlp_model_{symbol}.pkl')
                scaler_path = os.path.join(models_dir, f'mlp_scaler_{symbol}.pkl')
                metadata_path = os.path.join(metadata_dir, f'mlp_model_stats_{symbol}.json')
                
                if not os.path.exists(model_path):
                    raise ModelEvaluationError(f"No model file found: {model_path}")
                
                self.logger.info(f"Using model: {model_path}")
                self.logger.info(f"Using scaler: {scaler_path}")
                self.logger.info(f"Using metadata: {metadata_path}")
            else:
                self.logger.info("\n[1/5] Using provided model files...")
            
            # Step 2: Load model
            self.logger.info("\n[2/5] Loading model artifacts...")
            self.load_model(model_path, scaler_path, metadata_path)
            
            # Step 3: Load test data
            self.logger.info("\n[3/5] Loading test data...")
            X_test, y_test, test_dates = self.load_test_data(symbol)
            
            # Step 4: Evaluate model
            self.logger.info("\n[4/5] Evaluating model...")
            metrics = self.evaluate_model(X_test, y_test)
            
            # Get predictions for saving
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            predictions = {
                'y_test': y_test,
                'y_pred': y_pred,
                'dates': test_dates
            }
            
            # Step 5: Save results
            self.logger.info("\n[5/5] Saving evaluation results...")
            save_result = self.save_evaluation_results(symbol, metrics, predictions)
            
            # Log to MLflow
            self.logger.info("\nLogging to MLflow/DagsHub...")
            self.log_to_mlflow(symbol, metrics, model_path, scaler_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Summary
            result = {
                'success': True,
                'symbol': symbol,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metrics': metrics,
                'duration_seconds': duration,
                'results_file': save_result['metrics_file'],
                'predictions_file': save_result['predictions_file'],
                'mlflow_logged': self.mlflow_enabled
            }
            
            self.logger.info("="*70)
            self.logger.info("EVALUATION PIPELINE COMPLETED")
            self.logger.info("="*70)
            self.logger.info(f"\n✓ Test RMSE: {metrics['test_rmse']:.6f}")
            self.logger.info(f"✓ Test MAE: {metrics['test_mae']:.6f}")
            self.logger.info(f"✓ Test MAPE: {metrics['test_mape']:.2f}%")
            self.logger.info(f"✓ Direction Accuracy: {metrics['direction_accuracy']:.4f} ({metrics['direction_accuracy']*100:.2f}%)")
            self.logger.info(f"✓ Test Samples: {metrics['test_samples']}")
            self.logger.info(f"✓ Duration: {duration:.2f} seconds")
            self.logger.info(f"✓ Results saved: {save_result['metrics_file']}")
            self.logger.info(f"✓ MLflow logged: {self.mlflow_enabled}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation pipeline failed: {str(e)}")
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
        # Initialize evaluator
        evaluator = MLPModelEvaluator()
        
        # Get symbol from config
        symbol = evaluator.data_config['stock_symbol']
        
        # Run evaluation pipeline
        result = evaluator.run_evaluation_pipeline(symbol)
        
        if result['success']:
            print(f"\n✅ Model evaluation completed successfully!")
            print(f"   Test RMSE: {result['metrics']['test_rmse']:.6f}")
            print(f"   Test MAE: {result['metrics']['test_mae']:.6f}")
            print(f"   Test MAPE: {result['metrics']['test_mape']:.2f}%")
            print(f"   Direction Accuracy: {result['metrics']['direction_accuracy']:.4f} ({result['metrics']['direction_accuracy']*100:.2f}%)")
            print(f"   Results file: {result['results_file']}")
            print(f"   MLflow logged: {result['mlflow_logged']}")
        else:
            print(f"\n❌ Evaluation failed: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
