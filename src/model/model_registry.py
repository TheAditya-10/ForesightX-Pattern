"""
Model Registry Module for ForesightX
====================================

This module manages model registration in MLflow Model Registry (DagsHub).
It implements production-ready model versioning with:
- Model loading and validation
- Model registration with versioning
- Stage management (Staging, Production, Archived)
- Model metadata tracking
- Version comparison and promotion

Author: Aditya Pratap Singh Tomar
Date: December 2025
"""

import os
import sys
import yaml
import json
import pickle
from datetime import datetime
from pathlib import Path
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
from mlflow.exceptions import MlflowException

# Add project root to path
sys.path.insert(0, str(project_root))

from src.services.logger import get_logger, log_function_call


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class ModelRegistryError(Exception):
    """Custom exception for model registry errors"""
    pass


# =====================================================================
# MODEL REGISTRY CLASS
# =====================================================================

class ModelRegistry:
    """
    Model Registry Manager for MLflow/DagsHub.
    
    Handles model registration, versioning, and stage management.
    """
    
    def __init__(self, config_path='params.yaml'):
        """
        Initialize Model Registry Manager.
        
        Parameters:
        -----------
        config_path : str
            Path to YAML configuration file
        """
        self.logger = get_logger('ModelRegistry')
        self.config_path = config_path
        self.config = self._load_config()
        
        # Extract configuration
        self.mlflow_config = self.config.get('mlflow', {})
        self.registry_config = self.config.get('model_registry', {})
        self.data_config = self.config.get('data_ingestion', {})
        
        # Initialize MLflow client
        self.client = None
        self.mlflow_enabled = False
        self._setup_mlflow()
        
        self.logger.info("ModelRegistry initialized successfully")
    
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
            raise ModelRegistryError(f"Config load failed: {str(e)}")
    
    @log_function_call
    def _setup_mlflow(self):
        """Setup MLflow with DagsHub integration"""
        try:
            # Check if MLflow is enabled
            if not self.mlflow_config.get('enabled', False):
                self.logger.warning("MLflow is disabled in configuration")
                self.mlflow_enabled = False
                return
            
            # Get DagsHub token from environment
            dagshub_token = os.environ.get('DAGSHUB_TOKEN')
            
            if not dagshub_token:
                self.logger.warning("DAGSHUB_TOKEN not found in environment variables")
                self.logger.warning("Model registry will not be available")
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
            
            # Initialize MLflow client
            self.client = MlflowClient()
            
            self.mlflow_enabled = True
            self.logger.info(f"MLflow tracking configured: {tracking_uri}")
            self.logger.info(f"Model Registry enabled")
            
        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {str(e)} - registry not available")
            self.mlflow_enabled = False
    
    @log_function_call
    def load_model_info(self, model_path=None, scaler_path=None, metadata_path=None, symbol=None):
        """
        Load model information from files.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to model file (auto-detected if None)
        scaler_path : str, optional
            Path to scaler file (auto-detected if None)
        metadata_path : str, optional
            Path to metadata file (auto-detected if None)
        symbol : str, optional
            Stock symbol (from config if None)
            
        Returns:
        --------
        dict
            Model information including paths, metadata, and performance
        """
        try:
            self.logger.info("Loading model information...")
            
            # Get symbol
            if symbol is None:
                symbol = self.data_config.get('stock_symbol', 'AAPL')
            
            # Auto-detect model files if not provided
            if not model_path or not scaler_path or not metadata_path:
                self.logger.info("Auto-detecting latest model files...")
                models_dir = 'models'
                metadata_dir = 'metadata'
                
                # Use fixed filenames
                model_path = os.path.join(models_dir, f'mlp_model_{symbol}.pkl')
                scaler_path = os.path.join(models_dir, f'mlp_scaler_{symbol}.pkl')
                metadata_path = os.path.join(metadata_dir, f'mlp_model_stats_{symbol}.json')
                
                if not os.path.exists(model_path):
                    raise ModelRegistryError(f"No model file found: {model_path}")
            
            # Verify files exist
            if not os.path.exists(model_path):
                raise ModelRegistryError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise ModelRegistryError(f"Scaler file not found: {scaler_path}")
            if not os.path.exists(metadata_path):
                raise ModelRegistryError(f"Metadata file not found: {metadata_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load model to get details
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Compile model information
            model_info = {
                'symbol': symbol,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metadata_path': metadata_path,
                'model_type': metadata.get('model_type', 'MLP'),
                'architecture': metadata.get('architecture', []),
                'features_count': metadata.get('features_count', 0),
                'feature_names': metadata.get('feature_names', []),
                'timestamp': metadata.get('timestamp', ''),
                'metrics': metadata.get('metrics', {}),
                'config': metadata.get('config', {}),
                'model_object': model,
                'scaler_object': scaler,
                'metadata': metadata
            }
            
            self.logger.info(f"Model info loaded successfully")
            self.logger.info(f"  Model: {model_info['model_type']}")
            self.logger.info(f"  Architecture: {model_info['architecture']}")
            self.logger.info(f"  Features: {model_info['features_count']}")
            self.logger.info(f"  Timestamp: {model_info['timestamp']}")
            
            # Log metrics if available
            if 'val_rmse' in model_info['metrics']:
                self.logger.info(f"  Val RMSE: {model_info['metrics']['val_rmse']:.6f}")
            if 'test_rmse' in model_info['metrics']:
                self.logger.info(f"  Test RMSE: {model_info['metrics']['test_rmse']:.6f}")
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to load model info: {str(e)}")
            raise ModelRegistryError(f"Model info loading failed: {str(e)}")
    
    @log_function_call
    def register_model(self, model_info, model_name=None, stage=None, description=None, tags=None):
        """
        Register model in MLflow Model Registry.
        
        Parameters:
        -----------
        model_info : dict
            Model information from load_model_info()
        model_name : str, optional
            Name for registered model (default: "MLP_Stock_Predictor_<SYMBOL>")
        stage : str, optional
            Stage to assign (from config if None): "Staging", "Production", "Archived", or None
        description : str, optional
            Model description
        tags : dict, optional
            Additional tags for the model
            
        Returns:
        --------
        dict
            Registration result with version and stage info
        """
        if not self.mlflow_enabled:
            raise ModelRegistryError("MLflow not enabled - cannot register model")
        
        try:
            self.logger.info("="*70)
            self.logger.info("REGISTERING MODEL IN MLFLOW REGISTRY")
            self.logger.info("="*70)
            
            # Get model name
            if model_name is None:
                symbol = model_info['symbol']
                model_name = f"MLP_Stock_Predictor_{symbol}"
            
            # Get stage from params.yaml if not provided
            if stage is None:
                stage = self.registry_config.get('default_stage', None)
            
            # Validate stage
            valid_stages = [None, 'None', 'Staging', 'Production', 'Archived']
            if stage not in valid_stages:
                self.logger.warning(f"Invalid stage '{stage}'. Using None.")
                stage = None
            
            self.logger.info(f"Model name: {model_name}")
            self.logger.info(f"Target stage: {stage if stage else 'None'}")
            
            # Start MLflow run for registration
            with mlflow.start_run(run_name=f"Register_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log parameters
                mlflow.log_param("symbol", model_info['symbol'])
                mlflow.log_param("model_type", model_info['model_type'])
                mlflow.log_param("architecture", str(model_info['architecture']))
                mlflow.log_param("features_count", model_info['features_count'])
                mlflow.log_param("timestamp", model_info['timestamp'])
                
                # Log configuration
                for key, value in model_info['config'].items():
                    mlflow.log_param(f"config_{key}", value)
                
                # Log metrics
                metrics = model_info['metrics']
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # Log model with sklearn flavor
                self.logger.info("Logging model to MLflow...")
                mlflow.sklearn.log_model(
                    sk_model=model_info['model_object'],
                    name="model",
                    registered_model_name=model_name
                )
                
                # Log scaler as artifact
                mlflow.log_artifact(model_info['scaler_path'], "artifacts")
                mlflow.log_artifact(model_info['metadata_path'], "artifacts")
                
                # Set tags
                mlflow.set_tag("model_family", "MLP")
                mlflow.set_tag("task", "regression")
                mlflow.set_tag("target", "stock_returns")
                mlflow.set_tag("symbol", model_info['symbol'])
                mlflow.set_tag("version_timestamp", model_info['timestamp'])
                
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                # Get run info
                run_id = mlflow.active_run().info.run_id
                run = self.client.get_run(run_id)
                
                self.logger.info(f"Model logged successfully (Run ID: {run_id})")
            
            # Get the latest version of the registered model
            try:
                model_versions = self.client.search_model_versions(f"name='{model_name}'")
                latest_version = max([int(mv.version) for mv in model_versions])
                self.logger.info(f"Registered as version: {latest_version}")
            except Exception as e:
                self.logger.warning(f"Could not get version number: {str(e)}")
                latest_version = "Unknown"
            
            # Update model version stage if specified
            if stage and stage != 'None':
                try:
                    self.logger.info(f"Transitioning to stage: {stage}")
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version,
                        stage=stage,
                        archive_existing_versions=False
                    )
                    self.logger.info(f"✓ Model transitioned to {stage} stage")
                except Exception as e:
                    self.logger.warning(f"Stage transition failed: {str(e)}")
                    stage = None
            
            # Update model version description if provided
            if description:
                try:
                    self.client.update_model_version(
                        name=model_name,
                        version=latest_version,
                        description=description
                    )
                    self.logger.info(f"✓ Description added")
                except Exception as e:
                    self.logger.warning(f"Description update failed: {str(e)}")
            else:
                # Auto-generate description
                auto_description = (
                    f"MLP model for {model_info['symbol']} stock prediction. "
                    f"Architecture: {model_info['architecture']}. "
                    f"Features: {model_info['features_count']}. "
                    f"Val RMSE: {metrics.get('val_rmse', 'N/A')}"
                )
                try:
                    self.client.update_model_version(
                        name=model_name,
                        version=latest_version,
                        description=auto_description
                    )
                except:
                    pass
            
            # Get registration result
            result = {
                'success': True,
                'model_name': model_name,
                'version': latest_version,
                'stage': stage if stage else 'None',
                'run_id': run_id,
                'symbol': model_info['symbol'],
                'architecture': model_info['architecture'],
                'metrics': metrics,
                'timestamp': model_info['timestamp']
            }
            
            self.logger.info("="*70)
            self.logger.info("MODEL REGISTRATION COMPLETED")
            self.logger.info("="*70)
            self.logger.info(f"✓ Model: {model_name}")
            self.logger.info(f"✓ Version: {latest_version}")
            self.logger.info(f"✓ Stage: {stage if stage else 'None'}")
            self.logger.info(f"✓ Run ID: {run_id}")
            self.logger.info(f"✓ Symbol: {model_info['symbol']}")
            self.logger.info(f"✓ Architecture: {model_info['architecture']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model registration failed: {str(e)}")
            raise ModelRegistryError(f"Registration failed: {str(e)}")
    
    @log_function_call
    def get_model_versions(self, model_name):
        """
        Get all versions of a registered model.
        
        Parameters:
        -----------
        model_name : str
            Name of the registered model
            
        Returns:
        --------
        list
            List of model versions with details
        """
        if not self.mlflow_enabled:
            raise ModelRegistryError("MLflow not enabled")
        
        try:
            self.logger.info(f"Fetching versions for: {model_name}")
            
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            versions_info = []
            for mv in sorted(model_versions, key=lambda x: int(x.version), reverse=True):
                version_info = {
                    'version': mv.version,
                    'stage': mv.current_stage,
                    'status': mv.status,
                    'run_id': mv.run_id,
                    'creation_timestamp': mv.creation_timestamp,
                    'last_updated_timestamp': mv.last_updated_timestamp,
                    'description': mv.description
                }
                versions_info.append(version_info)
            
            self.logger.info(f"Found {len(versions_info)} versions")
            return versions_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model versions: {str(e)}")
            raise ModelRegistryError(f"Version fetch failed: {str(e)}")
    
    @log_function_call
    def transition_stage(self, model_name, version, stage):
        """
        Transition model version to a different stage.
        
        Parameters:
        -----------
        model_name : str
            Name of the registered model
        version : str or int
            Model version number
        stage : str
            Target stage: "Staging", "Production", or "Archived"
            
        Returns:
        --------
        dict
            Transition result
        """
        if not self.mlflow_enabled:
            raise ModelRegistryError("MLflow not enabled")
        
        valid_stages = ['Staging', 'Production', 'Archived']
        if stage not in valid_stages:
            raise ModelRegistryError(f"Invalid stage. Must be one of: {valid_stages}")
        
        try:
            self.logger.info(f"Transitioning {model_name} v{version} to {stage}")
            
            self.client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage=stage,
                archive_existing_versions=False
            )
            
            self.logger.info(f"✓ Successfully transitioned to {stage}")
            
            return {
                'success': True,
                'model_name': model_name,
                'version': version,
                'new_stage': stage
            }
            
        except Exception as e:
            self.logger.error(f"Stage transition failed: {str(e)}")
            raise ModelRegistryError(f"Transition failed: {str(e)}")
    
    @log_function_call
    def get_production_model(self, model_name):
        """
        Get the current production version of a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the registered model
            
        Returns:
        --------
        dict or None
            Production model info or None if no production model exists
        """
        if not self.mlflow_enabled:
            raise ModelRegistryError("MLflow not enabled")
        
        try:
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            for mv in model_versions:
                if mv.current_stage == 'Production':
                    self.logger.info(f"Production model: v{mv.version}")
                    return {
                        'version': mv.version,
                        'stage': mv.current_stage,
                        'run_id': mv.run_id,
                        'description': mv.description
                    }
            
            self.logger.info(f"No production model found for {model_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get production model: {str(e)}")
            raise ModelRegistryError(f"Production model fetch failed: {str(e)}")
    
    @log_function_call
    def run_registration_pipeline(self, symbol=None, model_path=None, stage=None):
        """
        Run complete model registration pipeline.
        
        Parameters:
        -----------
        symbol : str, optional
            Stock symbol (from config if None)
        model_path : str, optional
            Path to model file (auto-detected if None)
        stage : str, optional
            Target stage (from config if None)
            
        Returns:
        --------
        dict
            Pipeline execution results
        """
        try:
            self.logger.info("="*70)
            self.logger.info("STARTING MODEL REGISTRATION PIPELINE")
            self.logger.info("="*70)
            
            start_time = datetime.now()
            
            # Step 1: Load model information
            self.logger.info("\n[1/2] Loading model information...")
            model_info = self.load_model_info(
                model_path=model_path,
                symbol=symbol
            )
            
            # Step 2: Register model
            self.logger.info("\n[2/2] Registering model in MLflow Registry...")
            result = self.register_model(
                model_info=model_info,
                stage=stage
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result['duration_seconds'] = duration
            result['pipeline_success'] = True
            
            self.logger.info("="*70)
            self.logger.info("REGISTRATION PIPELINE COMPLETED")
            self.logger.info("="*70)
            self.logger.info(f"✓ Duration: {duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Registration pipeline failed: {str(e)}")
            return {
                'pipeline_success': False,
                'error': str(e)
            }


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main execution function"""
    try:
        # Initialize registry
        registry = ModelRegistry()
        
        # Get symbol from config
        symbol = registry.data_config.get('stock_symbol', 'AAPL')
        
        # Run registration pipeline
        result = registry.run_registration_pipeline(symbol=symbol)
        
        if result.get('pipeline_success'):
            print(f"\n✅ Model registration completed successfully!")
            print(f"   Model: {result['model_name']}")
            print(f"   Version: {result['version']}")
            print(f"   Stage: {result['stage']}")
            print(f"   Symbol: {result['symbol']}")
            print(f"   Architecture: {result['architecture']}")
            print(f"\n   View at: https://dagshub.com/{registry.mlflow_config.get('dagshub_username')}/{registry.mlflow_config.get('dagshub_repo')}")
        else:
            print(f"\n❌ Registration failed: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
