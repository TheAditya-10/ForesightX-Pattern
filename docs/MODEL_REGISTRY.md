# Model Registry Module - Complete

## Overview

The Model Registry module provides production-ready model versioning and lifecycle management using MLflow Model Registry on DagsHub. This is the final source module of the ForesightX project.

## Features

### 🎯 Core Functionality
- **Model Loading**: Load trained models with metadata
- **Model Registration**: Register models in MLflow Registry
- **Stage Management**: Transition models between lifecycle stages
- **Version Control**: Track multiple versions of models
- **Metadata Tracking**: Store model info, metrics, and configs

### 📊 Model Lifecycle Stages

```
None → Staging → Production → Archived
```

- **None**: Newly registered, not yet validated
- **Staging**: Under testing and validation
- **Production**: Actively serving predictions
- **Archived**: Deprecated or superseded

## Usage

### Quick Start

```bash
# 1. Ensure DagsHub token is set
export DAGSHUB_TOKEN="your-token"

# 2. Register the latest trained model
python src/model/model_registry.py
```

### Programmatic Usage

```python
from src.model.model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# Load model info
model_info = registry.load_model_info(symbol='AAPL')

# Register model (stage from params.yaml)
result = registry.register_model(model_info)

# Or specify stage explicitly
result = registry.register_model(model_info, stage='Production')
```

### Advanced Operations

```python
# Get all versions of a model
versions = registry.get_model_versions('MLP_Stock_Predictor_AAPL')
for v in versions:
    print(f"Version {v['version']}: {v['stage']}")

# Transition to production
registry.transition_stage(
    model_name='MLP_Stock_Predictor_AAPL',
    version=3,
    stage='Production'
)

# Get current production model
prod_model = registry.get_production_model('MLP_Stock_Predictor_AAPL')
print(f"Production: v{prod_model['version']}")
```

## Configuration

### params.yaml

```yaml
model_registry:
  track_experiments: true
  save_artifacts: true
  compare_metrics: true
  auto_archive: true
  retention_days: 90
  default_stage: "Staging"  # Stage for newly registered models

mlflow:
  enabled: true
  dagshub_username: 'TheAditya-10'
  dagshub_repo: 'ForesightX'
  experiment_name: 'MLP_Stock_Prediction'
```

## Module Structure

```python
class ModelRegistry:
    
    def load_model_info(self, model_path=None, symbol=None):
        """
        Load model information from files.
        Auto-detects latest model if paths not provided.
        
        Returns:
            dict: Model info including paths, metadata, metrics
        """
    
    def register_model(self, model_info, model_name=None, 
                      stage=None, description=None, tags=None):
        """
        Register model in MLflow Model Registry.
        
        Args:
            model_info: From load_model_info()
            model_name: Registry name (default: "MLP_Stock_Predictor_{SYMBOL}")
            stage: Target stage (from config if None)
            description: Model description
            tags: Additional tags
            
        Returns:
            dict: Registration result with version info
        """
    
    def get_model_versions(self, model_name):
        """
        Get all versions of a registered model.
        
        Returns:
            list: Version details with stage, status, timestamps
        """
    
    def transition_stage(self, model_name, version, stage):
        """
        Transition model version to different stage.
        
        Args:
            model_name: Registered model name
            version: Version number
            stage: "Staging", "Production", or "Archived"
        """
    
    def get_production_model(self, model_name):
        """
        Get current production model version.
        
        Returns:
            dict: Production model info or None
        """
    
    def run_registration_pipeline(self, symbol=None, 
                                  model_path=None, stage=None):
        """
        Complete registration pipeline.
        Loads model info and registers in one call.
        """
```

## What Gets Registered?

### Model Artifacts
- **Model**: Serialized scikit-learn MLP model
- **Scaler**: StandardScaler for feature normalization
- **Metadata**: JSON with full model information

### Parameters Logged
- Stock symbol
- Model type and architecture
- Features count
- Configuration (activation, solver, learning rate, etc.)
- Training timestamp

### Metrics Logged
- Validation: RMSE, MAE, iterations, training time
- Test: RMSE, MAE, MAPE, direction accuracy (if available)

### Tags Set
- model_family: "MLP"
- task: "regression"
- target: "stock_returns"
- symbol: Stock ticker
- version_timestamp: Model creation time

## Example Output

```
======================================================================
STARTING MODEL REGISTRATION PIPELINE
======================================================================

[1/2] Loading model information...
Auto-detecting latest model files...
Using model: models/mlp_model_AAPL_20251220_034123.pkl
✓ Model: MLP
✓ Architecture: [128, 64, 32]
✓ Features: 117
✓ Val RMSE: 0.250214

[2/2] Registering model in MLflow Registry...
======================================================================
REGISTERING MODEL IN MLFLOW REGISTRY
======================================================================
Model name: MLP_Stock_Predictor_AAPL
Target stage: Staging

Logging model to MLflow...
Model logged successfully (Run ID: abc123def456)
Registered as version: 3
Transitioning to stage: Staging
✓ Model transitioned to Staging stage
✓ Description added

======================================================================
MODEL REGISTRATION COMPLETED
======================================================================
✓ Model: MLP_Stock_Predictor_AAPL
✓ Version: 3
✓ Stage: Staging
✓ Run ID: abc123def456
✓ Symbol: AAPL
✓ Architecture: [128, 64, 32]

======================================================================
REGISTRATION PIPELINE COMPLETED
======================================================================
✓ Duration: 5.23 seconds

✅ Model registration completed successfully!
   Model: MLP_Stock_Predictor_AAPL
   Version: 3
   Stage: Staging
   Symbol: AAPL
   Architecture: [128, 64, 32]

   View at: https://dagshub.com/TheAditya-10/ForesightX
```

## Model Lifecycle Workflow

### 1. Initial Registration (Development)

```bash
# Train model
python src/model/train_model.py

# Register with Staging stage (default)
python src/model/model_registry.py
```

**Result**: Model v1 in Staging stage

### 2. Evaluation & Validation

```bash
# Evaluate model
python src/model/evaluate_model.py

# Review metrics on DagsHub
# Check performance, compare with previous versions
```

### 3. Promotion to Production

```python
from src.model.model_registry import ModelRegistry

registry = ModelRegistry()

# Promote to production after validation
registry.transition_stage(
    model_name='MLP_Stock_Predictor_AAPL',
    version=1,
    stage='Production'
)
```

**Result**: Model v1 now serving predictions

### 4. New Model Development

```bash
# Train improved model
python src/model/train_model.py

# Register (automatically gets v2)
python src/model/model_registry.py
```

**Result**: Model v2 in Staging, v1 still in Production

### 5. A/B Testing & Comparison

```python
# Load both versions for comparison
prod_model = registry.get_production_model('MLP_Stock_Predictor_AAPL')  # v1
# Test new version v2 on validation set
# Compare metrics
```

### 6. Model Replacement

```python
# If v2 performs better, promote it
registry.transition_stage(
    model_name='MLP_Stock_Predictor_AAPL',
    version=2,
    stage='Production'
)

# Archive old version
registry.transition_stage(
    model_name='MLP_Stock_Predictor_AAPL',
    version=1,
    stage='Archived'
)
```

**Result**: Model v2 in Production, v1 Archived

## Viewing Registered Models

### DagsHub UI

1. Go to: `https://dagshub.com/TheAditya-10/ForesightX`
2. Click "Models" tab
3. View all registered models and versions
4. See stage, metrics, artifacts for each version

### Programmatic Access

```python
# List all versions
versions = registry.get_model_versions('MLP_Stock_Predictor_AAPL')

for v in versions:
    print(f"Version {v['version']}")
    print(f"  Stage: {v['stage']}")
    print(f"  Status: {v['status']}")
    print(f"  Created: {v['creation_timestamp']}")
    print(f"  Run ID: {v['run_id']}")
    print()
```

## Best Practices

### 1. Stage Progression
```
Development → Staging → Production → Archived
```
Always validate in Staging before promoting to Production.

### 2. Version Descriptions
Add meaningful descriptions:
```python
registry.register_model(
    model_info,
    description="Improved feature engineering with 20% better RMSE"
)
```

### 3. Tagging Strategy
Use tags for organization:
```python
registry.register_model(
    model_info,
    tags={
        'experiment': 'hyperparameter_tuning_v3',
        'dataset_version': '2025-12',
        'performance_tier': 'high'
    }
)
```

### 4. Retention Policy
- Keep Production models indefinitely
- Archive Staging models after 90 days
- Document reason for archiving

### 5. Model Comparison
Before promoting, compare:
- Validation metrics
- Test metrics
- Inference time
- Resource usage

## Error Handling

The module handles common issues gracefully:

### MLflow Not Enabled
```
WARNING: MLflow is disabled in configuration
Model registry will not be available
```
**Solution**: Set `mlflow.enabled: true` in params.yaml

### Missing Token
```
WARNING: DAGSHUB_TOKEN not found in environment variables
Model registry will not be available
```
**Solution**: Export DAGSHUB_TOKEN or add to .env

### Model Not Found
```
ERROR: No model files found for AAPL
```
**Solution**: Train a model first with train_model.py

### Invalid Stage
```
WARNING: Invalid stage 'Testing'. Using None.
Valid stages: None, Staging, Production, Archived
```
**Solution**: Use correct stage name

## Integration with Other Modules

```
┌─────────────────┐
│  train_model.py │──┐
└─────────────────┘  │
                     ↓
              [Model Files]
                     ↓
┌─────────────────┐  │
│evaluate_model.py│──┤
└─────────────────┘  │
                     ↓
              [Test Metrics]
                     ↓
┌─────────────────┐  │
│model_registry.py│←─┘
└─────────────────┘
        ↓
   [DagsHub Registry]
        ↓
   [Production API]
```

## Production Deployment

### Load Production Model

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("https://dagshub.com/TheAditya-10/ForesightX.mlflow")

# Load production model
model_uri = "models:/MLP_Stock_Predictor_AAPL/Production"
model = mlflow.sklearn.load_model(model_uri)

# Use for predictions
predictions = model.predict(scaled_features)
```

### API Integration

```python
# In your FastAPI/Flask app
import mlflow
from src.model.model_registry import ModelRegistry

registry = ModelRegistry()

# Get production model info
prod_info = registry.get_production_model('MLP_Stock_Predictor_AAPL')

# Load for serving
model = mlflow.sklearn.load_model(
    f"models:/MLP_Stock_Predictor_AAPL/{prod_info['version']}"
)

@app.post("/predict")
def predict(features: dict):
    # Scale and predict
    return {"prediction": model.predict(...)}
```

## Troubleshooting

### Issue: "Model already exists"
**Solution**: MLflow auto-increments versions. Your new model will be registered as the next version.

### Issue: "Cannot transition stage"
**Solution**: Ensure version exists and stage name is valid.

### Issue: "Connection timeout"
**Solution**: Check internet connection and DagsHub accessibility.

### Issue: "Permission denied"
**Solution**: Verify DAGSHUB_TOKEN is correct and has appropriate permissions.

## Complete Pipeline Example

```bash
# 1. Train model
python src/model/train_model.py
# Output: models/mlp_model_AAPL_20251220_034123.pkl

# 2. Evaluate model
python src/model/evaluate_model.py
# Output: results/evaluation_metrics_AAPL_20251220_045830.json
# Logs to MLflow: metrics, artifacts

# 3. Register model
python src/model/model_registry.py
# Output: MLP_Stock_Predictor_AAPL v1 in Staging stage

# 4. Review on DagsHub
# https://dagshub.com/TheAditya-10/ForesightX

# 5. Promote to production (if validated)
python -c "
from src.model.model_registry import ModelRegistry
registry = ModelRegistry()
registry.transition_stage('MLP_Stock_Predictor_AAPL', 1, 'Production')
"
```

## Module Status

✅ **Production Ready**
- Complete model lifecycle management
- MLflow/DagsHub integration
- Stage transitions
- Version tracking
- Comprehensive logging
- Error handling
- Documentation

This is the **final source module** of the ForesightX ML pipeline!

## Resources

- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [DagsHub Models](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
- [Model Lifecycle Management](https://mlflow.org/docs/latest/model-registry.html#concepts)
