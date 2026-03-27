# MLflow & DagsHub Setup Guide

This guide explains how to set up MLflow experiment tracking with DagsHub for ForesightX.

## What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle, including:
- **Tracking**: Log parameters, metrics, and artifacts
- **Projects**: Package code in reusable format
- **Models**: Deploy models to various platforms
- **Registry**: Centralize model storage

## What is DagsHub?

DagsHub is a platform for data scientists that provides:
- **Hosted MLflow**: Free MLflow tracking server
- **Git-based**: Works seamlessly with GitHub
- **Data versioning**: Built-in DVC support
- **Collaboration**: Share experiments with team

## Setup Steps

### 1. Create DagsHub Account

1. Go to [DagsHub](https://dagshub.com/)
2. Sign up or log in with GitHub
3. Create a new repository or connect existing GitHub repo

### 2. Get Your DagsHub Token

1. Go to [DagsHub Settings](https://dagshub.com/user/settings/tokens)
2. Click "Create Token"
3. Give it a name (e.g., "ForesightX MLflow")
4. Copy the generated token (you won't see it again!)

### 3. Set Environment Variable

#### Linux/Mac:

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export DAGSHUB_TOKEN="your-token-here"
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

#### Windows (PowerShell):

```powershell
$env:DAGSHUB_TOKEN="your-token-here"
```

For permanent:
```powershell
[System.Environment]::SetEnvironmentVariable('DAGSHUB_TOKEN', 'your-token-here', 'User')
```

#### Using .env file (Recommended):

Create `.env` file in project root:

```bash
DAGSHUB_TOKEN=your-token-here
```

**Important**: Add `.env` to `.gitignore` to keep token private!

### 4. Update params.yaml

Edit `params.yaml` with your DagsHub details:

```yaml
mlflow:
  enabled: true
  dagshub_username: 'your-username'  # Your DagsHub username
  dagshub_repo: 'ForesightX'  # Your repo name
  experiment_name: 'MLP_Stock_Prediction'
```

### 5. Install Dependencies

```bash
pip install mlflow dagshub
```

Or:

```bash
pip install -r requirements.txt
```

## Usage

### Running Evaluation with MLflow Tracking

```bash
python src/model/evaluate_model.py
```

This will:
1. Load the trained model
2. Evaluate on test set
3. Log metrics to MLflow
4. Upload artifacts to DagsHub

### Viewing Results

1. Go to your DagsHub repository
2. Click on "Experiments" tab
3. View all tracked runs, metrics, and artifacts

Or visit directly:
```
https://dagshub.com/your-username/ForesightX
```

## What Gets Logged?

### Parameters
- Stock symbol
- Model architecture
- Hyperparameters (learning rate, activation, etc.)
- Feature count

### Metrics
- **Validation**: RMSE, MAE, iterations
- **Test**: RMSE, MAE, MAPE
- **Direction Accuracy**: Overall, up/down movements
- **Sample counts**: Test size, up/down movements

### Artifacts
- Trained model (.pkl)
- Scaler (.pkl)
- Metadata (JSON)
- Evaluation results (JSON)
- Predictions (CSV)

## Troubleshooting

### "DAGSHUB_TOKEN not found"

**Solution**: Set the environment variable properly and restart your terminal.

```bash
echo $DAGSHUB_TOKEN  # Should show your token
```

### "MLflow tracking failed"

**Solution**: The evaluation will continue with local logging. Check:
1. Token is correct
2. DagsHub username/repo in `params.yaml` match your account
3. Internet connection is working

### "Cannot connect to DagsHub"

**Solution**: 
1. Verify DagsHub is accessible: `https://dagshub.com`
2. Check firewall/proxy settings
3. Try manual connection test:

```python
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/username/repo.mlflow")
```

## Local-Only Mode

To disable MLflow tracking and use local storage only:

```yaml
# params.yaml
mlflow:
  enabled: false
```

Results will still be saved locally in `results/` directory.

## Best Practices

1. **Never commit tokens**: Keep `.env` in `.gitignore`
2. **Use descriptive run names**: Helps identify experiments
3. **Log relevant metrics**: Focus on metrics that matter
4. **Tag runs**: Use tags for easy filtering
5. **Compare runs**: Use DagsHub UI to compare experiments

## Example Workflow

```bash
# 1. Train model
python src/model/train_model.py

# 2. Evaluate with MLflow tracking
python src/model/evaluate_model.py

# 3. View results on DagsHub
# Go to: https://dagshub.com/your-username/ForesightX/experiments

# 4. Compare multiple runs
# Train with different hyperparameters and compare in DagsHub UI
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DagsHub Documentation](https://dagshub.com/docs/)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [DagsHub + MLflow Tutorial](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
