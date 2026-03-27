# ForesightX - Local Storage & DVC Setup
# =======================================

## Current Configuration: Local Storage Only

All data is stored locally in the `/data` directory structure:

```
data/
├── raw/              # Raw stock data from yfinance
├── interim/          # Intermediate processing steps
└── processed/        # Final train/test splits
```

## DVC (Data Version Control)

DVC is configured for **local storage only** during development:
- Data versions tracked locally in `.dvc/cache`
- No cloud storage required
- Git tracks `.dvc` metadata files
- Actual data files excluded from git

### DVC Commands

```bash
# Initialize DVC (already done)
dvc init

# Track data files
dvc add data/raw/stock_data_raw_AAPL.csv
dvc add data/processed/train_data_AAPL.csv
dvc add data/processed/test_data_AAPL.csv

# Commit DVC metadata
git add data/raw/.gitignore data/raw/stock_data_raw_AAPL.csv.dvc
git commit -m "Track raw data with DVC"

# Push/pull data (currently local only)
dvc push  # Pushes to local cache
dvc pull  # Pulls from local cache
```

## S3 Configuration (For Production Later)

When ready to use S3 remote storage:

### 1. Create `.env` file with AWS credentials:

```bash
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=foresightx-mlops
```

### 2. Update `params.yaml`:

```yaml
dvc:
  remote_storage: true

s3:
  enabled: true
```

### 3. Configure DVC remote:

```bash
# Add S3 as DVC remote
dvc remote add -d s3remote s3://foresightx-mlops/dvc-storage

# Set AWS credentials
dvc remote modify s3remote region us-east-1

# Push data to S3
dvc push
```

### 4. Benefits of S3 Remote:
- Team collaboration (shared data access)
- Data backup and disaster recovery
- CI/CD integration
- Large dataset handling
- Cost-effective storage for ML data

## Current Workflow (Local Only)

1. **Run Data Ingestion**:
   ```bash
   python src/data/make_dataset.py
   # Data saved to: data/raw/stock_data_raw_AAPL.csv
   ```

2. **Run Preprocessing**:
   ```bash
   python src/data/preprocess.py
   # Data saved to: data/processed/train_data_AAPL.csv
   #                 data/processed/test_data_AAPL.csv
   ```

3. **Track with DVC** (optional):
   ```bash
   dvc add data/raw/*.csv
   dvc add data/processed/*.csv
   git add data/**/*.dvc data/**/.gitignore
   git commit -m "Track datasets with DVC"
   ```

## Storage Location Summary

| Data Type | Location | Tracking |
|-----------|----------|----------|
| Raw Data | `data/raw/` | DVC (local) |
| Processed Data | `data/processed/` | DVC (local) |
| Models | `models/` | DVC (local) |
| Logs | `logs/` | Git ignored |
| Metadata | `metadata/` | Git tracked |
| Code | `src/` | Git tracked |

## Note

- **Development**: Use local storage (current setup)
- **Production**: Enable S3 remote storage
- Both workflows supported by the codebase
- S3 initialization errors are handled gracefully
- Modules work seamlessly with or without S3
