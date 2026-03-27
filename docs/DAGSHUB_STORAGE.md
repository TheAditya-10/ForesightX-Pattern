# DagsHub Storage Setup Guide
# ===========================

## Quick Setup (3 Steps)

### 1. Get Your DagsHub Token
Visit: https://dagshub.com/user/settings/tokens
- Click "New Token"
- Name it "ForesightX"
- Copy the token

### 2. Set Environment Variables
```bash
# Linux/Mac
export DAGSHUB_USERNAME="TheAditya-10"
export DAGSHUB_REPO="ForesightX"
export DAGSHUB_TOKEN="your_token_here"

# Or add to .env file
echo 'DAGSHUB_USERNAME=TheAditya-10' >> .env
echo 'DAGSHUB_REPO=ForesightX' >> .env
echo 'DAGSHUB_TOKEN=your_token_here' >> .env
```

### 3. Configure DVC Password
```bash
# Use your DAGSHUB_TOKEN
dvc remote modify dagshub --local password $DAGSHUB_TOKEN
```

## Test Connection

```bash
# Test DagsHub service
python src/services/dagshub_service.py

# Test DVC remote
dvc push
```

## Usage

### Push to DagsHub
```bash
# After running pipeline
dvc push

# This uploads:
# - data/raw/
# - data/processed/
# - data/features/
# - models/
# - results/
```

### Pull from DagsHub
```bash
# On another machine
git clone https://github.com/TheAditya-10/ForesightX.git
cd ForesightX
dvc pull  # Downloads all data/models
```

### View on Web
- Data: https://dagshub.com/TheAditya-10/ForesightX/src/main
- Models: https://dagshub.com/TheAditya-10/ForesightX (Models tab)
- Experiments: https://dagshub.com/TheAditya-10/ForesightX/experiments

## Storage Limits

**Free Tier:**
- Storage: 10 GB
- Bandwidth: Unlimited
- Repositories: Unlimited
- Perfect for solo developers!

**Paid Tier ($8/month):**
- Storage: 100 GB
- Everything else same

## Benefits vs S3

✅ **No AWS setup** - No IAM, keys, buckets
✅ **Free 10GB** - No credit card required
✅ **Web UI** - Browse data/models visually
✅ **Git-like** - Familiar workflow
✅ **MLflow integrated** - Experiments + Data in one place
✅ **No billing surprises** - Fixed pricing

## Troubleshooting

**Problem: dvc push fails with 401**
```bash
# Re-check token
echo $DAGSHUB_TOKEN

# Re-configure
dvc remote modify dagshub --local password $DAGSHUB_TOKEN
```

**Problem: Connection timeout**
```bash
# Check internet connection
ping dagshub.com

# Try with verbose
dvc push -v
```

**Problem: Token not found**
```bash
# Make sure .env is loaded
source .env

# Or export manually
export DAGSHUB_TOKEN="your_token"
```

## DagsHub vs S3 vs Local

| Feature | DagsHub | S3 | Local Only |
|---------|---------|-----|------------|
| Setup time | 2 min | 30 min | 0 min |
| Free storage | 10 GB | 5 GB (12mo) | Unlimited |
| Cost after free | $8/100GB | ~$2/100GB | $0 |
| Team sharing | Easy | Medium | Hard |
| Web UI | Yes | Basic | No |
| MLflow | Integrated | Separate | Separate |
| Best for | Solo/Small teams | Large scale | Development |

## Next Steps

After setup:
1. Run `dvc push` after each pipeline run
2. Commit `.dvc/config` to git (local config stays private)
3. Share repo - teammates can `dvc pull` to get data
4. Monitor storage at https://dagshub.com/TheAditya-10/ForesightX/settings

## Security Notes

⚠️ **Never commit `.dvc/config.local`** - Contains your token
✅ **Safe to commit `.dvc/config`** - Only has remote URL
✅ **Use `.env` file** - Keep tokens in environment variables
