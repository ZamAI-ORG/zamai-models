# 🧪 Testing Scripts

This directory contains scripts for testing and validating ZamAI models.

## Scripts

### `test_models.py`

Comprehensive testing suite for all your HF models.

- Tests all models under tasal9/
- Pashto conversation testing
- Performance benchmarking
- Saves results to JSON

**Usage:**

```bash
python test_models.py
```

### `check_hf_models.py`

Quick status check for Hugging Face models.

- Model availability
- Download statistics
- API status

### `quick_model_test.py`

Fast testing for development.

- Single model testing
- Quick response validation

### `simple_test.py`

Basic model functionality test.

- Simple input/output validation
- Error detection

## Test Results

Results are saved to `../../data/processed/` as JSON files for analysis.
