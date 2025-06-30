# 🛠️ Utility Scripts

This directory contains utility scripts for setup, validation, and maintenance.

## Scripts

### `validate_setup.py`

Complete setup validation for ZamAI environment.

- File structure check
- Configuration validation
- HF token verification
- Environment readiness

**Usage:**

```bash
python validate_setup.py
```

### `basic_check.py`

Basic environment and dependency check.

- Python packages
- System resources
- File permissions

### `check_dataset_access.py`

Validate access to your HF datasets.

- Dataset availability
- Access permissions
- File integrity

### `model_manager.py`

Model management utilities.

- Model listing
- Metadata management
- Cleanup tools

### `quick_dataset_check*.py`

Quick dataset validation scripts.

- Fast dataset access check
- Basic structure validation
- Connection testing

## Usage

These utilities should be run before training or testing to ensure everything is properly configured.
