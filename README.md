# Wash Trading Detector

A tool for generating and analyzing synthetic trading data to detect wash trading patterns.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

Run the main script:
```bash
python -m washdetector.main
```

## Development

The project uses:
- Pydantic for data validation
- NumPy for numerical operations
- Pytest for testing

### Directory Structure

```
wash-detector/
├── washdetector/
│   ├── generator/     # Data generation
│   ├── visualization/ # Visualization tools
│   └── utils/        # Utility functions
```
