# Legionella Protein Analysis System

Deep learning-based system for predicting effector proteins in *Legionella* and visualizing their characteristics through interactive dashboards.

## Overview

This system uses a fine-tuned ESM2 transformer model to:
- Predict whether proteins are effectors or non-effectors
- Extract embedding representations of proteins
- Capture attention patterns showing important sequence regions
- Generate interactive visualizations for exploration
- Create comprehensive HTML reports

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis

Simply run the main script:

```bash
python run_analysis.py
```

This will:
1. Load your trained model from `model/`
2. Analyze proteins from `data/master.csv`
3. Generate predictions, embeddings, and attention maps
4. Create t-SNE visualizations
5. Generate an interactive HTML report in `outputs/`

### 3. View Results


Open the generated report in your web browser:

```bash
# The report will be at:
outputs/legionella_analysis_report.html
```

## Configuration

All settings can be customized in `config.yaml`:

```yaml
# Example: Highlight specific proteins
visualization:
  highlight_proteins:
    - lpg1484
    - lpg0502
    - lpg2206

# Example: Change t-SNE parameters
analysis:
  tsne_perplexity: 30
  random_seed: 42
```

## Project Structure

```
7_DL_Pasteur/
├── src/                          # Source code modules
│   ├── model_inference.py        # Model inference & feature extraction
│   ├── visualization.py          # t-SNE and interactive plots
│   ├── dashboard.py              # HTML report generation
│   └── utils.py                  # Utilities (logging, data management)
│
├── data/                         # Data directory
│   ├── master.csv                # Input protein data
│   └── processed/                # Generated results
│       ├── embeddings.npz        # Protein embeddings
│       ├── attentions.npz        # Attention weights
│       ├── predictions.csv       # Model predictions
│       ├── tsne_results.npz      # t-SNE coordinates
│       └── logs/                 # Execution logs
│
├── model/                        # Trained ESM2 model
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
│
├── outputs/                      # Generated reports
│   ├── legionella_analysis_report.html    # Main report
│   ├── tsne_2d_standalone.html            # 2D t-SNE plot
│   └── tsne_3d_standalone.html            # 3D t-SNE plot
│
├── config.yaml                   # Configuration file
├── run_analysis.py               # Main script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Input Data Format

The input CSV file (`data/master.csv`) should have these columns:

| Column | Description | Required |
|--------|-------------|----------|
| `label` | Protein ID (e.g., lpg0001) | Yes |
| `sequence` | Amino acid sequence | Yes |
| `localisation` | Subcellular localization | Optional |
| `localisation_score` | Localization confidence | Optional |
| `is_effector` | True/False ground truth | Optional |
| `is_housekeeping` | True/False housekeeping gene | Optional |

## Output Files

### 1. Predictions (`data/processed/predictions.csv`)
Contains predictions for all proteins:
- `label`: Protein ID
- `prediction`: Effector or Non_effector
- `confidence`: Prediction confidence (0-1)
- `probability_effector`: Probability of being an effector
- `probability_non_effector`: Probability of being a non-effector

### 2. Embeddings (`data/processed/embeddings.npz`)
Numpy archive with:
- `embeddings`: Protein embeddings (N × 1280 for ESM2)
- `labels`: Protein IDs
- `sequences`: Amino acid sequences
- Additional metadata

### 3. Attention Weights (`data/processed/attentions.npz`)
Numpy archive with:
- `attentions`: Attention matrices (N × heads × seq_len × seq_len)
- `labels`: Protein IDs

### 4. t-SNE Results (`data/processed/tsne_results.npz`)
Numpy archive with:
- `tsne_2d`: 2D coordinates (N × 2)
- `tsne_3d`: 3D coordinates (N × 3)
- `labels`: Protein IDs

### 5. HTML Report (`outputs/legionella_analysis_report.html`)
Interactive dashboard with:
- Overview statistics
- Distribution plots
- Interactive 2D and 3D t-SNE visualizations
- Attention maps for highlighted proteins
- Filterable results table

## Advanced Usage

### Run Specific Steps Only

```bash
# Skip model inference (use existing results)
python run_analysis.py --skip-inference

# Skip t-SNE computation
python run_analysis.py --skip-tsne

# Skip dashboard generation
python run_analysis.py --skip-dashboard
```

### Use Custom Configuration

```bash
python run_analysis.py --config my_custom_config.yaml
```

### Highlight Specific Proteins

Edit `config.yaml`:

```yaml
visualization:
  highlight_proteins:
    - lpg1484
    - lpg0502
    - lpg2206
```

These proteins will be marked with gold stars in the visualizations and will have their attention maps displayed in the report.

## Understanding the Visualizations

### t-SNE Plots
- **2D/3D t-SNE**: Dimensionality reduction of protein embeddings
- **Colors**: Represent subcellular localization (or prediction category)
- **Hover**: Shows detailed information for each protein
- **Interaction**: Click legend to filter, drag to rotate (3D), zoom with scroll

### Attention Maps
- **Heatmap**: Shows which amino acid positions the model focuses on
- **Bar plot**: Average attention per sequence position
- **Higher values**: More important for the model's prediction
- **Use case**: Identify key functional regions in the sequence

## Model Information

- **Base model**: ESM2 (Facebook AI protein language model)
- **Architecture**: Transformer with classification head
- **Training**: Fine-tuned on Legionella effector/non-effector dataset
- **Embedding dimension**: 1280 (ESM2 hidden size)

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `config.yaml`:
```yaml
model:
  batch_size: 8  # or smaller
```

### t-SNE Taking Too Long
Reduce perplexity or use fewer samples:
```yaml
analysis:
  tsne_perplexity: 15  # lower value
```

### Missing Attention Maps
Ensure attention extraction is enabled:
```yaml
analysis:
  extract_attentions: true
```

## System Requirements

- **Python**: 3.8+
- **GPU**: Recommended for large datasets (optional)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~2GB for dependencies + model size

## Citation

If you use this system in your research, please cite:

```
[Add appropriate citation for the model and pipeline]
```

## Support

For questions or issues:
1. Check the logs in `data/processed/logs/`
2. Review this README
3. Contact the development team

## License

[Add license information]

---

**Version**: 1.0.0
**Last Updated**: 2025
**Developed by**: IRD-UMMISCO-EIDOS Team
