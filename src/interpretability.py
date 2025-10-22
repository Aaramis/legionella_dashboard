"""
Interpretability module using Integrated Gradients for model explanation
"""

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import multiprocessing as mp

from .utils import AnalysisLogger


class ESM2Wrapper(torch.nn.Module):
    """Wrapper for ESM2 model to work with Captum"""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return F.softmax(outputs.logits, dim=-1)


class IntegratedGradientsAnalyzer:
    """Analyzer for computing integrated gradients attributions"""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'auto',
        logger: Optional[AnalysisLogger] = None
    ):
        """
        Initialize analyzer

        Args:
            model: Trained model
            tokenizer: Model tokenizer
            device: Device to use
            logger: Logger instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger if logger else AnalysisLogger()

        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Extract base model if wrapped in DataParallel (Captum doesn't work well with DataParallel)
        if isinstance(self.model, torch.nn.DataParallel):
            self.logger.info("Extracting base model from DataParallel wrapper for interpretability")
            self.model = self.model.module

        self.model.to(self.device)
        self.model.eval()

        # Create wrapper
        self.wrapper = ESM2Wrapper(self.model, self.tokenizer)

        # Get embedding layer
        try:
            self.embedding_layer = self.model.esm.embeddings.word_embeddings
        except AttributeError:
            self.embedding_layer = self.model.embeddings.word_embeddings

        # Initialize integrated gradients
        self.lig = LayerIntegratedGradients(self.wrapper, self.embedding_layer)

    def compute_attributions(
        self,
        sequence: str,
        target_class: int = 1,
        n_steps: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compute integrated gradients attributions for a sequence

        Args:
            sequence: Protein sequence
            target_class: Target class for attribution (1=Effector, 0=Non-effector)
            n_steps: Number of steps for integrated gradients

        Returns:
            Tuple of (attributions, token_ids, tokens)
        """
        # Tokenize (vectorized padding/truncation)
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = inputs['input_ids'].to(self.device)

        # Baseline: full padding tokens (same shape)
        ref_input_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        # IMPORTANT: do NOT disable gradients here; Captum needs them
        attributions_emb = self.lig.attribute(
            inputs=input_ids,
            baselines=ref_input_ids,
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=8
        )

        # Sum over embedding dim -> shape (seq_len,)
        attributions = attributions_emb.sum(dim=-1).squeeze(0).detach().cpu().numpy()

        # Tokens
        token_ids = input_ids.detach().cpu().squeeze().numpy()
        tokens = [self.tokenizer.decode([t]) for t in token_ids]

        return attributions, token_ids, tokens

    def create_attribution_plot(
        self,
        sequence: str,
        protein_label: str,
        target_class: int = 1,
        max_length: Optional[int] = None
    ) -> go.Figure:
        """
        Create interactive Plotly bar-chart visualization of attributions (no heatmap).
        """
        # Compute attributions
        attributions, token_ids, tokens = self.compute_attributions(
            sequence, target_class
        )

        # Filter special tokens (CLS/SEP/PAD/EOS)
        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            getattr(self.tokenizer, "eos_token_id", None)
        }
        special_tokens.discard(None)

        valid_mask = np.array([tid not in special_tokens for tid in token_ids], dtype=bool)
        if not valid_mask.any():
            raise ValueError("No valid tokens found")

        attributions = attributions[valid_mask]
        tokens = [ (tokens[i].strip() or 'X') for i, keep in enumerate(valid_mask) if keep ]

        # Full sequence (no truncation) — max_length param is ignored by design
        n = len(tokens)
        positions = np.arange(n)

        self.logger.info(f"Displaying attribution bar chart for full sequence ({n} amino acids)")

        # Colors: vectorized (red for positive, blue for negative/zero)
        colors = np.where(attributions > 0, 'red', 'blue').tolist()

        # Build single-figure bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=positions,
                    y=attributions,
                    marker=dict(color=colors),
                    text=tokens,
                    hovertemplate='Amino Acid: %{text}<br>Position: %{x}<br>Attribution: %{y:.4f}<extra></extra>',
                    showlegend=False
                )
            ]
        )

        # Ligne horizontale à 0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        # Largeur dynamique (cap à 5000 px)
        dynamic_width = max(1200, min(n * 8, 5000))

        # Ticks intelligents pour longues séquences (≈100 ticks max)
        if n > 100:
            tick_step = max(1, n // 100)
            tick_positions = positions[::tick_step]
            tick_labels = [str(i) for i in tick_positions]
            fig.update_xaxes(
                title_text="Position in Sequence",
                tickmode='array',
                tickvals=tick_positions,
                ticktext=tick_labels,
                tickangle=-90
            )
        else:
            fig.update_xaxes(
                title_text="Position in Sequence",
                tickmode='array',
                tickvals=positions,
                ticktext=[str(i) for i in positions],
                tickangle=-90
            )

        fig.update_yaxes(title_text="Attribution Score")

        fig.update_layout(
            title=f'Integrated Gradients Attribution (Bars Only) - {protein_label} ({n} aa)',
            height=550,
            width=dynamic_width,
            hovermode='closest',
            dragmode='pan',
            bargap=0.05
        )

        return fig


    def create_attribution_gallery(
        self,
        sequences: List[str],
        labels: List[str],
        target_class: int = 1,
        max_proteins: int = 10,
        max_length: Optional[int] = None
    ) -> List[go.Figure]:
        """
        Create gallery of attribution plots

        Args:
            sequences: List of protein sequences
            labels: List of protein labels
            target_class: Target class
            max_proteins: Maximum number of proteins to process
            max_length: Maximum sequence length to display (None = full sequence)

        Returns:
            List of Plotly figures
        """
        figures = []

        self.logger.info(f"Creating attribution gallery for {min(len(sequences), max_proteins)} proteins...")

        for i, (seq, label) in enumerate(zip(sequences[:max_proteins], labels[:max_proteins])):
            try:
                self.logger.info(f"Processing protein {i+1}/{min(len(sequences), max_proteins)}: {label}")
                fig = self.create_attribution_plot(
                    sequence=seq,
                    protein_label=label,
                    target_class=target_class,
                    max_length=max_length  # Pass through, default None for full sequence
                )
                figures.append(fig)
            except Exception as e:
                self.logger.error(f"Error processing {label}: {str(e)}")
                continue

        return figures


def _compute_attributions_worker(
    gpu_id: int,
    model_path: str,
    protein_data: List[Tuple[str, str]],
    max_length: Optional[int] = None
) -> Dict[str, go.Figure]:
    """
    Worker function to compute attributions on a specific GPU

    Args:
        gpu_id: GPU device ID
        model_path: Path to model
        protein_data: List of (label, sequence) tuples
        max_length: Maximum sequence length (None = show full sequence)

    Returns:
        Dictionary of attribution figures
    """
    # Set GPU for this worker
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    # Load model and tokenizer on this GPU
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Create analyzer
    logger = AnalysisLogger()
    logger.info(f"Worker on GPU {gpu_id}: Processing {len(protein_data)} proteins")

    analyzer = IntegratedGradientsAnalyzer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        logger=logger
    )

    # Compute attributions
    attribution_figs = {}
    for label, sequence in protein_data:
        try:
            fig = analyzer.create_attribution_plot(
                sequence=sequence,
                protein_label=label,
                target_class=1,
                max_length=max_length  # None for full sequence
            )
            attribution_figs[label] = fig
        except Exception as e:
            logger.error(f"GPU {gpu_id} - Error for {label}: {str(e)}")
            continue

    logger.info(f"GPU {gpu_id}: Completed {len(attribution_figs)}/{len(protein_data)} proteins")
    return attribution_figs


def compute_attributions_for_dataset(
    model,
    tokenizer,
    predictions_df: pd.DataFrame,
    output_dir: str = "data/processed",
    protein_ids: Optional[List[str]] = None,
    device: str = 'auto',
    use_multi_gpu: bool = True,
    model_path: Optional[str] = None
) -> Dict[str, go.Figure]:
    """
    Compute attributions for proteins in dataset (with multi-GPU support)

    Args:
        model: Trained model (not used in multi-GPU mode, we reload from path)
        tokenizer: Model tokenizer
        predictions_df: Predictions DataFrame
        output_dir: Output directory
        protein_ids: Specific protein IDs to analyze (if None, analyze all)
        device: Device to use
        use_multi_gpu: Whether to use multi-GPU parallel processing
        model_path: Path to model (required for multi-GPU mode)

    Returns:
        Dictionary mapping protein IDs to figures
    """
    logger = AnalysisLogger(log_dir=f"{output_dir}/logs")
    logger.info("Starting integrated gradients analysis...")

    # Select proteins to analyze
    if protein_ids:
        df_subset = predictions_df[predictions_df['label'].isin(protein_ids)]
    else:
        df_subset = predictions_df

    if len(df_subset) == 0:
        logger.warning("No proteins to analyze")
        return {}

    # Check for sequence column
    if 'sequence' in df_subset.columns:
        sequence_col = 'sequence'
    elif 'sequence_x' in df_subset.columns:
        sequence_col = 'sequence_x'
        logger.warning("Found 'sequence_x' instead of 'sequence' - using it")
    elif 'sequence_y' in df_subset.columns:
        sequence_col = 'sequence_y'
        logger.warning("Found 'sequence_y' instead of 'sequence' - using it")
    else:
        logger.error("No sequence column found in predictions DataFrame")
        return {}

    logger.info(f"Analyzing {len(df_subset)} proteins")

    # Check for multi-GPU
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if use_multi_gpu and num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs for parallel attribution computation")

        # Get model path
        if model_path is None:
            # Try to extract from model config
            if hasattr(model, 'module'):
                model_config = model.module.config
            else:
                model_config = model.config
            model_path = getattr(model_config, '_name_or_path', 'model/')

        logger.info(f"Model path: {model_path}")

        # Prepare protein data
        protein_data = list(zip(
            df_subset['label'].tolist(),
            df_subset[sequence_col].tolist()
        ))

        # Split data into chunks for each GPU
        chunk_size = len(protein_data) // num_gpus
        chunks = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(protein_data)
            chunks.append(protein_data[start_idx:end_idx])

        logger.info(f"Split {len(protein_data)} proteins into {num_gpus} chunks")
        for i, chunk in enumerate(chunks):
            logger.info(f"  GPU {i}: {len(chunk)} proteins")

        # Use multiprocessing to parallelize across GPUs
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        # Prepare arguments for each worker: (gpu_id, model_path, protein_data, max_length)
        # max_length=None to show full sequences
        worker_args = [(i, model_path, chunk, None) for i, chunk in enumerate(chunks)]

        with mp.Pool(processes=num_gpus) as pool:
            results = pool.starmap(_compute_attributions_worker, worker_args)

        # Combine results from all GPUs
        attribution_figs = {}
        for result in results:
            attribution_figs.update(result)

        logger.info(f"Multi-GPU: Computed attributions for {len(attribution_figs)} proteins")

    else:
        # Single GPU/CPU mode
        if num_gpus > 1:
            logger.info("Multi-GPU disabled, using single GPU")

        # Initialize analyzer (single GPU)
        analyzer = IntegratedGradientsAnalyzer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            logger=logger
        )

        # Compute attributions sequentially
        attribution_figs = {}
        for idx, row in df_subset.iterrows():
            label = row['label']
            sequence = row[sequence_col]

            try:
                logger.info(f"Computing attributions for {label}")
                fig = analyzer.create_attribution_plot(
                    sequence=sequence,
                    protein_label=label,
                    target_class=1,
                    max_length=None  # Show full sequence
                )
                attribution_figs[label] = fig

            except Exception as e:
                logger.error(f"Error for {label}: {str(e)}")
                continue

        logger.info(f"Single GPU: Computed attributions for {len(attribution_figs)} proteins")

    return attribution_figs
