"""
Visualization module for creating interactive t-SNE plots with Plotly
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import time

from .utils import AnalysisLogger, DataManager


class TSNEVisualizer:
    """Create t-SNE visualizations"""

    def __init__(self, logger: Optional[AnalysisLogger] = None):
        """
        Initialize visualizer

        Args:
            logger: Logger instance
        """
        self.logger = logger if logger else AnalysisLogger()

    def compute_tsne(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42,
        n_jobs: int = -1
    ) -> np.ndarray:
        """
        Compute t-SNE dimensionality reduction

        Args:
            embeddings: Input embeddings (N, D)
            n_components: Number of dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter
            random_state: Random seed
            n_jobs: Number of parallel jobs

        Returns:
            t-SNE coordinates (N, n_components)
        """
        self.logger.info(f"Computing {n_components}D t-SNE...")
        self.logger.info(f"Input shape: {embeddings.shape}")
        self.logger.info(f"Perplexity: {perplexity}")
        self.logger.info(f"Random state: {random_state}")
        self.logger.info(f"Metric: cosine")

        start_time = time.time()

        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # Adjust perplexity if necessary
        max_perplexity = embeddings.shape[0] - 1
        if perplexity >= max_perplexity:
            perplexity = max(5, max_perplexity // 2)
            self.logger.warning(f"Perplexity adjusted to {perplexity}")

        # Compute t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            metric='cosine',
            n_jobs=n_jobs,
            verbose=1
        )

        tsne_coords = tsne.fit_transform(embeddings_scaled)

        elapsed = time.time() - start_time
        self.logger.info(f"t-SNE completed in {elapsed:.2f}s")
        self.logger.info(f"Output shape: {tsne_coords.shape}")

        return tsne_coords

    def create_tsne_plot_2d(
        self,
        tsne_coords: np.ndarray,
        labels: np.ndarray,
        metadata: pd.DataFrame,
        color_by: str = 'localisation',
        highlight_proteins: Optional[List[str]] = None,
        title: str = "2D t-SNE Visualization"
    ) -> go.Figure:
        """
        Create interactive 2D t-SNE plot with Plotly

        Args:
            tsne_coords: t-SNE coordinates (N, 2)
            labels: Protein labels (N,)
            metadata: DataFrame with additional information
            color_by: Column to color by ('localisation', 'category', 'prediction')
            highlight_proteins: List of protein IDs to highlight
            title: Plot title

        Returns:
            Plotly figure
        """
        self.logger.info(f"Creating 2D t-SNE plot (color by: {color_by})")

        # Create DataFrame
        df = pd.DataFrame({
            'x': tsne_coords[:, 0],
            'y': tsne_coords[:, 1],
            'label': labels
        })

        # Merge with metadata
        df = df.merge(metadata, on='label', how='left')

        # Determine category for visualization
        if 'is_effector' in df.columns:
            df['category'] = df['is_effector'].map({
                True: 'Effector',
                False: 'Non-effector'
            })
            df['category'] = df['category'].fillna('Unknown')
        elif 'prediction' in df.columns:
            df['category'] = df['prediction']
        else:
            df['category'] = 'Unknown'

        # Add highlighted status
        if highlight_proteins:
            df['is_highlighted'] = df['label'].isin(highlight_proteins)
        else:
            df['is_highlighted'] = False

        # Create hover text
        df['hover_text'] = df.apply(
            lambda row: self._create_hover_text(row), axis=1
        )

        # Color palette
        color_column = color_by if color_by in df.columns else 'category'

        # Create figure
        fig = px.scatter(
            df[~df['is_highlighted']],  # Non-highlighted points first
            x='x',
            y='y',
            color=color_column,
            hover_data={'hover_text': True, 'x': False, 'y': False, color_column: False},
            title=title,
            labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
            opacity=0.6,
            template='plotly_white'
        )

        # Add highlighted proteins on top
        if highlight_proteins and df['is_highlighted'].any():
            df_highlighted = df[df['is_highlighted']]

            fig.add_trace(go.Scatter(
                x=df_highlighted['x'],
                y=df_highlighted['y'],
                mode='markers',
                marker=dict(
                    size=12,
                    color='gold',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                text=df_highlighted['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name='Highlighted',
                showlegend=True
            ))

        # Update layout
        fig.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>'
        )

        fig.update_layout(
            width=1000,
            height=700,
            hovermode='closest',
            font=dict(size=12)
        )

        return fig

    def create_tsne_plot_3d(
        self,
        tsne_coords: np.ndarray,
        labels: np.ndarray,
        metadata: pd.DataFrame,
        color_by: str = 'localisation',
        highlight_proteins: Optional[List[str]] = None,
        title: str = "3D t-SNE Visualization"
    ) -> go.Figure:
        """
        Create interactive 3D t-SNE plot with Plotly

        Args:
            tsne_coords: t-SNE coordinates (N, 3)
            labels: Protein labels (N,)
            metadata: DataFrame with additional information
            color_by: Column to color by
            highlight_proteins: List of protein IDs to highlight
            title: Plot title

        Returns:
            Plotly figure
        """
        self.logger.info(f"Creating 3D t-SNE plot (color by: {color_by})")

        # Create DataFrame
        df = pd.DataFrame({
            'x': tsne_coords[:, 0],
            'y': tsne_coords[:, 1],
            'z': tsne_coords[:, 2],
            'label': labels
        })

        # Merge with metadata
        df = df.merge(metadata, on='label', how='left')

        # Determine category
        if 'is_effector' in df.columns:
            df['category'] = df['is_effector'].map({
                True: 'Effector',
                False: 'Non-effector'
            })
            df['category'] = df['category'].fillna('Unknown')
        elif 'prediction' in df.columns:
            df['category'] = df['prediction']
        else:
            df['category'] = 'Unknown'

        # Add highlighted status
        if highlight_proteins:
            df['is_highlighted'] = df['label'].isin(highlight_proteins)
        else:
            df['is_highlighted'] = False

        # Create hover text
        df['hover_text'] = df.apply(
            lambda row: self._create_hover_text(row), axis=1
        )

        # Color column
        color_column = color_by if color_by in df.columns else 'category'

        # Create figure
        fig = px.scatter_3d(
            df[~df['is_highlighted']],
            x='x',
            y='y',
            z='z',
            color=color_column,
            hover_data={'hover_text': True, 'x': False, 'y': False, 'z': False, color_column: False},
            title=title,
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'z': 't-SNE 3'},
            opacity=0.6,
            template='plotly_white'
        )

        # Add highlighted proteins
        if highlight_proteins and df['is_highlighted'].any():
            df_highlighted = df[df['is_highlighted']]

            fig.add_trace(go.Scatter3d(
                x=df_highlighted['x'],
                y=df_highlighted['y'],
                z=df_highlighted['z'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='gold',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                text=df_highlighted['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name='Highlighted',
                showlegend=True
            ))

        # Update layout
        fig.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>'
        )

        fig.update_layout(
            width=1100,
            height=800,
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3'
            ),
            hovermode='closest',
            font=dict(size=12)
        )

        return fig

    def _create_hover_text(self, row: pd.Series) -> str:
        """Create formatted hover text for a data point"""
        parts = [f"<b>{row['label']}</b>"]

        # Show true label if available
        if 'is_effector' in row and pd.notna(row['is_effector']):
            true_label = 'Effector' if row['is_effector'] else 'Non-effector'
            parts.append(f"True Label: {true_label}")
        elif 'is_housekeeping' in row and pd.notna(row['is_housekeeping']):
            true_label = 'Non-effector (housekeeping)' if row['is_housekeeping'] else 'Unknown'
            parts.append(f"True Label: {true_label}")
        else:
            parts.append("True Label: Unknown (not in training set)")

        # Show prediction
        if 'prediction' in row and pd.notna(row['prediction']):
            parts.append(f"Prediction: {row['prediction']}")

        if 'confidence' in row and pd.notna(row['confidence']):
            parts.append(f"Confidence: {row['confidence']:.3f}")

        if 'localisation' in row and pd.notna(row['localisation']):
            parts.append(f"Localisation: {row['localisation']}")

        if 'localisation_score' in row and pd.notna(row['localisation_score']):
            parts.append(f"Loc. Score: {row['localisation_score']:.2f}")

        if 'sequence' in row and pd.notna(row['sequence']):
            seq = str(row['sequence'])
            if len(seq) > 50:
                seq = seq[:50] + '...'
            parts.append(f"Sequence: {seq}")

        return '<br>'.join(parts)


class AttentionVisualizer:
    """Visualize attention weights"""

    def __init__(self, logger: Optional[AnalysisLogger] = None):
        """Initialize attention visualizer"""
        self.logger = logger if logger else AnalysisLogger()

    def create_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        sequence: str,
        protein_label: str,
        head_idx: int = 0,
        max_seq_len: int = 100
    ) -> go.Figure:
        """
        Create attention heatmap for a single protein

        Args:
            attention_weights: Attention matrix (num_heads, seq_len, seq_len)
            sequence: Protein sequence
            protein_label: Protein ID
            head_idx: Which attention head to visualize
            max_seq_len: Maximum sequence length to display

        Returns:
            Plotly figure
        """
        # Get attention for specific head
        attn = attention_weights[head_idx]  # (seq_len, seq_len)

        # Truncate if too long
        if attn.shape[0] > max_seq_len:
            attn = attn[:max_seq_len, :max_seq_len]
            sequence = sequence[:max_seq_len]

        # Create sequence labels
        seq_labels = list(sequence)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attn,
            x=seq_labels,
            y=seq_labels,
            colorscale='Viridis',
            hovertemplate='From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Attention Map - {protein_label} (Head {head_idx})",
            xaxis_title="Position",
            yaxis_title="Position",
            width=800,
            height=800,
            font=dict(size=10)
        )

        return fig

    def create_attention_summary(
        self,
        attention_weights: np.ndarray,
        sequence: str,
        protein_label: str,
        max_seq_len: int = 100
    ) -> go.Figure:
        """
        Create averaged attention summary across all heads

        Args:
            attention_weights: Attention matrix (num_heads, seq_len, seq_len)
            sequence: Protein sequence
            protein_label: Protein ID
            max_seq_len: Maximum sequence length to display

        Returns:
            Plotly figure showing average attention per position
        """
        # Average across all heads and target positions
        # Shape: (num_heads, seq_len, seq_len) -> (seq_len,)
        avg_attention = attention_weights.mean(axis=0).mean(axis=1)

        # Truncate if necessary
        if len(avg_attention) > max_seq_len:
            avg_attention = avg_attention[:max_seq_len]
            sequence = sequence[:max_seq_len]

        # Create bar plot
        fig = go.Figure(data=go.Bar(
            x=list(range(len(avg_attention))),
            y=avg_attention,
            text=list(sequence),
            hovertemplate='Position: %{x}<br>Amino acid: %{text}<br>Avg attention: %{y:.3f}<extra></extra>',
            marker=dict(
                color=avg_attention,
                colorscale='Viridis',
                showscale=True
            )
        ))

        fig.update_layout(
            title=f"Average Attention per Position - {protein_label}",
            xaxis_title="Sequence Position",
            yaxis_title="Average Attention",
            width=1000,
            height=400,
            font=dict(size=12)
        )

        return fig


def run_tsne_analysis(
    embeddings_path: str,
    predictions_path: str,
    output_dir: str = "data/processed",
    perplexity: int = 30,
    random_state: int = 42,
    highlight_proteins: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Run complete t-SNE analysis

    Args:
        embeddings_path: Path to embeddings file
        predictions_path: Path to predictions CSV
        output_dir: Output directory
        perplexity: t-SNE perplexity
        random_state: Random seed
        highlight_proteins: Proteins to highlight

    Returns:
        Dictionary with t-SNE results and figures
    """
    logger = AnalysisLogger(log_dir=f"{output_dir}/logs")
    data_manager = DataManager(base_dir=output_dir)
    visualizer = TSNEVisualizer(logger=logger)

    logger.info("="*60)
    logger.info("STARTING t-SNE ANALYSIS")
    logger.info("="*60)

    # Load data
    logger.info(f"Loading embeddings from: {embeddings_path}")
    emb_data = data_manager.load_embeddings(embeddings_path.split('/')[-1])
    embeddings = emb_data['embeddings']
    labels = emb_data['labels']

    logger.info(f"Loading predictions from: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)

    # Compute t-SNE
    tsne_2d = visualizer.compute_tsne(
        embeddings,
        n_components=2,
        perplexity=perplexity,
        random_state=random_state
    )

    tsne_3d = visualizer.compute_tsne(
        embeddings,
        n_components=3,
        perplexity=perplexity,
        random_state=random_state
    )

    # Save t-SNE results
    tsne_file = data_manager.save_tsne_results(
        tsne_2d=tsne_2d,
        tsne_3d=tsne_3d,
        labels=labels,
        metadata={k: v for k, v in emb_data.items() if k not in ['embeddings', 'labels', 'sequences']},
        filename="tsne_results.npz"
    )
    logger.info(f"t-SNE results saved: {tsne_file}")

    # Create visualizations
    logger.info("Creating visualizations...")

    fig_2d = visualizer.create_tsne_plot_2d(
        tsne_coords=tsne_2d,
        labels=labels,
        metadata=predictions_df,
        color_by='localisation',
        highlight_proteins=highlight_proteins,
        title="2D t-SNE - Protein Embeddings"
    )

    fig_3d = visualizer.create_tsne_plot_3d(
        tsne_coords=tsne_3d,
        labels=labels,
        metadata=predictions_df,
        color_by='localisation',
        highlight_proteins=highlight_proteins,
        title="3D t-SNE - Protein Embeddings"
    )

    logger.info("="*60)
    logger.info("t-SNE ANALYSIS COMPLETED")
    logger.info("="*60)

    return {
        'tsne_2d': tsne_2d,
        'tsne_3d': tsne_3d,
        'fig_2d': fig_2d,
        'fig_3d': fig_3d,
        'labels': labels
    }
