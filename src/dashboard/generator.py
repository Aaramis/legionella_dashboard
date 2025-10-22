"""
Main dashboard generator class
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import AnalysisLogger
from ..visualization import AttentionVisualizer
from .charts import (
    create_confusion_matrix,
    create_performance_metrics,
    create_prediction_comparison,
    create_statistics_overview,
    create_interactive_table,
    create_correlation_heatmap
)
from .statistics import generate_summary_stats
from .html_builder import build_complete_html


class DashboardGenerator:
    """Generate comprehensive HTML dashboard"""

    def __init__(self, logger: Optional[AnalysisLogger] = None):
        """Initialize dashboard generator"""
        self.logger = logger if logger else AnalysisLogger()

    def create_attention_gallery(
        self,
        attentions: np.ndarray,
        labels: np.ndarray,
        sequences: np.ndarray,
        attention_masks: np.ndarray,
        protein_ids: List[str],
        max_proteins: int = 6
    ) -> List[go.Figure]:
        """
        Create gallery of attention maps for selected proteins

        Args:
            attentions: All attention weights
            labels: All protein labels
            sequences: All protein sequences
            attention_masks: Attention masks
            protein_ids: Specific proteins to visualize
            max_proteins: Maximum number of proteins to show

        Returns:
            List of Plotly figures
        """
        att_viz = AttentionVisualizer(logger=self.logger)
        figures = []

        # Find indices of requested proteins
        for i, protein_id in enumerate(protein_ids[:max_proteins]):
            try:
                idx = np.where(labels == protein_id)[0][0]

                # Get data for this protein
                attn = attentions[idx]  # (num_heads, seq_len, seq_len)
                seq = sequences[idx]
                mask = attention_masks[idx]

                # Get actual sequence length (non-padded)
                seq_len = int(mask.sum())

                # Truncate to actual length
                attn = attn[:, :seq_len, :seq_len]
                seq = seq[:seq_len]

                # Create attention summary
                fig = att_viz.create_attention_summary(
                    attention_weights=attn,
                    sequence=seq,
                    protein_label=protein_id,
                    max_seq_len=100
                )

                figures.append(fig)

            except IndexError:
                self.logger.warning(f"Protein {protein_id} not found in data")
                continue

        return figures

    def generate_html_report(
        self,
        predictions_df: pd.DataFrame,
        fig_tsne_2d: go.Figure,
        fig_tsne_3d: go.Figure,
        attentions_data: Optional[Dict] = None,
        highlight_proteins: Optional[List[str]] = None,
        output_path: str = "outputs/report.html",
        attribution_figs: Optional[Dict[str, go.Figure]] = None
    ) -> Path:
        """
        Generate complete HTML report

        Args:
            predictions_df: Predictions DataFrame
            fig_tsne_2d: 2D t-SNE figure
            fig_tsne_3d: 3D t-SNE figure
            attentions_data: Dictionary with attention data
            highlight_proteins: Proteins to highlight in attention maps
            output_path: Output file path
            attribution_figs: Dictionary of attribution figures by protein ID

        Returns:
            Path to generated report
        """
        self.logger.info("Generating HTML report...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create overview statistics
        fig_stats = create_statistics_overview(predictions_df)

        # Create performance evaluation visualizations
        fig_confusion = create_confusion_matrix(predictions_df)
        fig_metrics = create_performance_metrics(predictions_df)
        fig_comparison = create_prediction_comparison(predictions_df)

        # Create attention gallery if data available
        attention_figs = []
        if attentions_data and highlight_proteins:
            self.logger.info("Creating attention visualizations...")
            attention_figs = self.create_attention_gallery(
                attentions=attentions_data['attentions'],
                labels=attentions_data['labels'],
                sequences=attentions_data.get('sequences', attentions_data['labels']),
                attention_masks=attentions_data['attention_masks'],
                protein_ids=highlight_proteins,
                max_proteins=6
            )

        # Create table
        table_html = create_interactive_table(predictions_df)

        # Generate summary statistics
        summary_stats = generate_summary_stats(predictions_df)

        # Build HTML
        html_content = build_complete_html(
            summary_stats=summary_stats,
            fig_stats=fig_stats,
            fig_tsne_2d=fig_tsne_2d,
            fig_tsne_3d=fig_tsne_3d,
            table_html=table_html,
            attention_figs=attention_figs,
            attribution_figs=attribution_figs,
            fig_confusion=fig_confusion,
            fig_metrics=fig_metrics,
            fig_comparison=fig_comparison
        )

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"Report saved: {output_path}")
        return output_path
