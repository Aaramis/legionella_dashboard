"""
Dashboard generation module for creating comprehensive HTML reports

This module has been restructured into a package for better maintainability.
This file serves as a backward-compatibility wrapper.

The dashboard functionality is now organized as follows:
- src/dashboard/generator.py: Main DashboardGenerator class
- src/dashboard/charts.py: Chart creation functions
- src/dashboard/statistics.py: Statistical analysis functions
- src/dashboard/html_builder.py: HTML document construction
- src/dashboard/templates/: CSS and JavaScript templates

For new code, import directly from the dashboard package:
    from dashboard import DashboardGenerator
"""

# Import everything from the dashboard package for backward compatibility
from .dashboard import (
    DashboardGenerator,
    create_confusion_matrix,
    create_performance_metrics,
    create_prediction_comparison,
    create_statistics_overview,
    create_interactive_table,
    create_correlation_heatmap,
    generate_summary_stats,
    build_complete_html
)

# Re-export the create_dashboard function for compatibility
import pandas as pd
from pathlib import Path
from typing import Optional, List
import plotly.graph_objects as go

from .utils import AnalysisLogger, DataManager


def create_dashboard(
    predictions_path: str,
    tsne_results_path: str,
    attentions_path: Optional[str] = None,
    output_dir: str = "outputs",
    report_name: str = "legionella_analysis_report.html",
    highlight_proteins: Optional[List[str]] = None
) -> Path:
    """
    Create complete dashboard from analysis results

    Args:
        predictions_path: Path to predictions CSV
        tsne_results_path: Path to t-SNE results
        attentions_path: Path to attention weights (optional)
        output_dir: Output directory
        report_name: Report filename
        highlight_proteins: Proteins to highlight

    Returns:
        Path to generated report
    """
    logger = AnalysisLogger(log_dir=f"{output_dir}/logs")
    dashboard = DashboardGenerator(logger=logger)

    logger.info("="*60)
    logger.info("GENERATING DASHBOARD")
    logger.info("="*60)

    # Load predictions
    predictions_df = pd.read_csv(predictions_path)

    # Load t-SNE results (will need to recreate figures)
    # This is a simplified version - in practice, you'd load the results
    # and recreate the figures using TSNEVisualizer

    # For now, create placeholder figures
    # TODO: Integrate with visualization module to recreate figures from saved data

    fig_2d = go.Figure()
    fig_3d = go.Figure()

    # Load attentions if available
    attentions_data = None
    if attentions_path:
        try:
            data_manager = DataManager(base_dir=Path(attentions_path).parent)
            attentions_data = data_manager.load_attentions(Path(attentions_path).name)
        except Exception as e:
            logger.warning(f"Could not load attentions: {e}")

    # Generate report
    report_path = dashboard.generate_html_report(
        predictions_df=predictions_df,
        fig_tsne_2d=fig_2d,
        fig_tsne_3d=fig_3d,
        attentions_data=attentions_data,
        highlight_proteins=highlight_proteins,
        output_path=f"{output_dir}/{report_name}"
    )

    return report_path


__all__ = [
    'DashboardGenerator',
    'create_dashboard',
    'create_confusion_matrix',
    'create_performance_metrics',
    'create_prediction_comparison',
    'create_statistics_overview',
    'create_interactive_table',
    'create_correlation_heatmap',
    'generate_summary_stats',
    'build_complete_html'
]
