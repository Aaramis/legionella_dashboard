"""
Dashboard module for creating comprehensive HTML reports

This module provides functionality for generating interactive HTML dashboards
with visualizations, statistics, and analysis results.

Main Components:
- DashboardGenerator: Main class for creating HTML reports
- Charts module: Functions for creating various visualizations
- Statistics module: Statistical analysis functions
- HTML Builder: HTML document construction

Example:
    from dashboard import DashboardGenerator

    generator = DashboardGenerator()
    report_path = generator.generate_html_report(
        predictions_df=predictions,
        fig_tsne_2d=tsne_2d_fig,
        fig_tsne_3d=tsne_3d_fig,
        output_path="outputs/report.html"
    )
"""

from .generator import DashboardGenerator
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

__all__ = [
    'DashboardGenerator',
    'create_confusion_matrix',
    'create_performance_metrics',
    'create_prediction_comparison',
    'create_statistics_overview',
    'create_interactive_table',
    'create_correlation_heatmap',
    'generate_summary_stats',
    'build_complete_html'
]

__version__ = '1.0.0'
