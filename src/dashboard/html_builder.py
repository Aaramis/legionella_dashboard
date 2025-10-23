"""
HTML document builder for dashboard
"""

import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional
import json
import base64


def read_template_file(filename: str) -> str:
    """Read a template file from the templates directory"""
    template_dir = Path(__file__).parent / 'templates'
    file_path = template_dir / filename
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def build_summary_cards_html(summary_stats: Dict) -> str:
    """Build summary cards HTML"""
    summary_html = f"""
    <div class="summary-cards">
        <div class="card">
            <h3>{summary_stats['total_proteins']}</h3>
            <p>Total Proteins</p>
        </div>
        <div class="card effector">
            <h3>{summary_stats['num_effectors']}</h3>
            <p>Predicted Effectors</p>
        </div>
        <div class="card non-effector">
            <h3>{summary_stats['num_non_effectors']}</h3>
            <p>Predicted Non-effectors</p>
        </div>
        <div class="card">
            <h3>{summary_stats['mean_confidence']:.3f}</h3>
            <p>Mean Confidence</p>
        </div>
    </div>
    """
    return summary_html


def build_localisation_cards_html(summary_stats: Dict) -> str:
    """Build localisation distribution cards HTML"""
    if 'localisations' not in summary_stats:
        return ""

    loc_cards = "\n".join([
        f'''
        <div class="card localisation">
            <h3>{count}</h3>
            <p>{loc}</p>
        </div>
        '''
        for loc, count in summary_stats['localisations'].items()
    ])

    return f"""
    <div class="section">
        <h2>Distribution by Cellular Localisation</h2>
        <div class="summary-cards">
            {loc_cards}
        </div>
    </div>
    """


def build_performance_section_html(
    fig_metrics: Optional[go.Figure],
    fig_comparison: Optional[go.Figure]
) -> str:
    """Build performance evaluation section HTML"""
    if not (fig_metrics or fig_comparison):
        return ""

    metrics_html = fig_metrics.to_html(full_html=False, include_plotlyjs=False) if fig_metrics else ""
    comparison_html = fig_comparison.to_html(full_html=False, include_plotlyjs=False) if fig_comparison else ""

    return f"""
    <div class="section">
        <h2>Model Performance Evaluation</h2>
        <p class="description">
            Performance metrics calculated on proteins with known labels from the training set.
            These metrics show how well the model predicts on the labeled subset of the proteome.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(550px, 1fr)); gap: 2rem; margin-top: 2rem;">
            {metrics_html}
            {comparison_html}
        </div>
    </div>
    """


def build_attention_section_html(attention_figs: List[go.Figure]) -> str:
    """Build attention maps section HTML"""
    if not attention_figs:
        return ""

    attention_sections = "\n".join([
        f'<div class="attention-item">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
        for fig in attention_figs
    ])

    return f"""
    <div class="section">
        <h2>Attention Maps - Highlighted Proteins</h2>
        <p class="description">
            These visualizations show the average attention weights across all attention heads
            for each position in the protein sequence. Higher values indicate positions that
            the model focuses on more when making predictions.
        </p>
        <div class="attention-gallery">
            {attention_sections}
        </div>
    </div>
    """


def build_attribution_section_html(attribution_figs: Optional[Dict[str, go.Figure]]) -> tuple:
    """
    Build attribution analysis section HTML

    Returns:
        Tuple of (html_section, attribution_data_json)
    """
    if not attribution_figs:
        return "", "{}"

    # Convert attribution figures to JSON-serializable format
    attribution_data = {}
    for protein_id, fig in attribution_figs.items():
        attribution_data[protein_id] = fig.to_json()

    attribution_data_json = json.dumps(attribution_data)

    # Create sorted list of protein IDs for dropdown
    sorted_protein_ids = sorted(attribution_figs.keys())
    protein_options = "\n".join([
        f'<option value="{pid}">{pid}</option>'
        for pid in sorted_protein_ids
    ])

    html_section = f"""
    <div class="section">
        <h2>Integrated Gradients Attribution Analysis</h2>
        <p class="description">
            Attribution analysis shows which amino acids the model focuses on when making predictions.
            <strong>Red bars</strong> indicate amino acids that push toward the Effector prediction,
            while <strong>blue bars</strong> indicate amino acids that push toward Non-effector.
        </p>
        <div class="search-container">
            <label for="protein-selector" style="font-weight: 600; min-width: 150px;">Select Protein:</label>
            <select id="protein-selector" class="protein-search" onchange="selectProteinFromDropdown()">
                <option value="">-- Choose a protein ({len(attribution_figs)} available) --</option>
                {protein_options}
            </select>
            <button onclick="clearAttribution()" class="reset-button">Clear</button>
        </div>
        <div id="attribution-container" style="min-height: 400px; padding: 20px; background: #f9f9f9; border-radius: 8px;">
            <p style="text-align: center; color: #999; padding: 40px;">
                Select a protein from the dropdown above
            </p>
        </div>
    </div>
    """

    return html_section, attribution_data_json


def build_complete_html(
    summary_stats: Dict,
    fig_stats: go.Figure,
    fig_tsne_2d: go.Figure,
    fig_tsne_3d: go.Figure,
    table_html: str,
    attention_figs: List[go.Figure] = None,
    attribution_figs: Optional[Dict[str, go.Figure]] = None,
    fig_metrics: Optional[go.Figure] = None,
    fig_comparison: Optional[go.Figure] = None
) -> str:
    """
    Build complete HTML document for dashboard

    Args:
        summary_stats: Summary statistics dictionary
        fig_stats: Statistics overview figure
        fig_tsne_2d: 2D t-SNE figure
        fig_tsne_3d: 3D t-SNE figure
        table_html: HTML string for results table
        attention_figs: List of attention figures
        attribution_figs: Dictionary of attribution figures by protein ID
        fig_metrics: Performance metrics figure
        fig_comparison: Prediction comparison figure

    Returns:
        Complete HTML string
    """
    # Read CSS and JavaScript templates
    css = read_template_file('styles.css')
    javascript = read_template_file('scripts.js')

    # Convert figures to HTML
    stats_html = fig_stats.to_html(full_html=False, include_plotlyjs='cdn')
    tsne_2d_html = fig_tsne_2d.to_html(full_html=False, include_plotlyjs=False)
    tsne_3d_html = fig_tsne_3d.to_html(full_html=False, include_plotlyjs=False)

    # Build sections
    summary_cards = build_summary_cards_html(summary_stats)
    loc_cards = build_localisation_cards_html(summary_stats)
    performance_section = build_performance_section_html(fig_metrics, fig_comparison)
    attention_section = build_attention_section_html(attention_figs or [])
    attribution_section, attribution_data_json = build_attribution_section_html(attribution_figs)

    # Read logos
    data_path = Path(__file__).parent.parent.parent / 'data'

    # Read EIDOS SVG logo
    eidos_logo_path = data_path / 'EIDOS_full_white.svg'
    with open(eidos_logo_path, 'r', encoding='utf-8') as f:
        eidos_logo = f.read()

    # Read IRD PNG logo and encode to base64
    ird_logo_path = data_path / 'log_ird_blanc.png'
    with open(ird_logo_path, 'rb') as f:
        ird_logo_base64 = base64.b64encode(f.read()).decode('utf-8')
    ird_logo_img = f'<img src="data:image/png;base64,{ird_logo_base64}" alt="IRD Logo" style="height: 80px; width: auto;">'

    # Build complete HTML document
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legionella Protein Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
{css}
    </style>
</head>
<body>
    <div class="header">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 2rem;">
            <div style="flex-shrink: 0; height: 80px; width: auto;">
                {eidos_logo}
            </div>
            <div style="flex-grow: 1; text-align: center;">
                <h1 style="margin: 0;">Legionella Protein Analysis Report</h1>
                <p style="margin: 0.5rem 0;">Deep Learning-based Effector Prediction and Visualization</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">Generated: {summary_stats['timestamp']}</p>
            </div>
            <div style="flex-shrink: 0; height: 80px; width: auto;">
                {ird_logo_img}
            </div>
        </div>
    </div>

    <div class="container">
        {summary_cards}
        {loc_cards}

        <div class="section">
            <h2>Overview Statistics</h2>
            <p class="description">
                Comprehensive overview of the analysis results including prediction distribution,
                localization patterns, confidence scores, and effector probabilities.
            </p>
            {stats_html}
        </div>

        {performance_section}

        <div class="section">
            <h2>2D t-SNE Visualization</h2>
            <p class="description">
                Interactive 2D visualization of protein embeddings using t-SNE dimensionality reduction.
                Proteins are colored by their subcellular localization. Hover over points to see details.
                Use the legend to filter specific categories. <strong>Search for a protein by typing its label below.</strong>
            </p>
            <div class="search-container">
                <input type="text" id="search-2d" class="protein-search" placeholder="Search protein (e.g., lpg3000)..." />
                <button onclick="searchProtein2D()" class="search-button">Search</button>
                <button onclick="resetHighlight2D()" class="reset-button">Reset</button>
                <span id="search-result-2d" class="search-result"></span>
            </div>
            {tsne_2d_html}
        </div>

        <div class="section">
            <h2>3D t-SNE Visualization</h2>
            <p class="description">
                Interactive 3D visualization of protein embeddings. You can rotate the plot by dragging,
                zoom with the scroll wheel, and hover over points to see detailed information.
                This view helps identify spatial clustering patterns. <strong>Search for a protein by typing its label below.</strong>
            </p>
            <div class="search-container">
                <input type="text" id="search-3d" class="protein-search" placeholder="Search protein (e.g., lpg3000)..." />
                <button onclick="searchProtein3D()" class="search-button">Search</button>
                <button onclick="resetHighlight3D()" class="reset-button">Reset</button>
                <span id="search-result-3d" class="search-result"></span>
            </div>
            {tsne_3d_html}
        </div>

        {attention_section}

        {attribution_section}

        <div class="section">
            <h2>Detailed Results Table</h2>
            <p class="description">
                Complete analysis results for all proteins. You can scroll through the table
                to explore individual predictions and their confidence scores.
            </p>
            {table_html}
        </div>
    </div>

    <div class="footer">
        <p>Legionella Protein Analysis System v1.0</p>
        <p>Generated by Deep Learning Pipeline - Pasteur Institute</p>
    </div>

    <script>
{javascript}

// Initialize attribution data
initAttributionData({attribution_data_json});
    </script>
</body>
</html>
    """

    return html