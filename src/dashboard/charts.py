"""
Chart creation functions for dashboard visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def create_confusion_matrix(predictions_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create a confusion matrix for known proteins based on true effector/housekeeping labels.

    Args:
        predictions_df (pd.DataFrame): DataFrame with columns:
            - 'prediction': predicted label ("Effector" / "Non-effector")
            - 'is_effector': True for known positive
            - 'is_housekeeping': True for known negative

    Returns:
        Optional[go.Figure]: Plotly confusion matrix figure, or None if no labeled data.
    """
    required_cols = {'prediction', 'is_effector', 'is_housekeeping'}
    if not required_cols.issubset(predictions_df.columns):
        print(f"Missing columns: {required_cols - set(predictions_df.columns)}")
        return None

    # --- Step 1. Create true label column ---
    df = predictions_df.copy()

    # Initialize with NaN
    df['true_label'] = np.nan

    # Effector = positive class
    df.loc[df['is_effector'] == True, 'true_label'] = 'Effector'

    # Housekeeping = negative class
    df.loc[df['is_housekeeping'] == True, 'true_label'] = 'Non-effector'

    # --- Step 2. Filter only labeled rows ---
    df_labeled = df[df['true_label'].notna()].copy()

    if df_labeled.empty:
        print("No labeled proteins found (is_effector or is_housekeeping True)")
        return None

    # --- Step 3. Compute confusion matrix ---
    labels = ['Non-effector', 'Effector']

    # Defensive normalization of prediction labels
    df_labeled['prediction'] = df_labeled['prediction'].str.strip().str.capitalize()

    cm = confusion_matrix(
        df_labeled['true_label'],
        df_labeled['prediction'],
        labels=labels
    )

    # --- Step 4. Percentages per row ---
    cm_percent = np.divide(
        cm.astype(float),
        cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype=float),
        where=cm.sum(axis=1, keepdims=True) != 0
    ) * 100

    # --- Step 5. Annotations ---
    annotations = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            value = cm[i, j]
            percent = cm_percent[i, j]
            annotations.append(dict(
                x=pred_label,
                y=true_label,
                text=f"{value}<br>({percent:.1f}%)",
                showarrow=False,
                font=dict(color='white' if percent > 50 else 'black', size=14)
            ))

    # --- Step 6. Plotly heatmap ---
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Count'),
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        )
    )

    fig.update_layout(
        title=f"Confusion Matrix ({len(df_labeled)} labeled proteins)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=650,
        height=550,
        annotations=annotations,
        template="plotly_white"
    )

    print(f"✅ Confusion matrix generated for {len(df_labeled)} labeled proteins.")
    return fig


def create_performance_metrics(predictions_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create performance metrics visualization

    Args:
        predictions_df: Predictions DataFrame

    Returns:
        Plotly figure or None
    """
    # Check if we have true labels
    if 'is_effector' not in predictions_df.columns:
        return None

    df_labeled = predictions_df[predictions_df['is_effector'].notna()].copy()

    if len(df_labeled) == 0:
        return None

    # Create binary labels
    y_true = df_labeled['is_effector'].astype(int)
    y_pred = df_labeled['prediction_numeric']

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker=dict(
                color=list(metrics.values()),
                colorscale='RdYlGn',
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(title='Score')
            ),
            text=[f'{v:.3f}' for v in metrics.values()],
            textposition='auto',
            hovertemplate='%{x}<br>Score: %{y:.4f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=f'Model Performance Metrics ({len(df_labeled)} labeled proteins)',
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis_range=[0, 1.1],
        width=600,
        height=400
    )

    return fig


def create_prediction_comparison(predictions_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create comparison of predictions vs true labels

    Args:
        predictions_df: Predictions DataFrame

    Returns:
        Plotly figure or None
    """
    if 'is_effector' not in predictions_df.columns:
        return None

    df_labeled = predictions_df[predictions_df['is_effector'].notna()].copy()

    if len(df_labeled) == 0:
        return None

    # Categorize predictions
    df_labeled['result'] = 'Unknown'
    df_labeled.loc[
        (df_labeled['is_effector'] == 1) & (df_labeled['prediction_numeric'] == 1),
        'result'
    ] = 'True Positive'
    df_labeled.loc[
        (df_labeled['is_effector'] == 0) & (df_labeled['prediction_numeric'] == 0),
        'result'
    ] = 'True Negative'
    df_labeled.loc[
        (df_labeled['is_effector'] == 0) & (df_labeled['prediction_numeric'] == 1),
        'result'
    ] = 'False Positive'
    df_labeled.loc[
        (df_labeled['is_effector'] == 1) & (df_labeled['prediction_numeric'] == 0),
        'result'
    ] = 'False Negative'

    # Count each category
    result_counts = df_labeled['result'].value_counts()

    # Define colors
    color_map = {
        'True Positive': '#27ae60',
        'True Negative': '#3498db',
        'False Positive': '#e74c3c',
        'False Negative': '#f39c12'
    }

    colors = [color_map.get(cat, '#95a5a6') for cat in result_counts.index]

    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=result_counts.index,
        values=result_counts.values,
        marker=dict(colors=colors),
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        textinfo='label+value+percent'
    )])

    fig.update_layout(
        title=f'Prediction Results Breakdown ({len(df_labeled)} labeled proteins)',
        width=600,
        height=500
    )

    return fig


def create_statistics_overview(predictions_df: pd.DataFrame) -> go.Figure:
    """
    Create overview statistics panel

    Args:
        predictions_df: Predictions DataFrame

    Returns:
        Plotly figure with statistics
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Prediction Distribution',
            'Localisation Distribution',
            'Confidence Distribution',
            'Effector Probability Distribution'
        ),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'histogram'}, {'type': 'histogram'}]]
    )

    # 1. Prediction distribution (pie chart)
    pred_counts = predictions_df['prediction'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=pred_counts.index,
            values=pred_counts.values,
            marker=dict(colors=['#3498db', '#e74c3c']),
            hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. Localisation distribution (bar chart)
    if 'localisation' in predictions_df.columns:
        loc_counts = predictions_df['localisation'].value_counts()
        fig.add_trace(
            go.Bar(
                x=loc_counts.index,
                y=loc_counts.values,
                marker_color='#9b59b6',
                hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

    # 3. Confidence distribution (histogram)
    fig.add_trace(
        go.Histogram(
            x=predictions_df['confidence'],
            nbinsx=30,
            marker_color='#1abc9c',
            hovertemplate='Confidence: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Effector probability distribution
    fig.add_trace(
        go.Histogram(
            x=predictions_df['probability_effector'],
            nbinsx=30,
            marker_color='#f39c12',
            hovertemplate='Probability: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Localisation", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Confidence", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Effector Probability", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Analysis Overview Statistics",
        title_font_size=20
    )

    return fig


def create_interactive_table(predictions_df: pd.DataFrame) -> str:
    """
    Create interactive filterable table

    Args:
        predictions_df: Predictions DataFrame

    Returns:
        HTML string for table
    """
    # Select relevant columns
    columns_to_show = [
        'label', 'prediction', 'confidence',
        'probability_effector', 'localisation', 'localisation_score'
    ]
    columns_to_show = [c for c in columns_to_show if c in predictions_df.columns]

    df_display = predictions_df[columns_to_show].copy()

    # Round numeric columns
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(3)

    # Create Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in df_display.columns],
            fill_color='#3498db',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[df_display[col] for col in df_display.columns],
            fill_color=[['#ecf0f1', 'white'] * len(df_display)],
            align='left',
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title="Protein Analysis Results Table",
        height=600
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_correlation_heatmap(predictions_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create correlation heatmap of numeric features

    Args:
        predictions_df: Predictions DataFrame

    Returns:
        Plotly figure or None
    """
    # Select numeric columns
    numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None

    # Compute correlation matrix
    corr_matrix = predictions_df[numeric_cols].corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="Feature Correlation Matrix",
        width=800,
        height=700,
        xaxis={'side': 'bottom'},
        font=dict(size=10)
    )

    return fig
