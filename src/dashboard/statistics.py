"""
Statistical analysis functions for dashboard
"""

import pandas as pd
from typing import Dict, Any
from datetime import datetime


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for the dashboard

    Args:
        df: Predictions DataFrame

    Returns:
        Dictionary with summary statistics
    """
    stats = {
        'total_proteins': len(df),
        'num_effectors': int((df['prediction'] == 'Effector').sum()),
        'num_non_effectors': int((df['prediction'] == 'Non_effector').sum()),
        'mean_confidence': float(df['confidence'].mean()),
        'median_confidence': float(df['confidence'].median()),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Add labeled proteins stats if available
    if 'is_effector' in df.columns:
        df_labeled = df[df['is_effector'].notna()]
        stats['num_labeled'] = len(df_labeled)
        stats['num_unlabeled'] = len(df) - len(df_labeled)

        if len(df_labeled) > 0:
            # Calculate accuracy on labeled subset
            y_true = df_labeled['is_effector'].astype(int)
            y_pred = df_labeled['prediction_numeric']
            from sklearn.metrics import accuracy_score
            stats['accuracy'] = float(accuracy_score(y_true, y_pred))

    if 'localisation' in df.columns:
        stats['localisations'] = df['localisation'].value_counts().to_dict()

    return stats
