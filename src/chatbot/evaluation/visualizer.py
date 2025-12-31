"""
Visualization Module for RAG Chatbot Evaluation.

Generates charts and graphs for evaluation reports:
- Metrics summary bar chart
- Correlation heatmap
- Distribution histograms
- Flow analysis diagrams

Author: AI Evaluation Framework
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# Configure matplotlib for Vietnamese text
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

# Try to use a font that supports Vietnamese
try:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
except:
    pass

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.unicode_minus'] = False


class EvaluationVisualizer:
    """
    Visualization generator for RAG evaluation results.
    
    Creates various charts and graphs for report generation.
    """
    
    def __init__(self, output_dir: str | Path = "output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_summary(
        self,
        scores: Dict[str, float],
        title: str = "Evaluation Metrics Summary",
        filename: str = "metrics_summary.png"
    ) -> Path:
        """
        Create a bar chart summarizing all metric scores.
        
        Args:
            scores: Dictionary of metric_name -> score
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = list(scores.keys())
        values = list(scores.values())
        
        # Color based on score value
        colors = []
        for v in values:
            if v >= 0.8:
                colors.append('#2ecc71')  # Green
            elif v >= 0.6:
                colors.append('#3498db')  # Blue
            elif v >= 0.4:
                colors.append('#f39c12')  # Orange
            else:
                colors.append('#e74c3c')  # Red
        
        bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
        
        # Add threshold lines
        ax.axhline(y=0.8, color='#2ecc71', linestyle='--', alpha=0.5, label='Excellent (0.8)')
        ax.axhline(y=0.6, color='#f39c12', linestyle='--', alpha=0.5, label='Good (0.6)')
        ax.axhline(y=0.4, color='#e74c3c', linestyle='--', alpha=0.5, label='Warning (0.4)')
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right', fontsize=9)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None,
        title: str = "Metrics Correlation Heatmap",
        filename: str = "correlation_heatmap.png"
    ) -> Path:
        """
        Create a heatmap showing correlation between metrics.
        
        Args:
            df: DataFrame with evaluation results
            metric_columns: List of metric column names (auto-detected if None)
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved chart
        """
        import seaborn as sns
        
        # Auto-detect metric columns (numeric columns with reasonable values)
        if metric_columns is None:
            metric_columns = [
                col for col in df.columns 
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']
                and col not in ['latency_ms']
                and df[col].max() <= 1.0  # Likely a metric score
            ]
        
        if len(metric_columns) < 2:
            print("Not enough metrics for correlation heatmap")
            return None
        
        # Compute correlation matrix
        corr_matrix = df[metric_columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def plot_distribution(
        self,
        df: pd.DataFrame,
        metric_name: str,
        title: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Create a histogram showing distribution of a metric.
        
        Args:
            df: DataFrame with evaluation results
            metric_name: Name of the metric column
            title: Chart title (auto-generated if None)
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved chart
        """
        import seaborn as sns
        
        if metric_name not in df.columns:
            print(f"Metric '{metric_name}' not found in DataFrame")
            return None
        
        if title is None:
            title = f"Distribution of {metric_name}"
        
        if filename is None:
            safe_name = metric_name.lower().replace(' ', '_')
            filename = f"distribution_{safe_name}.png"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = df[metric_name].dropna()
        
        # Histogram with KDE
        sns.histplot(data, kde=True, color='#3498db', ax=ax, bins=20)
        
        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        
        ax.axvline(mean_val, color='#e74c3c', linestyle='--', 
                   label=f'Mean: {mean_val:.3f}', linewidth=2)
        ax.axvline(median_val, color='#2ecc71', linestyle='--', 
                   label=f'Median: {median_val:.3f}', linewidth=2)
        
        # Add threshold markers
        ax.axvline(0.8, color='gray', linestyle=':', alpha=0.5, label='Threshold (0.8)')
        
        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def plot_metrics_by_evolution_type(
        self,
        df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None,
        title: str = "Metrics by Question Type",
        filename: str = "metrics_by_type.png"
    ) -> Path:
        """
        Create a grouped bar chart showing metrics by evolution type.
        
        Args:
            df: DataFrame with evaluation results
            metric_columns: List of metric column names
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved chart
        """
        if 'evolution_type' not in df.columns:
            print("Column 'evolution_type' not found")
            return None
        
        # Auto-detect metric columns
        if metric_columns is None:
            metric_columns = [
                col for col in df.columns 
                if df[col].dtype in ['float64', 'float32']
                and col not in ['latency_ms']
                and df[col].max() <= 1.0
            ]
        
        if not metric_columns:
            print("No metric columns found")
            return None
        
        # Group by evolution type and calculate means
        grouped = df.groupby('evolution_type')[metric_columns].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create grouped bar chart
        x = np.arange(len(grouped.index))
        width = 0.8 / len(metric_columns)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(metric_columns)))
        
        for i, (metric, color) in enumerate(zip(metric_columns, colors)):
            offset = (i - len(metric_columns)/2 + 0.5) * width
            bars = ax.bar(x + offset, grouped[metric], width, 
                         label=metric, color=color, edgecolor='white')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 2),
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=8, rotation=0)
        
        ax.set_xlabel('Question Type', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.set_ylim(0, 1.2)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def plot_latency_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Response Latency Distribution",
        filename: str = "latency_distribution.png"
    ) -> Path:
        """
        Create a histogram showing response latency distribution.
        
        Args:
            df: DataFrame with evaluation results
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved chart
        """
        import seaborn as sns
        
        if 'latency_ms' not in df.columns:
            print("Column 'latency_ms' not found")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = df['latency_ms'].dropna()
        
        # Histogram with KDE
        sns.histplot(data, kde=True, color='#9b59b6', ax=ax, bins=20)
        
        # Add statistics
        mean_val = data.mean()
        p95_val = data.quantile(0.95)
        
        ax.axvline(mean_val, color='#e74c3c', linestyle='--', 
                   label=f'Mean: {mean_val:.0f}ms', linewidth=2)
        ax.axvline(p95_val, color='#f39c12', linestyle='--', 
                   label=f'P95: {p95_val:.0f}ms', linewidth=2)
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def plot_radar_chart(
        self,
        scores: Dict[str, float],
        title: str = "RAG Metrics Radar Chart",
        filename: str = "radar_chart.png"
    ) -> Path:
        """
        Create a radar/spider chart for RAG metrics visualization.
        
        Args:
            scores: Dictionary of metric_name -> score
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved chart
        """
        # Filter to RAG-specific metrics for cleaner radar
        rag_metrics = ['Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall']
        available_metrics = [m for m in rag_metrics if m in scores]
        
        if len(available_metrics) < 3:
            # Use all available metrics
            available_metrics = list(scores.keys())
        
        if len(available_metrics) < 3:
            print("Not enough metrics for radar chart")
            return None
        
        values = [scores[m] for m in available_metrics]
        num_vars = len(available_metrics)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        values += values[:1]  # Close the polygon
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Draw the chart
        ax.fill(angles, values, color='#3498db', alpha=0.25)
        ax.plot(angles, values, color='#3498db', linewidth=2, marker='o')
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics, size=11)
        
        # Set y-axis
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
        
        # Add threshold circle
        threshold_angles = np.linspace(0, 2*np.pi, 100)
        ax.plot(threshold_angles, [0.6]*100, 'g--', alpha=0.5, linewidth=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def generate_full_report(
        self,
        df: pd.DataFrame,
        aggregate_scores: Dict[str, float]
    ) -> List[Path]:
        """
        Generate all visualization charts for a complete report.
        
        Args:
            df: DataFrame with evaluation results
            aggregate_scores: Dictionary of aggregate metric scores
            
        Returns:
            List of paths to generated charts
        """
        charts = []
        
        print("\n📊 Generating visualization report...")
        print("="*50)
        
        # 1. Metrics Summary Bar Chart
        path = self.plot_metrics_summary(aggregate_scores)
        if path:
            charts.append(path)
        
        # 2. Radar Chart
        path = self.plot_radar_chart(aggregate_scores)
        if path:
            charts.append(path)
        
        # 3. Correlation Heatmap
        path = self.plot_correlation_heatmap(df)
        if path:
            charts.append(path)
        
        # 4. Distribution for each metric
        metric_columns = [
            col for col in df.columns 
            if df[col].dtype in ['float64', 'float32']
            and col not in ['latency_ms']
            and 0 <= df[col].max() <= 1.0
        ]
        
        for metric in metric_columns:
            path = self.plot_distribution(df, metric)
            if path:
                charts.append(path)
        
        # 5. Metrics by Evolution Type
        path = self.plot_metrics_by_evolution_type(df)
        if path:
            charts.append(path)
        
        # 6. Latency Distribution
        path = self.plot_latency_distribution(df)
        if path:
            charts.append(path)
        
        print(f"\n✅ Generated {len(charts)} charts in {self.output_dir}")
        
        return charts
