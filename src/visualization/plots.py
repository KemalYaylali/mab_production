"""
Visualization Components for Fed-Batch Optimizer

Professional, publication-quality plots for:
- Real-time state trajectories
- Control inputs
- Comparison between strategies
- Economic analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..models.bioreactor import SimulationResults


# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Professional color scheme
COLORS = {
    'fixed': '#E74C3C',      # Red - baseline
    'exponential': '#F39C12', # Orange - intermediate
    'mpc': '#27AE60',        # Green - optimal
    'glucose': '#3498DB',    # Blue
    'lactate': '#9B59B6',    # Purple
    'cells': '#1ABC9C',      # Teal
    'mab': '#E67E22'         # Dark orange
}


def plot_comparison_dashboard(results_dict: Dict[str, SimulationResults],
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive comparison dashboard
    
    Args:
        results_dict: Dictionary of {strategy_name: SimulationResults}
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Fed-Batch Strategy Comparison', fontsize=16, fontweight='bold')
    
    colors = [COLORS['fixed'], COLORS['exponential'], COLORS['mpc']]
    
    for idx, (name, results) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        data = results.to_dict()
        t_days = data['t'] / 24  # Convert to days
        
        # Row 1: Growth & Production
        # Viable cells
        axes[0, 0].plot(t_days, data['Xv']/1e9, label=name, color=color, linewidth=2)
        axes[0, 0].set_ylabel('Viable Cells (10⁹ cells/L)', fontweight='bold')
        axes[0, 0].set_title('Cell Growth')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Viability
        axes[0, 1].plot(t_days, data['viability'], label=name, color=color, linewidth=2)
        axes[0, 1].set_ylabel('Viability (%)', fontweight='bold')
        axes[0, 1].set_title('Culture Viability')
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # mAb titer
        axes[0, 2].plot(t_days, data['mAb']/1000, label=name, color=color, linewidth=2)
        axes[0, 2].set_ylabel('mAb Titer (g/L)', fontweight='bold')
        axes[0, 2].set_title('Product Accumulation')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Substrates
        # Glucose
        axes[1, 0].plot(t_days, data['glc'], label=name, color=color, linewidth=2)
        axes[1, 0].set_ylabel('Glucose (mM)', fontweight='bold')
        axes[1, 0].set_title('Glucose Profile')
        axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.3, label='Inhibition')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Lactate
        axes[1, 1].plot(t_days, data['lac'], label=name, color=color, linewidth=2)
        axes[1, 1].set_ylabel('Lactate (mM)', fontweight='bold')
        axes[1, 1].set_title('Lactate Accumulation')
        axes[1, 1].axhline(y=40, color='r', linestyle='--', alpha=0.3, label='Inhibition')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Glutamine
        axes[1, 2].plot(t_days, data['gln'], label=name, color=color, linewidth=2)
        axes[1, 2].set_ylabel('Glutamine (mM)', fontweight='bold')
        axes[1, 2].set_title('Glutamine Profile')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Row 3: Process variables
        # Volume
        axes[2, 0].plot(t_days, data['V'], label=name, color=color, linewidth=2)
        axes[2, 0].set_ylabel('Volume (L)', fontweight='bold')
        axes[2, 0].set_xlabel('Time (days)', fontweight='bold')
        axes[2, 0].set_title('Culture Volume')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Feed rate
        axes[2, 1].plot(t_days, data['F'], label=name, color=color, linewidth=2)
        axes[2, 1].set_ylabel('Feed Rate (L/h)', fontweight='bold')
        axes[2, 1].set_xlabel('Time (days)', fontweight='bold')
        axes[2, 1].set_title('Feeding Strategy')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Total mAb mass
        axes[2, 2].plot(t_days, data['total_mAb'], label=name, color=color, linewidth=2)
        axes[2, 2].set_ylabel('Total mAb (g)', fontweight='bold')
        axes[2, 2].set_xlabel('Time (days)', fontweight='bold')
        axes[2, 2].set_title('Total Product Mass')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_interactive_dashboard(results_dict: Dict[str, SimulationResults]) -> go.Figure:
    """
    Create interactive Plotly dashboard for Streamlit
    
    Args:
        results_dict: Dictionary of {strategy_name: SimulationResults}
    
    Returns:
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Viable Cell Density', 'Viability', 'mAb Titer',
            'Glucose', 'Lactate', 'Glutamine',
            'Culture Volume', 'Feed Rate', 'Total mAb Mass'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    colors_map = {
        'Fixed Recipe': COLORS['fixed'],
        'Exponential': COLORS['exponential'],
        'MPC': COLORS['mpc']
    }
    
    for name, results in results_dict.items():
        data = results.to_dict()
        t_days = data['t'] / 24
        color = colors_map.get(name, '#000000')
        
        # Row 1
        fig.add_trace(go.Scatter(x=t_days, y=data['Xv']/1e9, name=name, 
                                line=dict(color=color, width=2), legendgroup=name),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=t_days, y=data['viability'], name=name,
                                line=dict(color=color, width=2), legendgroup=name,
                                showlegend=False),
                     row=1, col=2)
        
        fig.add_trace(go.Scatter(x=t_days, y=data['mAb']/1000, name=name,
                                line=dict(color=color, width=2), legendgroup=name,
                                showlegend=False),
                     row=1, col=3)
        
        # Row 2
        fig.add_trace(go.Scatter(x=t_days, y=data['glc'], name=name,
                                line=dict(color=color, width=2), legendgroup=name,
                                showlegend=False),
                     row=2, col=1)
        
        fig.add_trace(go.Scatter(x=t_days, y=data['lac'], name=name,
                                line=dict(color=color, width=2), legendgroup=name,
                                showlegend=False),
                     row=2, col=2)
        
        fig.add_trace(go.Scatter(x=t_days, y=data['gln'], name=name,
                                line=dict(color=color, width=2), legendgroup=name,
                                showlegend=False),
                     row=2, col=3)
        
        # Row 3
        fig.add_trace(go.Scatter(x=t_days, y=data['V'], name=name,
                                line=dict(color=color, width=2), legendgroup=name,
                                showlegend=False),
                     row=3, col=1)
        
        fig.add_trace(go.Scatter(x=t_days, y=data['F'], name=name,
                                line=dict(color=color, width=2), legendgroup=name,
                                showlegend=False),
                     row=3, col=2)
        
        fig.add_trace(go.Scatter(x=t_days, y=data['total_mAb'], name=name,
                                line=dict(color=color, width=2), legendgroup=name,
                                showlegend=False),
                     row=3, col=3)
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (days)", row=3, col=1)
    fig.update_xaxes(title_text="Time (days)", row=3, col=2)
    fig.update_xaxes(title_text="Time (days)", row=3, col=3)
    
    fig.update_yaxes(title_text="10⁹ cells/L", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_yaxes(title_text="g/L", row=1, col=3)
    fig.update_yaxes(title_text="mM", row=2, col=1)
    fig.update_yaxes(title_text="mM", row=2, col=2)
    fig.update_yaxes(title_text="mM", row=2, col=3)
    fig.update_yaxes(title_text="L", row=3, col=1)
    fig.update_yaxes(title_text="L/h", row=3, col=2)
    fig.update_yaxes(title_text="g", row=3, col=3)
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Fed-Batch Strategy Comparison",
        title_font_size=20,
        hovermode='x unified'
    )
    
    return fig


def plot_economic_comparison(results_dict: Dict[str, SimulationResults],
                             V_max: float = 2000.0,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create economic impact visualization
    
    Args:
        results_dict: Dictionary of results
        V_max: Reactor volume (L)
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Calculate metrics
    strategies = list(results_dict.keys())
    final_titers = [results.final_titer for results in results_dict.values()]
    final_masses = [results.final_mass for results in results_dict.values()]
    batch_times = [results.t[-1] for results in results_dict.values()]
    
    # Economic calculations (simplified)
    # Assumptions: $10,000/g mAb value, $50K batch cost
    revenues = [mass * 10000 for mass in final_masses]
    costs = [50000] * len(strategies)  # Fixed batch cost
    profits = [r - c for r, c in zip(revenues, costs)]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Economic Impact Analysis', fontsize=16, fontweight='bold')
    
    colors = [COLORS['fixed'], COLORS['exponential'], COLORS['mpc']]
    
    # Final titer comparison
    axes[0, 0].bar(strategies, final_titers, color=colors)
    axes[0, 0].set_ylabel('Final Titer (g/L)', fontweight='bold')
    axes[0, 0].set_title('Product Titer Comparison')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(final_titers):
        axes[0, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Total mass
    axes[0, 1].bar(strategies, final_masses, color=colors)
    axes[0, 1].set_ylabel('Total mAb (g)', fontweight='bold')
    axes[0, 1].set_title('Total Product Mass')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(final_masses):
        axes[0, 1].text(i, v + 10, f'{v:.0f}', ha='center', fontweight='bold')
    
    # Batch time
    batch_days = [t/24 for t in batch_times]
    axes[1, 0].bar(strategies, batch_days, color=colors)
    axes[1, 0].set_ylabel('Batch Duration (days)', fontweight='bold')
    axes[1, 0].set_title('Time to Completion')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(batch_days):
        axes[1, 0].text(i, v + 0.2, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Profit per batch
    profit_M = [p/1e6 for p in profits]
    axes[1, 1].bar(strategies, profit_M, color=colors)
    axes[1, 1].set_ylabel('Profit per Batch ($M)', fontweight='bold')
    axes[1, 1].set_title('Economic Value')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(profit_M):
        axes[1, 1].text(i, v + 0.05, f'${v:.2f}M', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_table(results_dict: Dict[str, SimulationResults]) -> str:
    """
    Create formatted summary table for display
    
    Args:
        results_dict: Dictionary of results
    
    Returns:
        Formatted markdown table string
    """
    headers = ['Metric', 'Fixed Recipe', 'Exponential', 'MPC', 'Improvement']
    
    # Extract metrics
    strategies = list(results_dict.keys())
    titers = [r.final_titer for r in results_dict.values()]
    masses = [r.final_mass for r in results_dict.values()]
    times = [r.t[-1]/24 for r in results_dict.values()]
    
    # Calculate improvements (vs fixed recipe baseline)
    titer_improv = [(t - titers[0])/titers[0] * 100 for t in titers]
    time_improv = [(times[0] - t)/times[0] * 100 for t in times]
    
    # Build table
    table = "| Metric | Fixed Recipe | Exponential | MPC | Improvement |\n"
    table += "|--------|--------------|-------------|-----|-------------|\n"
    table += f"| Final Titer (g/L) | {titers[0]:.2f} | {titers[1]:.2f} | {titers[2]:.2f} | +{titer_improv[2]:.1f}% |\n"
    table += f"| Total mAb (g) | {masses[0]:.0f} | {masses[1]:.0f} | {masses[2]:.0f} | +{(masses[2]-masses[0])/masses[0]*100:.1f}% |\n"
    table += f"| Batch Time (days) | {times[0]:.1f} | {times[1]:.1f} | {times[2]:.1f} | -{time_improv[2]:.1f}% |\n"
    table += f"| Value ($M) | ${masses[0]*0.01:.2f} | ${masses[1]*0.01:.2f} | ${masses[2]*0.01:.2f} | +${(masses[2]-masses[0])*0.01:.2f}M |\n"
    
    return table


if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Available functions:")
    print("  - plot_comparison_dashboard()")
    print("  - plot_interactive_dashboard()")
    print("  - plot_economic_comparison()")
    print("  - create_summary_table()")
