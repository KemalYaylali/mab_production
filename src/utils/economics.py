"""
Economic Analysis for Fed-Batch Optimization

Calculate cost savings and revenue impact from improved feeding strategies
"""

from typing import Dict
import numpy as np
from ..models.bioreactor import SimulationResults


class EconomicAnalyzer:
    """
    Calculate economic impact of fed-batch optimization
    
    Assumptions (industry typical):
    - mAb value: $10,000 - $50,000 per gram (varies by product)
    - Batch cost: $50,000 - $200,000 (labor, materials, QC)
    - Facility cost: $200M - $500M (amortized)
    - Batches per year: 15-25 (per product per line)
    """
    
    def __init__(self,
                 mab_value_per_g: float = 20000.0,
                 batch_cost: float = 100000.0,
                 facility_cost_per_batch: float = 50000.0,
                 batches_per_year: int = 20):
        """
        Initialize economic analyzer
        
        Args:
            mab_value_per_g: Market value of mAb ($/g)
            batch_cost: Cost to run one batch ($)
            facility_cost_per_batch: Amortized facility cost per batch ($)
            batches_per_year: Number of batches per year
        """
        self.mab_value = mab_value_per_g
        self.batch_cost = batch_cost
        self.facility_cost = facility_cost_per_batch
        self.batches_per_year = batches_per_year
    
    def calculate_batch_metrics(self, results: SimulationResults) -> Dict[str, float]:
        """
        Calculate economic metrics for a single batch
        
        Args:
            results: Simulation results
        
        Returns:
            Dictionary of economic metrics
        """
        # Product metrics
        final_titer = results.final_titer  # g/L
        total_mass = results.final_mass    # g
        batch_time = results.t[-1] / 24    # days
        
        # Revenue
        revenue = total_mass * self.mab_value
        
        # Costs
        total_cost = self.batch_cost + self.facility_cost
        
        # Profit
        profit = revenue - total_cost
        margin = (profit / revenue * 100) if revenue > 0 else 0
        
        # Productivity
        productivity = total_mass / batch_time  # g/day
        
        return {
            'titer_g_L': final_titer,
            'mass_g': total_mass,
            'batch_days': batch_time,
            'revenue_$': revenue,
            'cost_$': total_cost,
            'profit_$': profit,
            'margin_%': margin,
            'productivity_g_per_day': productivity,
            'revenue_$M': revenue / 1e6,
            'profit_$M': profit / 1e6
        }
    
    def compare_strategies(self, 
                          baseline_results: SimulationResults,
                          improved_results: SimulationResults) -> Dict[str, float]:
        """
        Compare economic impact of improved vs baseline strategy
        
        Args:
            baseline_results: Results from baseline strategy (e.g., fixed recipe)
            improved_results: Results from improved strategy (e.g., MPC)
        
        Returns:
            Dictionary of improvements
        """
        baseline = self.calculate_batch_metrics(baseline_results)
        improved = self.calculate_batch_metrics(improved_results)
        
        # Calculate improvements
        titer_improvement = ((improved['titer_g_L'] - baseline['titer_g_L']) / 
                            baseline['titer_g_L'] * 100)
        
        mass_improvement = ((improved['mass_g'] - baseline['mass_g']) / 
                           baseline['mass_g'] * 100)
        
        time_reduction = ((baseline['batch_days'] - improved['batch_days']) / 
                         baseline['batch_days'] * 100)
        
        profit_increase = improved['profit_$'] - baseline['profit_$']
        profit_increase_pct = (profit_increase / baseline['profit_$'] * 100 
                              if baseline['profit_$'] > 0 else 0)
        
        # Annual impact
        annual_profit_increase = profit_increase * self.batches_per_year
        
        # Additional batches possible due to time reduction
        if time_reduction > 0:
            additional_batches = int(self.batches_per_year * time_reduction / 100)
            additional_revenue = additional_batches * improved['revenue_$']
        else:
            additional_batches = 0
            additional_revenue = 0
        
        return {
            'titer_improvement_%': titer_improvement,
            'mass_improvement_%': mass_improvement,
            'time_reduction_%': time_reduction,
            'profit_per_batch_increase_$': profit_increase,
            'profit_increase_%': profit_increase_pct,
            'annual_profit_increase_$M': annual_profit_increase / 1e6,
            'additional_batches_per_year': additional_batches,
            'additional_annual_revenue_$M': additional_revenue / 1e6,
            'total_annual_benefit_$M': (annual_profit_increase + additional_revenue) / 1e6
        }
    
    def format_comparison(self, 
                         baseline_results: SimulationResults,
                         improved_results: SimulationResults,
                         strategy_names: tuple = ('Baseline', 'Improved')) -> str:
        """
        Create formatted comparison report
        
        Args:
            baseline_results: Baseline strategy results
            improved_results: Improved strategy results
            strategy_names: Names for the strategies
        
        Returns:
            Formatted report string
        """
        baseline = self.calculate_batch_metrics(baseline_results)
        improved = self.calculate_batch_metrics(improved_results)
        comparison = self.compare_strategies(baseline_results, improved_results)
        
        report = "=" * 70 + "\n"
        report += "ECONOMIC IMPACT ANALYSIS\n"
        report += "=" * 70 + "\n\n"
        
        report += "PER-BATCH METRICS\n"
        report += "-" * 70 + "\n"
        report += f"{'Metric':<30} {strategy_names[0]:>15} {strategy_names[1]:>15} {'Change':>10}\n"
        report += "-" * 70 + "\n"
        
        report += f"{'Final Titer (g/L)':<30} {baseline['titer_g_L']:>15.2f} {improved['titer_g_L']:>15.2f} {comparison['titer_improvement_%']:>9.1f}%\n"
        report += f"{'Total mAb (g)':<30} {baseline['mass_g']:>15.0f} {improved['mass_g']:>15.0f} {comparison['mass_improvement_%']:>9.1f}%\n"
        report += f"{'Batch Time (days)':<30} {baseline['batch_days']:>15.1f} {improved['batch_days']:>15.1f} {-comparison['time_reduction_%']:>9.1f}%\n"
        report += f"{'Revenue ($M)':<30} {baseline['revenue_$M']:>15.2f} {improved['revenue_$M']:>15.2f} {comparison['profit_increase_%']:>9.1f}%\n"
        report += f"{'Profit ($M)':<30} {baseline['profit_$M']:>15.2f} {improved['profit_$M']:>15.2f} {comparison['profit_increase_%']:>9.1f}%\n"
        
        report += "\n" + "=" * 70 + "\n"
        report += "ANNUAL IMPACT (Production Scale)\n"
        report += "=" * 70 + "\n"
        report += f"Batches per year:              {self.batches_per_year}\n"
        report += f"Additional profit per year:     ${comparison['annual_profit_increase_$M']:.2f}M\n"
        report += f"Additional batches possible:    {comparison['additional_batches_per_year']}\n"
        report += f"Additional revenue opportunity: ${comparison['additional_annual_revenue_$M']:.2f}M\n"
        report += f"\nTOTAL ANNUAL BENEFIT:           ${comparison['total_annual_benefit_$M']:.2f}M\n"
        report += "=" * 70 + "\n"
        
        return report


def quick_roi_analysis(titer_improvement_pct: float,
                       time_reduction_pct: float,
                       baseline_titer: float = 3.0) -> Dict[str, float]:
    """
    Quick ROI calculator for presentations
    
    Args:
        titer_improvement_pct: Titer improvement (%)
        time_reduction_pct: Time reduction (%)
        baseline_titer: Baseline titer (g/L)
    
    Returns:
        Dictionary of ROI metrics
    """
    # Assumptions
    mab_value = 20000  # $/g
    reactor_volume = 2000  # L
    batches_per_year = 20
    
    # Baseline
    baseline_mass = baseline_titer * reactor_volume
    baseline_revenue = baseline_mass * mab_value
    
    # Improved
    improved_titer = baseline_titer * (1 + titer_improvement_pct/100)
    improved_mass = improved_titer * reactor_volume
    improved_revenue = improved_mass * mab_value
    
    # Additional batches from time savings
    additional_batches = int(batches_per_year * time_reduction_pct / 100)
    
    # Annual impact
    revenue_per_batch_increase = improved_revenue - baseline_revenue
    annual_increase = revenue_per_batch_increase * batches_per_year
    additional_revenue = additional_batches * improved_revenue
    
    total_annual_benefit = annual_increase + additional_revenue
    
    return {
        'baseline_titer_g_L': baseline_titer,
        'improved_titer_g_L': improved_titer,
        'titer_improvement_%': titer_improvement_pct,
        'time_reduction_%': time_reduction_pct,
        'revenue_per_batch_increase_$M': revenue_per_batch_increase / 1e6,
        'annual_revenue_increase_$M': annual_increase / 1e6,
        'additional_batches': additional_batches,
        'additional_annual_revenue_$M': additional_revenue / 1e6,
        'total_annual_benefit_$M': total_annual_benefit / 1e6
    }


if __name__ == "__main__":
    print("Economic Analysis Module")
    print("=" * 60)
    
    # Quick ROI example
    print("\nQuick ROI Analysis Example:")
    print("Scenario: +45% titer, -30% time reduction")
    
    roi = quick_roi_analysis(
        titer_improvement_pct=45.0,
        time_reduction_pct=30.0,
        baseline_titer=3.0
    )
    
    print(f"\nResults:")
    print(f"  Baseline titer:     {roi['baseline_titer_g_L']:.2f} g/L")
    print(f"  Improved titer:     {roi['improved_titer_g_L']:.2f} g/L")
    print(f"  Per-batch increase: ${roi['revenue_per_batch_increase_$M']:.2f}M")
    print(f"  Annual increase:    ${roi['annual_revenue_increase_$M']:.2f}M")
    print(f"  Additional batches: {roi['additional_batches']}")
    print(f"  Additional revenue: ${roi['additional_annual_revenue_$M']:.2f}M")
    print(f"\n  TOTAL ANNUAL:       ${roi['total_annual_benefit_$M']:.2f}M")
