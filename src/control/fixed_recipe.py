"""
Fixed Feeding Strategy (Industry Baseline)

Implements traditional fed-batch feeding based on predetermined recipe
This is the most common approach in industry - simple but suboptimal
"""

import numpy as np
from typing import Optional


class FixedFeedingStrategy:
    """
    Fixed feeding strategy with constant or bolus additions
    
    This represents the "traditional" industry approach:
    - Batch phase (no feeding)
    - Fed-batch phase with constant feed rate
    - Optional bolus additions at fixed times
    """
    
    def __init__(self,
                 feed_start: float = 48.0,
                 feed_rate: float = 2.0,
                 feed_duration: float = 240.0,
                 bolus_times: Optional[list] = None,
                 bolus_volumes: Optional[list] = None):
        """
        Initialize fixed feeding strategy
        
        Args:
            feed_start: Time to start feeding (h)
            feed_rate: Constant feed rate (L/h)
            feed_duration: Duration of feeding (h)
            bolus_times: Times for bolus additions (h)
            bolus_volumes: Volumes for bolus additions (L)
        """
        self.feed_start = feed_start
        self.feed_rate = feed_rate
        self.feed_duration = feed_duration
        self.feed_end = feed_start + feed_duration
        
        self.bolus_times = bolus_times or []
        self.bolus_volumes = bolus_volumes or []
        
        if len(self.bolus_times) != len(self.bolus_volumes):
            raise ValueError("Bolus times and volumes must have same length")
    
    def __call__(self, t: float, state: np.ndarray) -> float:
        """
        Calculate feed rate at time t
        
        Args:
            t: Current time (h)
            state: Current state vector (not used in fixed strategy)
        
        Returns:
            Feed rate (L/h)
        """
        # Constant feeding during fed-batch phase
        if self.feed_start <= t <= self.feed_end:
            return self.feed_rate
        
        # Bolus additions (approximated as high feed rate over short period)
        for t_bolus, V_bolus in zip(self.bolus_times, self.bolus_volumes):
            if abs(t - t_bolus) < 0.1:  # Within 6 min of bolus time
                return V_bolus / 0.1  # Deliver volume over 0.1 h
        
        return 0.0
    
    def get_info(self) -> dict:
        """Get strategy information"""
        return {
            'type': 'Fixed Recipe',
            'feed_start': self.feed_start,
            'feed_rate': self.feed_rate,
            'feed_duration': self.feed_duration,
            'n_bolus': len(self.bolus_times)
        }


class FixedRecipeLibrary:
    """
    Library of common fixed feeding recipes from literature
    """
    
    @staticmethod
    def conservative_recipe(V_max: float = 2000.0) -> FixedFeedingStrategy:
        """
        Conservative feeding - low risk, but lower titer
        
        Args:
            V_max: Maximum reactor volume (L)
        
        Returns:
            FixedFeedingStrategy
        """
        # Feed at 1% working volume per day
        feed_rate = V_max * 0.01 / 24
        
        return FixedFeedingStrategy(
            feed_start=48.0,
            feed_rate=feed_rate,
            feed_duration=240.0
        )
    
    @staticmethod
    def aggressive_recipe(V_max: float = 2000.0) -> FixedFeedingStrategy:
        """
        Aggressive feeding - higher titer but risk of substrate inhibition
        
        Args:
            V_max: Maximum reactor volume (L)
        
        Returns:
            FixedFeedingStrategy
        """
        # Feed at 2.5% working volume per day
        feed_rate = V_max * 0.025 / 24
        
        return FixedFeedingStrategy(
            feed_start=24.0,
            feed_rate=feed_rate,
            feed_duration=264.0
        )
    
    @staticmethod
    def typical_industry(V_max: float = 2000.0) -> FixedFeedingStrategy:
        """
        Typical industry recipe - balanced approach
        
        Args:
            V_max: Maximum reactor volume (L)
        
        Returns:
            FixedFeedingStrategy
        """
        # Feed at 2.0% working volume per day (increased for cell growth)
        feed_rate = V_max * 0.020 / 24
        
        # Add glucose bolus on day 7
        return FixedFeedingStrategy(
            feed_start=36.0,  # Start earlier (day 1.5)
            feed_rate=feed_rate,
            feed_duration=300.0,  # Feed longer
            bolus_times=[168.0],  # Day 7
            bolus_volumes=[V_max * 0.05]  # 5% volume bolus
        )


if __name__ == "__main__":
    print("Fixed Feeding Strategy Examples")
    print("=" * 60)
    
    # Test different recipes
    recipes = {
        'Conservative': FixedRecipeLibrary.conservative_recipe(),
        'Aggressive': FixedRecipeLibrary.aggressive_recipe(),
        'Typical Industry': FixedRecipeLibrary.typical_industry()
    }
    
    for name, strategy in recipes.items():
        print(f"\n{name}:")
        info = strategy.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test feed rate at different times
        print(f"\n  Feed rates:")
        for t in [0, 48, 100, 168, 300]:
            F = strategy(t, np.zeros(8))
            print(f"    t={t:3.0f}h: F={F:.3f} L/h")
