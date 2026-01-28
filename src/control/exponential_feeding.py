"""
Exponential Feeding Strategy

Implements exponential feeding to maintain constant specific growth rate
More advanced than fixed recipes, but still open-loop (no feedback)

Based on: F(t) = F0 * exp(μ_set * t)
"""

import numpy as np
from typing import Optional


class ExponentialFeedingStrategy:
    """
    Exponential feeding strategy
    
    Designed to maintain constant specific growth rate by exponentially
    increasing feed rate. More sophisticated than fixed recipes, but
    still open-loop (doesn't adapt to actual culture conditions).
    
    Theory:
        For constant μ: dX/dt = μ*X → X(t) = X0*exp(μ*t)
        To maintain substrate: F(t) = F0*exp(μ*t)
    """
    
    def __init__(self,
                 feed_start: float = 48.0,
                 F0: float = 1.0,
                 mu_set: float = 0.02,
                 feed_duration: float = 240.0,
                 max_feed_rate: float = 20.0):
        """
        Initialize exponential feeding strategy
        
        Args:
            feed_start: Time to start feeding (h)
            F0: Initial feed rate (L/h)
            mu_set: Target specific growth rate (1/h)
            feed_duration: Duration of feeding (h)
            max_feed_rate: Maximum allowed feed rate (L/h)
        """
        self.feed_start = feed_start
        self.F0 = F0
        self.mu_set = mu_set
        self.feed_duration = feed_duration
        self.feed_end = feed_start + feed_duration
        self.max_feed_rate = max_feed_rate
    
    def __call__(self, t: float, state: np.ndarray) -> float:
        """
        Calculate feed rate at time t
        
        Args:
            t: Current time (h)
            state: Current state vector (not used in open-loop strategy)
        
        Returns:
            Feed rate (L/h)
        """
        if not (self.feed_start <= t <= self.feed_end):
            return 0.0
        
        # Time since feeding started
        t_feed = t - self.feed_start
        
        # Exponential feed rate
        F = self.F0 * np.exp(self.mu_set * t_feed)
        
        # Cap at maximum
        return min(F, self.max_feed_rate)
    
    def get_info(self) -> dict:
        """Get strategy information"""
        return {
            'type': 'Exponential Feeding',
            'feed_start': self.feed_start,
            'F0': self.F0,
            'mu_set': self.mu_set,
            'feed_duration': self.feed_duration,
            'max_feed_rate': self.max_feed_rate
        }


class AdaptiveExponentialStrategy:
    """
    Adaptive exponential feeding with phase switching
    
    Combines exponential feeding with simple heuristics:
    - Growth phase: High μ_set
    - Production phase: Low μ_set
    
    Still open-loop, but attempts to match culture phases
    """
    
    def __init__(self,
                 feed_start: float = 48.0,
                 F0: float = 1.0,
                 mu_growth: float = 0.025,
                 mu_production: float = 0.01,
                 t_switch: float = 168.0,
                 max_feed_rate: float = 20.0):
        """
        Initialize adaptive exponential strategy
        
        Args:
            feed_start: Time to start feeding (h)
            F0: Initial feed rate (L/h)
            mu_growth: Target μ during growth phase (1/h)
            mu_production: Target μ during production phase (1/h)
            t_switch: Time to switch from growth to production (h)
            max_feed_rate: Maximum allowed feed rate (L/h)
        """
        self.feed_start = feed_start
        self.F0 = F0
        self.mu_growth = mu_growth
        self.mu_production = mu_production
        self.t_switch = t_switch
        self.max_feed_rate = max_feed_rate
    
    def __call__(self, t: float, state: np.ndarray) -> float:
        """
        Calculate feed rate with phase-dependent exponential
        
        Args:
            t: Current time (h)
            state: Current state vector
        
        Returns:
            Feed rate (L/h)
        """
        if t < self.feed_start:
            return 0.0
        
        t_feed = t - self.feed_start
        
        # Phase-dependent growth rate
        if t < self.t_switch:
            # Growth phase
            mu_eff = self.mu_growth
            F = self.F0 * np.exp(mu_eff * t_feed)
        else:
            # Production phase - restart exponential at lower rate
            t_switch_feed = self.t_switch - self.feed_start
            F_switch = self.F0 * np.exp(self.mu_growth * t_switch_feed)
            t_prod = t - self.t_switch
            F = F_switch * np.exp(self.mu_production * t_prod)
        
        return min(F, self.max_feed_rate)
    
    def get_info(self) -> dict:
        """Get strategy information"""
        return {
            'type': 'Adaptive Exponential',
            'feed_start': self.feed_start,
            'F0': self.F0,
            'mu_growth': self.mu_growth,
            'mu_production': self.mu_production,
            't_switch': self.t_switch,
            'max_feed_rate': self.max_feed_rate
        }


class ExponentialRecipeLibrary:
    """
    Library of exponential feeding recipes
    """
    
    @staticmethod
    def standard_exponential(V_max: float = 2000.0) -> ExponentialFeedingStrategy:
        """
        Standard exponential feeding
        
        Args:
            V_max: Maximum reactor volume (L)
        
        Returns:
            ExponentialFeedingStrategy
        """
        F0 = V_max * 0.005  # Start at 0.5% volume/h
        
        return ExponentialFeedingStrategy(
            feed_start=48.0,
            F0=F0,
            mu_set=0.020,  # Moderate growth rate
            feed_duration=240.0,
            max_feed_rate=V_max * 0.05  # Cap at 5% volume/h
        )
    
    @staticmethod
    def growth_optimized(V_max: float = 2000.0) -> ExponentialFeedingStrategy:
        """
        Optimized for maximizing cell density
        
        Args:
            V_max: Maximum reactor volume (L)
        
        Returns:
            ExponentialFeedingStrategy
        """
        F0 = V_max * 0.008
        
        return ExponentialFeedingStrategy(
            feed_start=24.0,
            F0=F0,
            mu_set=0.030,  # Higher growth rate
            feed_duration=168.0,
            max_feed_rate=V_max * 0.08
        )
    
    @staticmethod
    def two_phase_adaptive(V_max: float = 2000.0) -> AdaptiveExponentialStrategy:
        """
        Two-phase strategy: growth then production
        
        Args:
            V_max: Maximum reactor volume (L)
        
        Returns:
            AdaptiveExponentialStrategy
        """
        F0 = V_max * 0.006
        
        return AdaptiveExponentialStrategy(
            feed_start=48.0,
            F0=F0,
            mu_growth=0.025,
            mu_production=0.008,
            t_switch=168.0,  # Switch at day 7
            max_feed_rate=V_max * 0.06
        )


if __name__ == "__main__":
    print("Exponential Feeding Strategy Examples")
    print("=" * 60)
    
    # Test different recipes
    recipes = {
        'Standard Exponential': ExponentialRecipeLibrary.standard_exponential(),
        'Growth Optimized': ExponentialRecipeLibrary.growth_optimized(),
        'Two-Phase Adaptive': ExponentialRecipeLibrary.two_phase_adaptive()
    }
    
    for name, strategy in recipes.items():
        print(f"\n{name}:")
        info = strategy.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test feed rate progression
        print(f"\n  Feed rate progression:")
        for t in [48, 96, 168, 240, 288]:
            F = strategy(t, np.zeros(8))
            print(f"    t={t:3.0f}h (Day {t/24:4.1f}): F={F:.3f} L/h")
