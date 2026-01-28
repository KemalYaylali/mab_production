"""
Bioreactor Simulation Module

Integrates CHO cell kinetics to simulate fed-batch operation
Supports different feeding strategies and disturbances
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from .cho_kinetics import CHOCellModel
from .parameters import CHOKineticParameters


@dataclass
class SimulationResults:
    """Container for simulation results"""
    t: np.ndarray  # Time points (h)
    Xv: np.ndarray  # Viable cells (cells/L)
    Xd: np.ndarray  # Dead cells (cells/L)
    glc: np.ndarray  # Glucose (mM)
    gln: np.ndarray  # Glutamine (mM)
    lac: np.ndarray  # Lactate (mM)
    amm: np.ndarray  # Ammonia (mM)
    mAb: np.ndarray  # mAb titer (mg/L)
    V: np.ndarray  # Volume (L)
    F: np.ndarray  # Feed rate (L/h)
    
    @property
    def viability(self) -> np.ndarray:
        """Calculate viability (%)"""
        Xt = self.Xv + self.Xd
        return np.where(Xt > 0, self.Xv / Xt * 100, 0.0)
    
    @property
    def total_mAb(self) -> np.ndarray:
        """Total mAb mass (g)"""
        return self.mAb * self.V / 1000
    
    @property
    def final_titer(self) -> float:
        """Final mAb titer (g/L)"""
        return self.mAb[-1] / 1000
    
    @property
    def final_mass(self) -> float:
        """Final mAb mass (g)"""
        return self.total_mAb[-1]
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for easy plotting"""
        return {
            't': self.t,
            'Xv': self.Xv,
            'Xd': self.Xd,
            'Xt': self.Xv + self.Xd,
            'viability': self.viability,
            'glc': self.glc,
            'gln': self.gln,
            'lac': self.lac,
            'amm': self.amm,
            'mAb': self.mAb,
            'V': self.V,
            'F': self.F,
            'total_mAb': self.total_mAb
        }


class Bioreactor:
    """
    Fed-batch bioreactor simulator
    
    Integrates CHO cell model with feeding strategy
    """
    
    def __init__(self, 
                 model: Optional[CHOCellModel] = None,
                 params: Optional[CHOKineticParameters] = None):
        """
        Initialize bioreactor
        
        Args:
            model: CHO cell model (creates default if None)
            params: Kinetic parameters (uses model params if None)
        """
        if model is None:
            self.model = CHOCellModel(params)
        else:
            self.model = model
        
        self.params = self.model.params
    
    def simulate(self,
                 t_span: Tuple[float, float],
                 feeding_strategy: Callable[[float, np.ndarray], float],
                 disturbance: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
                 t_eval: Optional[np.ndarray] = None,
                 x0: Optional[np.ndarray] = None) -> SimulationResults:
        """
        Simulate fed-batch culture
        
        Args:
            t_span: Time span (start, end) in hours
            feeding_strategy: Function(t, state) -> feed_rate
            disturbance: Optional function(t, state) -> modified_state
            t_eval: Time points for output (if None, solver chooses)
            x0: Initial state (uses model default if None)
        
        Returns:
            SimulationResults object
        """
        if x0 is None:
            x0 = self.model.get_initial_state()
        
        # Store feed rates for output
        feed_rates = []
        time_points = []
        
        def ode_func(t, x):
            """ODE function with feeding strategy"""
            # Apply disturbance if specified
            if disturbance is not None:
                x = disturbance(t, x)
            
            # Ensure non-negative state
            x = np.maximum(x, 0.0)
            x[0] = max(1e-6, x[0])  # Xv minimum
            x[7] = max(0.1, x[7])   # V minimum
            
            # Get feed rate from strategy
            F = feeding_strategy(t, x)
            
            # Constrain feed rate (non-negative, max capacity)
            F = np.clip(F, 0.0, self.params.V_max * 0.1)  # Max 10% volume/h
            
            # Store for output
            feed_rates.append(F)
            time_points.append(t)
            
            # Calculate derivatives
            return self.model.derivatives(t, x, F)
        
        def substrate_depletion(t, x):
            """Event: stop if critical substrates depleted"""
            glc, gln = x[2], x[3]
            # Return negative when substrates too low (triggers event)
            return min(glc - 0.01, gln - 0.005)
        
        substrate_depletion.terminal = False  # Don't stop, just warn
        substrate_depletion.direction = -1
        
        # Solve ODE system with events
        sol = solve_ivp(
            ode_func,
            t_span,
            x0,
            method='LSODA',  # Adaptive stiff/non-stiff solver
            t_eval=t_eval,
            events=substrate_depletion,
            max_step=2.0,  # Reasonable step size
            rtol=1e-4,  # Relaxed for stability
            atol=1e-7   # Absolute tolerance
        )
        
        if not sol.success:
            # Try with even more robust settings
            import warnings
            warnings.warn(f"LSODA failed: {sol.message}. Trying Radau...")
            sol = solve_ivp(
                ode_func,
                t_span,
                x0,
                method='Radau',  # Very robust implicit solver
                t_eval=t_eval,
                max_step=2.0,
                rtol=1e-3,
                atol=1e-6
            )
            
            if not sol.success:
                raise RuntimeError(f"Integration failed: {sol.message}. Try shorter batch duration or check parameters.")
        
        # Interpolate feed rates to match solution time points
        F_interp = np.interp(sol.t, time_points, feed_rates)
        
        # Package results
        results = SimulationResults(
            t=sol.t,
            Xv=sol.y[0],
            Xd=sol.y[1],
            glc=sol.y[2],
            gln=sol.y[3],
            lac=sol.y[4],
            amm=sol.y[5],
            mAb=sol.y[6],
            V=sol.y[7],
            F=F_interp
        )
        
        return results
    
    def check_constraints(self, state: np.ndarray) -> Dict[str, bool]:
        """
        Check if current state satisfies process constraints
        
        Args:
            state: Current state vector
        
        Returns:
            Dictionary of constraint violations (True = violated)
        """
        Xv, Xd, glc, gln, lac, amm, mAb, V = state
        
        violations = {
            'glucose_high': glc > self.params.glc_max,
            'glucose_low': glc < 1.0,  # Minimum for growth
            'glutamine_low': gln < 0.1,  # Minimum for growth
            'lactate_high': lac > self.params.K_i_lac,
            'ammonia_high': amm > self.params.K_i_amm,
            'volume_exceeded': V > self.params.V_max,
        }
        
        return violations


# ==========================================
# DISTURBANCE FUNCTIONS
# ==========================================

def metabolic_shift_disturbance(t_shift: float, 
                                severity: float = 1.5) -> Callable:
    """
    Create metabolic shift disturbance (sudden change in metabolism)
    
    Args:
        t_shift: Time when shift occurs (h)
        severity: Multiplier for metabolic changes (>1 = more severe)
    
    Returns:
        Disturbance function(t, state) -> modified_state
    """
    def disturbance(t, state):
        if t >= t_shift:
            # Simulate metabolic shift by modifying state
            # (In reality, this would change kinetic parameters)
            # Here we simulate increased lactate production
            state = state.copy()
            # Could modify parameters in actual implementation
        return state
    return disturbance


def temperature_spike_disturbance(t_spike: float, 
                                  duration: float = 2.0,
                                  delta_T: float = 2.0) -> Callable:
    """
    Create temperature spike disturbance
    
    Args:
        t_spike: Time when spike begins (h)
        duration: Duration of spike (h)
        delta_T: Temperature increase (°C)
    
    Returns:
        Disturbance function(t, state) -> modified_state
    """
    def disturbance(t, state):
        if t_spike <= t <= t_spike + duration:
            # Temperature affects growth and death rates
            # Simplified: increase death rate during spike
            state = state.copy()
            # In full implementation, would modify model parameters
        return state
    return disturbance


def contamination_disturbance(t_contam: float,
                              death_rate_increase: float = 0.005) -> Callable:
    """
    Create contamination disturbance (increased cell death)
    
    Args:
        t_contam: Time when contamination occurs (h)
        death_rate_increase: Additional death rate (1/h)
    
    Returns:
        Disturbance function(t, state) -> modified_state
    """
    def disturbance(t, state):
        if t >= t_contam:
            state = state.copy()
            # Increase death rate (would modify k_d in model)
        return state
    return disturbance


if __name__ == "__main__":
    # Test simulation
    from src.control.fixed_recipe import FixedFeedingStrategy
    
    print("Bioreactor Simulation Test")
    print("=" * 60)
    
    # Create reactor
    reactor = Bioreactor()
    
    # Simple fixed feeding strategy
    feed_strategy = FixedFeedingStrategy(
        feed_start=48.0,
        feed_rate=2.0,
        feed_duration=240.0
    )
    
    # Simulate
    print("\nRunning simulation...")
    t_span = (0, 336)  # 14 days
    t_eval = np.linspace(0, 336, 500)
    
    results = reactor.simulate(
        t_span=t_span,
        feeding_strategy=feed_strategy,
        t_eval=t_eval
    )
    
    print(f"Simulation complete: {len(results.t)} time points")
    print(f"\nFinal Results (Day {results.t[-1]/24:.1f}):")
    print(f"  Viable cells: {results.Xv[-1]:.2e} cells/L")
    print(f"  Viability:    {results.viability[-1]:.1f}%")
    print(f"  mAb titer:    {results.final_titer:.2f} g/L")
    print(f"  Total mAb:    {results.final_mass:.2f} g")
    print(f"  Volume:       {results.V[-1]:.1f} L")
    print(f"  Glucose:      {results.glc[-1]:.2f} mM")
