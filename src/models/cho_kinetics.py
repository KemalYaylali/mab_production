"""
CHO Cell Kinetics Model for Fed-Batch mAb Production

Implements mechanistic model based on:
- Monod growth kinetics with substrate limitation
- Substrate inhibition (glucose, lactate, ammonia)
- Luedeking-Piret product formation
- Dynamic metabolic shifts

References:
- Xing et al. (2009) Biotechnology Progress
- Naderi et al. (2011) Chemical Engineering & Technology
"""

import numpy as np
from typing import Tuple, Dict, Optional
from .parameters import CHOKineticParameters, DEFAULT_PARAMS


class CHOCellModel:
    """
    Mechanistic model for CHO cell growth, metabolism, and mAb production
    
    State variables:
        Xv: Viable cell density (cells/L)
        Xd: Dead cell density (cells/L)
        glc: Glucose concentration (mM)
        gln: Glutamine concentration (mM)
        lac: Lactate concentration (mM)
        amm: Ammonia concentration (mM)
        mAb: mAb titer (mg/L)
        V: Culture volume (L)
    
    Control inputs:
        F_feed: Feed rate (L/h)
    """
    
    def __init__(self, params: Optional[CHOKineticParameters] = None):
        """
        Initialize CHO cell model
        
        Args:
            params: Kinetic parameters (uses defaults if None)
        """
        self.params = params if params is not None else DEFAULT_PARAMS
        
        # State indices for array-based calculations
        self.state_idx = {
            'Xv': 0, 'Xd': 1, 'glc': 2, 'gln': 3,
            'lac': 4, 'amm': 5, 'mAb': 6, 'V': 7
        }
        self.n_states = len(self.state_idx)
    
    def get_initial_state(self) -> np.ndarray:
        """
        Get initial state vector
        
        Returns:
            Initial state [Xv, Xd, glc, gln, lac, amm, mAb, V]
        """
        p = self.params
        V_initial = p.V0 * p.V_max
        
        return np.array([
            p.Xv0,      # Viable cells
            p.Xd0,      # Dead cells
            p.glc0,     # Glucose
            p.gln0,     # Glutamine
            p.lac0,     # Lactate
            p.amm0,     # Ammonia
            p.mAb0,     # mAb
            V_initial   # Volume
        ])
    
    def specific_growth_rate(self, glc: float, gln: float, lac: float, amm: float) -> float:
        """
        Calculate specific growth rate with Monod kinetics and inhibition
        
        μ = μ_max * (glc/(K_glc + glc)) * (gln/(K_gln + gln)) * I_glc * I_lac * I_amm
        
        Args:
            glc: Glucose concentration (mM)
            gln: Glutamine concentration (mM)
            lac: Lactate concentration (mM)
            amm: Ammonia concentration (mM)
        
        Returns:
            Specific growth rate (1/h)
        """
        p = self.params
        
        # Ensure non-negative inputs
        glc = max(0.0, glc)
        gln = max(0.0, gln)
        lac = max(0.0, lac)
        amm = max(0.0, amm)
        
        # Monod kinetics for substrate limitation
        mu_glc = glc / (p.K_glc + glc)
        mu_gln = gln / (p.K_gln + gln)
        
        # Substrate inhibition (glucose)
        I_glc = p.K_i_glc / (p.K_i_glc + glc)
        
        # Product inhibition (lactate)
        I_lac = p.K_i_lac / (p.K_i_lac + lac)
        
        # Ammonia toxicity
        I_amm = p.K_i_amm / (p.K_i_amm + amm)
        
        mu = p.mu_max * mu_glc * mu_gln * I_glc * I_lac * I_amm
        
        return max(0.0, min(mu, p.mu_max))  # Bound between 0 and mu_max
    
    def specific_death_rate(self, lac: float, amm: float) -> float:
        """
        Calculate specific death rate
        
        k_d_total = k_d + k_d_lac * (lac/K_i_lac) + k_d_amm * (amm/K_i_amm)
        
        Args:
            lac: Lactate concentration (mM)
            amm: Ammonia concentration (mM)
        
        Returns:
            Specific death rate (1/h)
        """
        p = self.params
        
        # Basal death + lactate-induced + ammonia-induced
        k_d_total = (p.k_d + 
                     p.k_d_lac * (lac / p.K_i_lac) + 
                     p.k_d_amm * (amm / p.K_i_amm))
        
        return max(0.0, k_d_total)
    
    def specific_glucose_uptake(self, Xv: float, mu: float, glc: float) -> float:
        """
        Calculate specific glucose uptake rate
        
        q_glc = (mu/Y_Xv_glc + m_glc) * (glc/(K_glc + glc))
        
        Args:
            Xv: Viable cell density (cells/L)
            mu: Specific growth rate (1/h)
            glc: Glucose concentration (mM)
        
        Returns:
            Specific glucose uptake (mmol/10^6 cells/h)
        """
        p = self.params
        
        # Ensure non-negative
        glc = max(0.0, glc)
        
        # Growth-associated + maintenance
        # Y_Xv_glc is in cells/mmol, need (10^6 cells)/mmol for this calculation
        Y_per_million = p.Y_Xv_glc / 1e6  # (10^6 cells)/mmol
        
        if glc < 0.01:  # Very low glucose, minimal uptake
            q_glc = p.m_glc * 0.1  # Only minimal maintenance
        else:
            # q_glc (mmol/10^6cells/h) = mu (1/h) / Y (10^6cells/mmol) + m (mmol/10^6cells/h)
            q_glc = (mu / Y_per_million + p.m_glc) * (glc / (p.K_glc + glc))
        
        return max(0.0, min(q_glc, p.q_glc_max))  # Bound between 0 and max
    
    def specific_lactate_production(self, q_glc: float, lac: float, mu: float) -> float:
        """
        Calculate specific lactate production/consumption rate
        
        During growth: Lactate produced from glucose
        During stationary: Lactate consumed (metabolic shift)
        
        Args:
            q_glc: Specific glucose uptake (mmol/10^6 cells/h)
            lac: Lactate concentration (mM)
            mu: Specific growth rate (1/h)
        
        Returns:
            Specific lactate rate (mmol/10^6 cells/h, positive = production)
        """
        p = self.params
        
        # Lactate production from glucose (exponential phase)
        q_lac_prod = p.Y_lac_glc * q_glc
        
        # Lactate consumption (stationary phase, metabolic shift)
        # Only consume if growth is slow and lactate is high
        if mu < 0.01 and lac > 5.0:
            q_lac_cons = p.q_lac_max * (lac / (p.K_lac + lac))
        else:
            q_lac_cons = 0.0
        
        return q_lac_prod + q_lac_cons  # q_lac_cons is negative
    
    def specific_glutamine_uptake(self, mu: float, gln: float) -> float:
        """
        Calculate specific glutamine uptake rate
        
        Args:
            mu: Specific growth rate (1/h)
            gln: Glutamine concentration (mM)
        
        Returns:
            Specific glutamine uptake (mmol/10^6 cells/h)
        """
        p = self.params
        
        # Growth-associated + maintenance
        Y_per_million = p.Y_Xv_gln / 1e6  # (10^6 cells)/mmol
        q_gln = (mu / Y_per_million + p.m_gln) * (gln / (p.K_gln + max(gln, 0.01)))
        
        return max(0.0, min(q_gln, p.q_gln_max))
    
    def specific_ammonia_production(self, q_gln: float) -> float:
        """
        Calculate specific ammonia production from glutamine
        
        Args:
            q_gln: Specific glutamine uptake (mmol/10^6 cells/h)
        
        Returns:
            Specific ammonia production (mmol/10^6 cells/h)
        """
        return self.params.Y_amm_gln * q_gln
    
    def specific_mAb_production(self, mu: float) -> float:
        """
        Calculate specific mAb production rate (Luedeking-Piret model)
        
        q_mAb = alpha * mu + beta
        
        Args:
            mu: Specific growth rate (1/h)
        
        Returns:
            Specific mAb production (pg/cell/h)
        """
        p = self.params
        return p.alpha_mAb * mu + p.beta_mAb
    
    def derivatives(self, t: float, state: np.ndarray, u: float) -> np.ndarray:
        """
        Calculate state derivatives (ODE right-hand side)
        
        Args:
            t: Time (h)
            state: Current state [Xv, Xd, glc, gln, lac, amm, mAb, V]
            u: Control input (feed rate F, L/h)
        
        Returns:
            State derivatives [dXv/dt, dXd/dt, dglc/dt, ..., dV/dt]
        """
        # Unpack state
        Xv, Xd, glc, gln, lac, amm, mAb, V = state
        F = max(0.0, u)  # Feed rate (ensure non-negative)
        
        p = self.params
        
        # Ensure non-negative concentrations (numerical stability)
        Xv = max(1e-6, Xv)  # Small positive minimum to prevent division by zero
        Xd = max(0.0, Xd)
        glc = max(0.0, glc)
        gln = max(0.0, gln)
        lac = max(0.0, lac)
        amm = max(0.0, amm)
        mAb = max(0.0, mAb)
        V = max(0.1, V)  # Minimum volume to prevent division by zero
        
        # Calculate specific rates
        mu = self.specific_growth_rate(glc, gln, lac, amm)
        k_d = self.specific_death_rate(lac, amm)
        
        # Specific uptake/production rates (per 10^6 cells)
        Xv_e6 = Xv / 1e6
        q_glc = self.specific_glucose_uptake(Xv, mu, glc)
        q_lac = self.specific_lactate_production(q_glc, lac, mu)
        q_gln = self.specific_glutamine_uptake(mu, gln)
        q_amm = self.specific_ammonia_production(q_gln)
        q_mAb = self.specific_mAb_production(mu)
        
        # Dilution rate
        D = F / V if V > 0 else 0.0
        
        # ODE system
        dXv_dt = (mu - k_d - D) * Xv
        dXd_dt = (k_d * Xv - D * Xd)
        
        # Substrate balances with consumption limits
        dglc_dt = -q_glc * Xv_e6 + D * (p.glc_feed - glc)
        dgln_dt = -q_gln * Xv_e6 + D * (p.gln_feed - gln)
        
        # Prevent substrates from going negative (consumption can't exceed availability)
        if glc < 0.1 and dglc_dt < 0:
            dglc_dt = max(dglc_dt, -glc / 0.1)  # Limit consumption rate
        if gln < 0.05 and dgln_dt < 0:
            dgln_dt = max(dgln_dt, -gln / 0.1)  # Limit consumption rate
        
        dlac_dt = q_lac * Xv_e6 - D * lac
        damm_dt = q_amm * Xv_e6 - D * amm
        
        # mAb production (pg/cell/h → mg/L/h conversion: q * Xv / 1e9)
        dmAb_dt = q_mAb * Xv / 1e9 - p.k_deg_mAb * mAb - D * mAb
        
        # Ensure mAb doesn't decrease (can only increase or stay same)
        dmAb_dt = max(0.0, dmAb_dt)
        
        dV_dt = F
        
        return np.array([dXv_dt, dXd_dt, dglc_dt, dgln_dt, 
                        dlac_dt, damm_dt, dmAb_dt, dV_dt])
    
    def get_performance_metrics(self, state: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics from current state
        
        Args:
            state: Current state vector
        
        Returns:
            Dictionary of performance metrics
        """
        Xv, Xd, glc, gln, lac, amm, mAb, V = state
        Xt = Xv + Xd  # Total cells
        viability = (Xv / Xt * 100) if Xt > 0 else 0.0
        
        return {
            'Xv': Xv,
            'Xd': Xd,
            'Xt': Xt,
            'viability': viability,
            'glc': glc,
            'gln': gln,
            'lac': lac,
            'amm': amm,
            'mAb': mAb,
            'V': V,
            'total_mAb_mass': mAb * V / 1000,  # Convert mg/L * L → g
        }


if __name__ == "__main__":
    # Test the model
    model = CHOCellModel()
    
    print("CHO Cell Mechanistic Model")
    print("=" * 60)
    
    # Initial state
    x0 = model.get_initial_state()
    print("\nInitial State:")
    for name, idx in model.state_idx.items():
        print(f"{name:10s}: {x0[idx]:.2e}")
    
    # Test derivatives at t=0 with no feed
    dx_dt = model.derivatives(0.0, x0, u=0.0)
    print("\nInitial Derivatives (no feed):")
    for name, idx in model.state_idx.items():
        print(f"d{name}/dt:   {dx_dt[idx]:.2e}")
    
    # Test metrics
    metrics = model.get_performance_metrics(x0)
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key:15s}: {value:.2e}")
