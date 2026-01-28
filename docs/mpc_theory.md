# Model Predictive Control Theory

## Mathematical Foundation

### System Model

The fed-batch CHO cell culture system is described by a set of nonlinear ordinary differential equations (ODEs):

$$
\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, u, t)
$$

Where:
- $\mathbf{x} = [X_v, X_d, Glc, Gln, Lac, Amm, mAb, V]^T$ is the state vector
- $u = F$ is the control input (feed rate)
- $t$ is time

### State Variables

| Symbol | Description | Units |
|--------|-------------|-------|
| $X_v$ | Viable cell density | cells/L |
| $X_d$ | Dead cell density | cells/L |
| $Glc$ | Glucose concentration | mM |
| $Gln$ | Glutamine concentration | mM |
| $Lac$ | Lactate concentration | mM |
| $Amm$ | Ammonia concentration | mM |
| $mAb$ | mAb concentration | mg/L |
| $V$ | Culture volume | L |

### Control Input

| Symbol | Description | Units | Constraints |
|--------|-------------|-------|-------------|
| $F$ | Feed rate | L/h | $0 \leq F \leq F_{max}$ |

## Mechanistic Model Equations

### Cell Growth and Death

**Viable cells:**
$$
\frac{dX_v}{dt} = (\mu - k_d - D) X_v
$$

**Dead cells:**
$$
\frac{dX_d}{dt} = k_d X_v - D X_d
$$

Where $D = F/V$ is the dilution rate.

### Specific Growth Rate (Monod with Inhibition)

$$
\mu = \mu_{max} \cdot \frac{Glc}{K_{Glc} + Glc} \cdot \frac{Gln}{K_{Gln} + Gln} \cdot \frac{K_{i,Glc}}{K_{i,Glc} + Glc} \cdot \frac{K_{i,Lac}}{K_{i,Lac} + Lac} \cdot \frac{K_{i,Amm}}{K_{i,Amm} + Amm}
$$

Terms:
- Monod kinetics: $\frac{S}{K_S + S}$ for substrate limitation
- Substrate inhibition: $\frac{K_I}{K_I + S}$ for toxicity

### Specific Death Rate

$$
k_d = k_{d,0} + k_{d,Lac} \frac{Lac}{K_{i,Lac}} + k_{d,Amm} \frac{Amm}{K_{i,Amm}}
$$

### Substrate Consumption

**Glucose:**
$$
\frac{dGlc}{dt} = -q_{Glc} \frac{X_v}{10^6} + D(Glc_{feed} - Glc)
$$

$$
q_{Glc} = \left(\frac{\mu}{Y_{X/Glc}/10^6} + m_{Glc}\right) \frac{Glc}{K_{Glc} + Glc}
$$

**Glutamine:**
$$
\frac{dGln}{dt} = -q_{Gln} \frac{X_v}{10^6} + D(Gln_{feed} - Gln)
$$

$$
q_{Gln} = \left(\frac{\mu}{Y_{X/Gln}/10^6} + m_{Gln}\right) \frac{Gln}{K_{Gln} + Gln}
$$

### Metabolite Production

**Lactate:**
$$
\frac{dLac}{dt} = q_{Lac} \frac{X_v}{10^6} - D \cdot Lac
$$

$$
q_{Lac} = Y_{Lac/Glc} \cdot q_{Glc} + q_{Lac,cons} \frac{Lac}{K_{Lac} + Lac}
$$

Where $q_{Lac,cons} < 0$ represents lactate consumption during metabolic shift.

**Ammonia:**
$$
\frac{dAmm}{dt} = q_{Amm} \frac{X_v}{10^6} - D \cdot Amm
$$

$$
q_{Amm} = Y_{Amm/Gln} \cdot q_{Gln}
$$

### Product Formation (Luedeking-Piret)

$$
\frac{dmAb}{dt} = q_{mAb} \frac{X_v}{10^9} - k_{deg} \cdot mAb - D \cdot mAb
$$

$$
q_{mAb} = \alpha_{mAb} \mu + \beta_{mAb}
$$

- $\alpha_{mAb}$: Growth-associated production (pg/cell/h)
- $\beta_{mAb}$: Non-growth-associated production (pg/cell/h)

### Volume

$$
\frac{dV}{dt} = F
$$

## Model Predictive Control Formulation

### Optimization Problem

At each sampling time $t_k$, solve:

$$
\min_{\mathbf{u}_{k:k+H_c-1}} J(\mathbf{x}_k, \mathbf{u}_{k:k+H_c-1})
$$

Subject to:
1. System dynamics: $\mathbf{x}_{k+1} = \mathbf{x}_k + f(\mathbf{x}_k, u_k, t_k)\Delta t$
2. State constraints: $\mathbf{x}_{min} \leq \mathbf{x}_k \leq \mathbf{x}_{max}$
3. Control constraints: $u_{min} \leq u_k \leq u_{max}$
4. Control rate constraints: $|\Delta u_k| \leq \Delta u_{max}$

### Objective Function

$$
J = -w_1 \sum_{i=k}^{k+H_p} mAb_i + w_2 \sum_{i=k}^{k+H_c} F_i + w_3 \sum_{i=k}^{k+H_p} \text{penalties}_i
$$

Terms:
- **Maximize titer:** $-w_1 \sum mAb$ (negative because we minimize)
- **Minimize feed cost:** $w_2 \sum F$
- **Penalty terms:** For constraint violations

### Horizons

- **Prediction Horizon ($H_p$):** How far ahead to predict (typically 24-48 hours)
- **Control Horizon ($H_c$):** How many control moves to optimize (typically $H_c < H_p$)
- **Sampling Time ($\Delta t$):** Time between control updates (typically 0.5-2 hours)

### Constraints

**State constraints:**
$$
\begin{aligned}
Glc_{min} &\leq Glc \leq Glc_{max} \\
Gln_{min} &\leq Gln &\\
Lac &\leq Lac_{max} \\
Amm &\leq Amm_{max} \\
V &\leq V_{max}
\end{aligned}
$$

**Control constraints:**
$$
\begin{aligned}
0 &\leq F \leq F_{max} \\
|\Delta F| &\leq \Delta F_{max}
\end{aligned}
$$

### Receding Horizon Strategy

1. At time $t_k$, measure state $\mathbf{x}_k$
2. Solve optimization problem for $u_k, u_{k+1}, \ldots, u_{k+H_c-1}$
3. Apply only $u_k$ to the system
4. Move to next time step $t_{k+1}$
5. Repeat

This provides:
- **Feedback:** Measurement updates correct for model errors
- **Robustness:** Re-optimization handles disturbances
- **Feasibility:** Constraints are checked at every step

## Numerical Implementation

### Prediction Model

For computational efficiency, use Euler integration:

$$
\mathbf{x}_{i+1} = \mathbf{x}_i + f(\mathbf{x}_i, u_i, t_i) \Delta t
$$

For better accuracy, use Runge-Kutta 4th order (RK4):

$$
\begin{aligned}
k_1 &= f(\mathbf{x}_i, u_i, t_i) \\
k_2 &= f(\mathbf{x}_i + \frac{\Delta t}{2}k_1, u_i, t_i + \frac{\Delta t}{2}) \\
k_3 &= f(\mathbf{x}_i + \frac{\Delta t}{2}k_2, u_i, t_i + \frac{\Delta t}{2}) \\
k_4 &= f(\mathbf{x}_i + \Delta t k_3, u_i, t_i + \Delta t) \\
\mathbf{x}_{i+1} &= \mathbf{x}_i + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

### Optimization Solver

Use Sequential Least Squares Programming (SLSQP) because:
- Handles nonlinear constraints efficiently
- Works well with smooth gradients
- Robust to local minima (with proper initialization)

Alternative solvers:
- Interior Point (IPOPT)
- Active Set methods
- Direct Multiple Shooting

### Constraint Handling

**Soft constraints:** Add slack variables with penalties

$$
J_{penalty} = w_s \sum_{i} (\max(0, Glc_i - Glc_{max}))^2
$$

**Hard constraints:** Enforce via optimizer constraints

$$
g(\mathbf{x}, \mathbf{u}) \geq 0
$$

## Performance Analysis

### Closed-Loop Stability

MPC provides stability if:
1. Prediction horizon $H_p$ is sufficiently large
2. Terminal constraint or penalty is used
3. Model is accurate enough for short-term prediction

### Robustness

Sources of uncertainty:
- **Model mismatch:** Real kinetics differ from model
- **Measurement noise:** Sensor errors
- **Disturbances:** Metabolic shifts, temperature changes

MPC handles these through:
- Frequent re-optimization (feedback)
- Conservative constraint margins
- Robust horizon selection

### Computational Complexity

**Per MPC iteration:**
- Prediction: $O(H_p \times n_{states})$
- Optimization: $O(H_c^3)$ for SLSQP
- Total: Typically 0.5-2 seconds on modern hardware

For hourly control, this is very manageable.

## Parameter Tuning

### Objective Weights

**Maximize titer:**
- Increase $w_1$ (titer weight)
- Decrease $w_2$ (feed cost weight)

**Minimize time:**
- Add time penalty: $w_t \sum (mAb_{target} - mAb_i)$
- Increase aggressiveness of feeding

**Minimize cost:**
- Increase $w_2$ (feed cost)
- Add explicit economic objective

### Horizon Selection

**Prediction horizon:**
- Rule of thumb: 1-2× batch time constant
- For CHO: 24-48 hours works well
- Longer → better optimization, higher computation

**Control horizon:**
- Typically $H_c = 0.5 H_p$
- Too long → overfitting to model errors
- Too short → conservative, suboptimal

### Constraint Margins

Add safety margins to prevent violations:

$$
Glc_{effective,max} = Glc_{max} - \text{margin}
$$

Typical margins:
- Glucose: ±2 mM
- Volume: -5% of $V_{max}$
- Feed rate: -10% of $F_{max}$

## References

1. **MPC Theory:**
   - Camacho & Bordons, "Model Predictive Control" (2007)
   - Rawlings & Mayne, "Model Predictive Control: Theory and Design" (2009)

2. **Bioprocess Modeling:**
   - Xing et al., Biotechnology Progress (2009)
   - Naderi et al., Chemical Engineering & Technology (2011)
   - Kontoravdi et al., Trends in Biotechnology (2010)

3. **MPC in Bioprocessing:**
   - Craven et al., Biotechnology & Bioengineering (2014)
   - Steinboeck et al., Control Engineering Practice (2015)

## Implementation Notes

**Initialization:**
- Start with previous control as initial guess
- Gradually increase $H_c$ from 1 to full horizon
- Use "warm start" from previous solution

**Failure Handling:**
- If optimization fails, use last successful solution
- Implement fallback to conservative fixed recipe
- Log all failures for analysis

**Validation:**
- Compare predictions vs. measurements every cycle
- Track prediction error over time
- Adjust model parameters if systematic bias observed

---

This document provides the mathematical foundation for the MPC implementation in `src/control/mpc_controller.py`.
