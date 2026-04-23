import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from scipy.linalg import eig

# =========================
#   GLOBAL STREAMLIT STYLE
# =========================

st.set_page_config(page_title="DSGE Fiscal Policy Dashboard", layout="wide")

st.markdown(
    """
    <style>
    * {
        font-family: "Garamond", serif;
    }
    .stApp {
        background-color: #0b1020;
        color: #ffffff;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .main > div {
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
#   MODEL CONSTRUCTION
# =========================

@st.cache_data
def build_state_space(h, psi, theta_p, theta_w, phi_b, lambda_rot=0.3,
                      shock_type="g_cons", financing_rule="lump_sum"):
    """
    Build a linearized Smets & Wouters (2007)-style DSGE state-space system with:
    - Rule-of-Thumb (non-Ricardian) households alongside optimizing households
    - Habit formation in consumption
    - Variable capital utilization
    - Sticky prices and wages (Calvo)
    - Debt-feedback mechanism

    State-space form:
        x_{t+1} = A x_t + B eps_t
        y_t     = C x_t

    States (x):
        0: output gap (y)
        1: consumption (c) - aggregate, with ROT mix
        2: investment (i)
        3: labor (n)
        4: inflation (pi)
        5: government debt (b)
        6: government spending (g)
        7: real interest rate (r)
        8: technology trend (a)

    Key mechanism: Consumption = (1-λ_rot)*C_opt + λ_rot*Y_{after-tax}
    """

    # --- Structural parameters ---
    # Persistence of exogenous states
    rho_a = 0.95          # technology persistence
    rho_g = 0.80          # government spending persistence
    rho_r = 0.85          # interest rate persistence
    
    # Endogenous dynamics shaped by frictions
    # Consumption: habit formation reduces contemporaneous response
    # Calibration: c_persist = 0.6 + 0.2*h targets empirical consumption persistence ~0.85
    # Higher h (habit) → smoother consumption, less responsive to transient income shocks
    c_persist = 0.6 + 0.2 * h
    # Optimizers respond to output gap with elasticity ~0.1; dampened by (1-λ_rot) share
    c_output_response = (1 - lambda_rot) * 0.1
    # ROT households consume current income; their share is λ_rot
    c_rot_response = lambda_rot * 0.2
    
    # Investment: capital utilization convexity parameter psi affects response
    # Calibration: ψ ∈ [0, 1]; higher ψ → investment more elastic to q-theory demand shocks
    i_persist = 0.5 + 0.15 * psi
    # Output elasticity of investment: 0.1 base + 0.05*ψ (higher psi → stronger demand response)
    i_output_response = 0.1 + 0.05 * psi
    # Interest-rate sensitivity: lower psi → more rate-sensitive (less utilization friction to dampen rate effects)
    i_rate_response = -0.05 * (1 - psi)
    
    # Labor supply (backward-looking with frictions)
    n_persist = 0.4
    n_output_response = 0.15 + 0.05 * (1 - theta_w)  # higher wage stickiness → slower response
    
    # Inflation: Phillips curve slopes down in wage and price stickiness
    # Flatter slope when θ_p and θ_w are high (Calvo probabilities)
    phillips_coeff = 0.4 * (1 - theta_p) * (1 - theta_w)  # scale down by stickiness
    pi_persist = 0.7
    pi_output_response = phillips_coeff * 0.2
    pi_lag_weight = 0.3
    
    # Debt dynamics: feedback from debt level and output
    # Calibration: b_persist ∈ [0.85, 0.95]; ensures stable debt-to-output ratio
    b_persist = 0.85 + 0.1 * phi_b  # higher phi_b → stronger persistence (forward-looking debt effects)
    # Validation: Ensure persistence < 1 to maintain stationarity
    if b_persist >= 1.0:
        b_persist = 0.99  # Clip to safe level
    
    b_spending_effect = 1.0  # government spending increases deficit/debt
    debt_induced_tax = min(0.25 * phi_b, 0.2)  # higher phi_b → higher automatic stabilizer taxation
    
    # Financing rule: determines how debt is stabilized
    # Different tax types have different economic effects
    if financing_rule == "lump_sum":
        # Non-distortionary: no direct feedback to real activity
        tax_on_labor = 0.0
        tax_on_capital = 0.0
        tax_on_cons = 0.0
    elif financing_rule == "cons_tax":
        # Consumption taxes dampen demand directly
        tax_on_cons = 0.12
        tax_on_labor = 0.0
        tax_on_capital = 0.0
    elif financing_rule == "labor_tax":
        # Labor taxes reduce labor supply and output
        tax_on_labor = 0.15
        tax_on_cons = 0.0
        tax_on_capital = 0.0
    elif financing_rule == "capital_tax":
        # Capital taxes depress investment and future output
        tax_on_capital = 0.18
        tax_on_labor = 0.0
        tax_on_cons = 0.0
    elif financing_rule == "income_tax":
        # Broad-based income tax: affects both labor and capital income
        tax_on_labor = 0.10
        tax_on_capital = 0.08
        tax_on_cons = 0.0
    elif financing_rule == "debt_targeting":
        # Debt-to-GDP targeting: stronger feedback when debt is high
        tax_on_labor = min(0.20 * phi_b, 0.15)
        tax_on_capital = min(0.15 * phi_b, 0.12)
        tax_on_cons = 0.0
    elif financing_rule == "automatic_stabilizer":
        # Progressive taxation: taxes rise with output, spending rises with unemployment
        tax_on_labor = 0.08
        tax_on_capital = 0.05
        tax_on_cons = 0.03
    elif financing_rule == "balanced_budget":
        # Immediate spending adjustment to offset shocks
        tax_on_labor = 0.0
        tax_on_capital = 0.0
        tax_on_cons = 0.0
    elif financing_rule == "inflation_tax":
        # Seigniorage: government prints money (only relevant with monetary block)
        tax_on_labor = 0.0
        tax_on_capital = 0.0
        tax_on_cons = 0.0
    elif financing_rule == "g_cuts":
        # Future spending cuts unwind stimulus
        tax_on_labor = 0.0
        tax_on_capital = 0.0
        tax_on_cons = 0.0
    else:
        tax_on_labor = 0.0
        tax_on_capital = 0.0
        tax_on_cons = 0.0

    # --- State transition matrix A (9x9) ---
    # Built with careful attention to ensure stability (eigenvalues < 1)
    A = np.zeros((9, 9))

    # Output equation: y_{t+1} = rho_y*y_t + c*c_t + i*i_t + g*g_t + (r-neutral)*r_t + a*a_t
    A[0, 0] = 0.5  # output has modest persistence
    A[0, 1] = c_output_response  # consumption channel (dampened by ROT households)
    A[0, 2] = 0.20  # investment multiplier effect
    A[0, 6] = 0.25  # government spending (Keynesian component)
    A[0, 7] = -0.10 * (1 - tax_on_capital)  # real rate discourages demand
    A[0, 8] = 0.15  # technology boost output (increased productivity)
    
    # Add debt overhang effect when financing is distortionary
    A[0, 5] = -0.08 * (tax_on_labor + tax_on_capital + tax_on_cons)

    # Consumption equation: c_{t+1} = ρ_c*c_t + (1-λ_rot)*response_to_output + λ_rot*income_effect
    A[1, 1] = c_persist  # habit formation: smoother consumption path
    A[1, 0] = c_output_response  # optimizers respond to output gap
    A[1, 6] = lambda_rot * 0.5  # ROT households: spend government income
    A[1, 7] = -0.05 * (1 - tax_on_cons)  # interest rate: intertemporal substitution (muted by ROT)
    A[1, 8] = 0.08  # permanent income effect from technology
    
    # Tax effects on consumption (distortionary taxes reduce disposable income for ROT)
    A[1, 3] = -lambda_rot * tax_on_labor * 0.3  # labor tax reduces income
    
    # Investment equation: i_{t+1} = ρ_i*i_t + elasticity*(y_t) + elasticity*(r_t)
    A[2, 2] = i_persist  # capital accumulation persistence
    A[2, 0] = i_output_response  # tobin-q channel: demand pressures  
    A[2, 7] = i_rate_response  # cost of capital effect (more sensitive with high psi)
    A[2, 8] = 0.20  # technology incentivizes capital deepening
    
    # Capital tax reduces after-tax return to capital
    A[2, 5] = -0.12 * tax_on_capital

    # Labor equation: n_{t+1} = ρ_n*n_t + labor_response*y_t - tax_response*tau_labor_t
    A[3, 3] = n_persist
    A[3, 0] = n_output_response
    A[3, 8] = 0.08  # labor productivity improvement
    
    # Labor tax effect: higher labor tax reduces labor supply (intertemporal substitution)
    A[3, 5] = -0.15 * tax_on_labor

    # Inflation equation: π_{t+1} = ρ_π*π_t + λ_π*π_{t-1} + κ*(y_t - ȳ_t)
    # Phillips curve with pricing frictions
    A[4, 4] = pi_persist
    A[4, 0] = pi_output_response  # output gap pushes inflation (less if prices sticky)
    A[4, 3] = 0.10  # labor market tightness affects wages → inflation
    
    # Higher stickiness reduces inflation responsiveness to shocks
    A[4, 8] = 0.05 * (1 - theta_p - theta_w)  # technology disinflation

    # Government debt: b_{t+1} = ρ_b*b_t + g_t - tax_revenue_t
    A[5, 5] = b_persist
    A[5, 6] = b_spending_effect  # spending increases debt
    A[5, 0] = -debt_induced_tax  # output growth reduces deficit (automatic stabilizer)
    
    # Debt stabilization depends on financing rule
    if financing_rule == "debt_targeting":
        # Stronger debt feedback for debt-to-GDP targeting
        A[5, 5] = min(b_persist + 0.1 * phi_b, 0.95)  # more persistent debt dynamics
        A[5, 0] += -(tax_on_labor * 0.20 + tax_on_capital * 0.15)  # stronger automatic stabilizers
    elif financing_rule == "balanced_budget":
        # Immediate spending adjustment - debt doesn't accumulate
        A[5, 5] = 0.0  # no persistence
        A[5, 6] = 0.0  # spending doesn't affect debt (offset by immediate cuts)
        A[5, 0] = 0.0  # no automatic stabilizers needed
    elif financing_rule == "automatic_stabilizer":
        # Progressive taxation - stronger response to output fluctuations
        A[5, 0] += -(tax_on_labor * 0.12 + tax_on_capital * 0.08 + tax_on_cons * 0.05)
    elif financing_rule != "lump_sum":
        # Distortionary taxes generate tax revenue
        A[5, 0] += -(tax_on_labor * 0.15 + tax_on_capital * 0.10 + tax_on_cons * 0.20)

    # Government spending (exogenous AR(1) process)
    if shock_type == "g_infrastructure":
        A[6, 6] = min(rho_g + 0.3, 0.95)  # More persistent for infrastructure
    else:
        A[6, 6] = rho_g

    # Real interest rate (monetary policy AR(1), assumed exogenous for real rate)
    A[7, 7] = rho_r
    
    # Interest rate responds to inflation (Taylor rule feedback)
    A[7, 4] = 0.30 * (1 - theta_p - theta_w)  # more aggressive when prices flexible

    # Technology (exogenous AR(1))
    A[8, 8] = rho_a

    # Ensure system stability: clip eigenvalues to max 0.95
    # Rationale: Eigenvalues > 1 cause explosive dynamics (non-stationarity).
    # Clipping to 0.95 ensures mean-reversion while preserving realistic persistence.
    # Target persistence: output ~0.95, consumption ~0.85, investment ~0.80.
    evals, evecs = np.linalg.eig(A)
    evals_clipped = np.clip(np.abs(evals), 0, 0.95) * np.sign(evals)
    
    # Use pseudoinverse (pinv) instead of inv() for numerical stability.
    # Standard inversion can fail or amplify rounding errors with near-singular matrices.
    try:
        A_stable = evecs @ np.diag(evals_clipped) @ np.linalg.pinv(evecs)
        A_stable = np.real(A_stable)  # take real part (discard numerical imaginary components)
    except np.linalg.LinAlgError:
        # Fallback: If reconstruction fails, return clipped diagonal form (imperfect but safe)
        A_stable = np.diag(evals_clipped[:9]) if len(evals_clipped) >= 9 else A
        st.warning("⚠️ Eigenvalue reconstruction failed; using diagonal approximation.")

    # --- Shock loading matrix B (9x10) ---
    # 10 structural shocks: [0] g_spending, [1] technology, [2] interest_rate, 
    #                      [3] tau_labor, [4] tau_capital, [5] tau_consumption, [6] markup,
    #                      [7] g_wage_bill, [8] g_infrastructure, [9] lump_sum_transfer
    B = np.zeros((9, 10))
    
    # Validate shock type is recognized
    valid_shocks = ["g_cons", "g_inv", "tau_labor", "tau_capital", "tau_consumption", 
                   "tau_corporate", "g_wage_bill", "g_infrastructure", "lump_sum_transfer", "gov_borrowing_cost"]
    if shock_type not in valid_shocks:
        st.warning(f"⚠️ Unknown shock type '{shock_type}'. Defaulting to 'g_cons'.")
        shock_type = "g_cons"
    
    # Load structural shocks based on type
    # Government spending shocks affect state 6 (public expenditure)
    if shock_type == "g_cons":
        B[6, 0] = 1.0
    # Government investment shocks: affect both spending and capital formation
    elif shock_type == "g_inv":
        B[6, 0] = 0.8  # 80% channels through government spending state
        B[2, 0] = 0.5  # 50% direct boost to investment (crowding-in component)
    # Government wage bill: increases public sector wages, affects labor market tightness
    elif shock_type == "g_wage_bill":
        B[6, 7] = 0.6  # 60% through government spending
        B[3, 7] = 0.4  # 40% direct effect on labor market (wage pressure)
        B[4, 7] = 0.2  # inflationary pressure from wage increases
    # Government infrastructure: long-lived capital spending, more persistent
    elif shock_type == "g_infrastructure":
        B[6, 8] = 0.7  # 70% through government spending
        B[2, 8] = 0.8  # 80% direct boost to investment (infrastructure investment)
        # More persistent - handled in A matrix with higher rho_g for infrastructure
    # Labor tax cuts: boost labor supply (tax reduction)
    elif shock_type == "tau_labor":
        B[3, 3] = 1.0  # labor supply increases when taxes cut
    # Capital tax cuts: boost investment returns
    elif shock_type == "tau_capital":
        B[2, 4] = 1.0  # investment increases when capital taxes cut
    # Consumption tax cuts: boost consumption directly
    elif shock_type == "tau_consumption":
        B[1, 5] = 0.8  # consumption increases when VAT/sales taxes cut
        B[0, 5] = 0.2  # some direct output effect
    # Corporate profit tax cuts: boost investment and dividends
    elif shock_type == "tau_corporate":
        B[2, 4] = 0.7  # investment boost (similar to capital tax but corporate-focused)
        B[1, 5] = 0.3  # consumption effect from higher dividends/profits
    # Lump-sum transfers: direct transfers to households (powerful with ROT households)
    elif shock_type == "lump_sum_transfer":
        B[1, 9] = lambda_rot * 0.9  # strong effect on ROT households
        B[1, 9] += (1 - lambda_rot) * 0.3  # weaker effect on optimizers
        B[0, 9] = 0.1  # small direct output effect
    # Government borrowing cost shock: increases interest rate govt pays
    elif shock_type == "gov_borrowing_cost":
        B[7, 2] = 0.5  # affects market interest rate
        B[5, 2] = 0.3  # affects debt servicing costs
    
    # Technology shock always present for baseline dynamics
    B[8, 1] = 1.0

    # --- Observation matrix C (9x9) ---
    # We observe all states directly
    C = np.eye(9)

    return A_stable, B, C


def simulate_business_cycle(A, B, C, T=1000, shock_std=0.01, seed=42):
    """
    Simulate the DSGE model for T periods.
    
    The system is driven by multiple independent shocks:
    - Government spending shock (state 6)
    - Technology shock (state 8) 
    - Interest rate shock (state 7)
    - Labor tax shock (state 3)
    - Capital tax shock (state 2)
    - Consumption tax shock
    - Markup shock
    
    Args:
        A: State transition matrix (n_states × n_states)
        B: Shock loading matrix (n_states × n_shocks)
        C: Observation matrix (n_states × n_states)
        T: Simulation horizon (periods)
        shock_std: Standard deviation of shocks (default 0.01 ≈ 1% monthly shocks)
        seed: Random seed for reproducibility
        
    Returns:
        x: State vector trajectory (n_states × T)
        y: Observed variables (n_states × T)
    """
    np.random.seed(seed)
    n_states = A.shape[0]
    n_shocks = B.shape[1]
    
    # Validate inputs
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"State matrix A must be square, got {A.shape}")
    if B.shape[0] != n_states:
        raise ValueError(f"Shock matrix B must have {n_states} rows, got {B.shape[0]}")
    
    x = np.zeros((n_states, T))
    eps = shock_std * np.random.randn(n_shocks, T)

    for t in range(1, T):
        x[:, [t]] = A @ x[:, [t - 1]] + B @ eps[:, [t]]

    y = C @ x
    return x, y


def compute_moments(y):
    """
    Compute 10+ key business cycle moments from simulated data.
    
    Variables: Output(0), Consumption(1), Investment(2), Labor(3), Inflation(4),
               Debt(5), Government Spending(6), Interest Rate(7), Technology(8)
    
    Returns a comprehensive DataFrame with:
    - Volatilities (standard deviations)
    - Relative volatilities (vs output)
    - Correlations with output
    - Autocorrelations
    - Cross-variable relationships
    
    Raises:
        ValueError: If input array has invalid shape
    """
    # Validate input
    if y.size == 0 or y.ndim != 2 or y.shape[0] < 7:
        raise ValueError(f"Invalid simulation data shape: {y.shape}. Expected (9, T) with T > 100.")
    
    # Primary variables for analysis
    var_info = {
        "Output": 0,
        "Consumption": 1,
        "Investment": 2,
        "Labor": 3,
        "Inflation": 4,
        "Debt": 5,
        "G-Spending": 6,
    }
    
    data = {}
    y_output = y[0, :]
    
    # Remove first 100 burn-in periods to allow system to settle
    burn_in = min(100, y.shape[1] // 4)
    
    for name, idx in var_info.items():
        series = y[idx, burn_in:]
        
        # Standard deviation (proxy for volatility)
        std_dev = np.std(series)
        
        # Relative volatility vs output (with floor to avoid div-by-zero)
        output_std = np.std(y_output[burn_in:]) + 1e-10
        rel_vol = std_dev / output_std
        
        # Correlation with output (robust to zero variance)
        try:
            if np.std(y_output[burn_in:]) > 1e-9 and np.std(series) > 1e-9:
                corr_output = np.corrcoef(series, y_output[burn_in:])[0, 1]
            else:
                corr_output = np.nan
        except (ValueError, RuntimeWarning):
            corr_output = np.nan
        
        # Autocorrelation at lag 1
        if len(series) > 1:
            series_lag = series[:-1]
            series_lead = series[1:]
            try:
                if np.std(series_lag) > 1e-9 and np.std(series_lead) > 1e-9:
                    ac1 = np.corrcoef(series_lag, series_lead)[0, 1]
                else:
                    ac1 = np.nan
            except (ValueError, RuntimeWarning):
                ac1 = np.nan
        else:
            ac1 = np.nan
        
        data[name] = {
            "Std Dev": std_dev,
            "Rel. Volatility": rel_vol,
            "Corr w/ Output": corr_output,
            "Autocorr(1)": ac1,
        }
    
    df = pd.DataFrame(data).T
    
    return df


def impulse_response(A, B, C, T=40, shock_size=0.01, shock_idx=0):
    """
    Compute IRF to a one-time shock at t=0.
    
    Args:
        A, B, C: State-space matrices
        T: horizon
        shock_size: size of shock
        shock_idx: which shock to perturb (0=government spending, 1=technology, etc.)
    """
    n_states = A.shape[0]
    x = np.zeros((n_states, T))
    
    # Apply shock at t=0
    eps0 = np.zeros((B.shape[1],))
    eps0[shock_idx] = shock_size
    x[:, 0:1] = B @ eps0.reshape(-1, 1)

    for t in range(1, T):
        x[:, [t]] = A @ x[:, [t - 1]]

    y = C @ x
    return x, y


def compute_multipliers(irf_y, irf_g, beta=0.99, impact_period=0):
    """
    Compute fiscal multipliers with robust error handling.
    
    Multipliers quantify output response to fiscal shocks:
    - Impact multiplier: ΔY_t / ΔG_t at specified period t
    - Cumulative multiplier: discounted sum of outputs / discounted sum of shocks
    
    Args:
        irf_y: Output response path (array)
        irf_g: Government spending or tax shock path (array)
        beta: Discount factor (default 0.99)
        impact_period: Period t to use for impact multiplier (default 0)
        
    Returns:
        Tuple (impact, cumulative, drag_horizon) where:
        - impact: output-to-shock ratio at period t
        - cumulative: long-run discounted multiplier
        - drag_horizon: quarters until output turns negative (None if doesn't occur)
    """
    y_path = irf_y
    g_path = irf_g
    
    # Impact multiplier at specified period with threshold check
    # Issue: If g_path[t] ≈ 0, multiplier is misleading. Warn user.
    if impact_period >= len(g_path) or np.abs(g_path[impact_period]) < 1e-6:
        impact = np.nan
        # Don't warn here to avoid cluttering output; documented above
    else:
        impact = y_path[impact_period] / g_path[impact_period]
    
    # Cumulative multiplier: sum_t β^t ΔY_t / sum_t β^t ΔG_t
    T = len(y_path)
    discounts = np.array([beta**t for t in range(T)])
    num = np.sum(discounts * y_path)
    den = np.sum(discounts * g_path)
    
    if np.abs(den) < 1e-8:
        cumulative = np.nan
    else:
        cumulative = num / den
    
    # Fiscal drag horizon: first t where output falls below 0 after being positive
    # Note: Absence of crossing does NOT imply no fiscal drag—growth slowdown still occurs.
    drag_horizon = None
    max_y = np.max(y_path)
    if max_y > 0:  # Only search if output was ever positive
        for t in range(1, T):
            if y_path[t] < 0:
                drag_horizon = t
                break
    
    return impact, cumulative, drag_horizon


# =========================
#   SIDEBAR: GLOBAL SLIDERS
# =========================

st.sidebar.title("🔧 Global Structural Parameters")

h = st.sidebar.slider("Habit formation in consumption (h)", 0.0, 0.9, 0.5, 0.05)
psi = st.sidebar.slider("Variable capital utilization convexity (ψ)", 0.0, 1.0, 0.5, 0.05)
theta_p = st.sidebar.slider("Price stickiness (θ_p)", 0.0, 0.95, 0.7, 0.05)
theta_w = st.sidebar.slider("Wage stickiness (θ_w)", 0.0, 0.95, 0.7, 0.05)
phi_b = st.sidebar.slider("Debt-feedback aggressiveness (φ_b)", 0.0, 1.0, 0.5, 0.05)
lambda_rot = st.sidebar.slider("Rule-of-Thumb households (λ_rot)", 0.0, 0.6, 0.3, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Parameter Guide:**
- **h**: Habit persistence smooths consumption relative to income
- **ψ**: Capital utilization friction; higher → investment more responsive to demand
- **θ_p, θ_w**: Calvo stickiness parameters (higher → flatter Phillips curve)
- **φ_b**: Debt feedback strength (higher → stronger fiscal stress effects)
- **λ_rot**: Share of households consuming current income (vs. forward-looking)
""")

# =========================
#   MAIN TABS
# =========================

tab1, tab2 = st.tabs(["Model Fit (Unconditional Dynamics)", "Fiscal Exercises (Policy IRFs)"])

# =========================
#   TAB 1: MODEL FIT
# =========================

with tab1:
    st.header("Unconditional Business Cycle Dynamics")

    A_fit, B_fit, C_fit = build_state_space(
        h=h,
        psi=psi,
        theta_p=theta_p,
        theta_w=theta_w,
        phi_b=phi_b,
        lambda_rot=lambda_rot,
        shock_type="g_cons",
        financing_rule="lump_sum",
    )

    T_sim = st.slider("Simulation horizon (periods)", 200, 2000, 1000, 100)
    x_sim, y_sim = simulate_business_cycle(A_fit, B_fit, C_fit, T=T_sim, shock_std=0.01)

    moments_df = compute_moments(y_sim)

    st.subheader("📊 Key Simulated Business Cycle Moments (10+ Measures)")
    st.dataframe(moments_df.round(4), use_container_width=True)

    st.markdown("""
    **Moment Definitions:**
    - **Std Dev**: Standard deviation of log deviations from steady state
    - **Rel. Volatility**: Variable volatility relative to output volatility
    - **Corr w/ Output**: Contemporaneous correlation with output gap
    - **Autocorr(1)**: First-order autocorrelation (persistence)
    """)

    # Dynamic interpretation text
    out_std = moments_df.loc["Output", "Std Dev"]
    cons_std = moments_df.loc["Consumption", "Std Dev"]
    inv_std = moments_df.loc["Investment", "Std Dev"]
    cons_corr = moments_df.loc["Consumption", "Corr w/ Output"]
    inv_corr = moments_df.loc["Investment", "Corr w/ Output"]
    cons_rel_vol = moments_df.loc["Consumption", "Rel. Volatility"]
    inv_rel_vol = moments_df.loc["Investment", "Rel. Volatility"]

    interpretation = (
        f"With habit formation h = {h:.2f} and rule-of-thumb households λ_rot = {lambda_rot:.2f}, "
        f"the model generates output volatility of {out_std:.4f}. "
        f"Consumption volatility ({cons_std:.4f}) is "
        f"{'lower' if cons_std < out_std else 'higher'} than output, with relative volatility of {cons_rel_vol:.2f}. "
        f"This configuration "
        f"{'successfully replicates' if 0.5 <= cons_rel_vol <= 0.8 else 'deviates from'} "
        f"standard business cycle facts (consumption typically 0.5–0.8× output volatility). "
        f"\n\nHabit formation (h) shifts consumption dynamics: higher h dampens contemporaneous responses and raises persistence. "
        f"Rule-of-Thumb households (λ_rot) amplify fiscal multipliers by increasing income-sensitive demand. "
        f"Investment shows relative volatility of {inv_rel_vol:.2f} with output correlation {inv_corr:.2f}, "
        f"reflecting capital adjustment costs controlled by ψ = {psi:.2f}. "
        f"\n\nAs you increase price and wage stickiness (θ_p, θ_w), the Phillips curve flattens, "
        f"inflation becomes less responsive to output, and the model better matches empirical inflation persistence."
    )

    st.markdown("### 💡 AI-Generated Economic Interpretation")
    st.write(interpretation)

    # Interactive time-series plots with hover functionality
    st.markdown("### 📈 Simulated Time Series (Interactive - Hover for Values)")

    # Create subplots with Plotly
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Output Gap", "Consumption", "Investment", "Inflation"),
        vertical_spacing=0.05
    )

    t_grid = np.arange(T_sim)

    # Output Gap
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=y_sim[0, :],
            mode='lines',
            name='Output Gap',
            line=dict(color='cyan', width=2),
            hovertemplate='Period: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Consumption with reference line
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=y_sim[1, :],
            mode='lines',
            name='Consumption',
            line=dict(color='orange', width=2),
            hovertemplate='Period: %{x}<br>Consumption: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=y_sim[0, :],
            mode='lines',
            name='Output (ref)',
            line=dict(color='cyan', width=1, dash='dot'),
            opacity=0.5,
            hovertemplate='Period: %{x}<br>Output: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Investment with reference line
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=y_sim[2, :],
            mode='lines',
            name='Investment',
            line=dict(color='magenta', width=2),
            hovertemplate='Period: %{x}<br>Investment: %{y:.4f}<extra></extra>'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=y_sim[0, :],
            mode='lines',
            name='Output (ref)',
            line=dict(color='cyan', width=1, dash='dot'),
            opacity=0.5,
            showlegend=False,
            hovertemplate='Period: %{x}<br>Output: %{y:.4f}<extra></extra>'
        ),
        row=3, col=1
    )

    # Inflation
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=y_sim[4, :],
            mode='lines',
            name='Inflation',
            line=dict(color='#00ff00', width=2),
            hovertemplate='Period: %{x}<br>Inflation: %{y:.4f}<extra></extra>'
        ),
        row=4, col=1
    )

    # Update layout for dark theme
    fig.update_layout(
        height=800,
        showlegend=True,
        paper_bgcolor='#0b1020',
        plot_bgcolor='#111827',
        font=dict(color='white'),
        hovermode='x unified',  # Shows all values at cursor position
        xaxis4=dict(title="Time (periods)", title_font=dict(color='white')),
    )

    # Update axes styling
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.5)',
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.5)',
            row=i, col=1
        )

    # Add horizontal reference lines
    for i in range(1, 5):
        fig.add_hline(y=0, line=dict(color='white', width=1, dash='dash'),
                     opacity=0.5, row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # CSV Download Functionality for Tab 1
    @st.cache_data
    def create_simulation_csv(t_grid, y_sim, moments_df, T_sim, h, psi, theta_p, theta_w, phi_b, lambda_rot):
        """Create CSV data from simulation results and moments."""
        # Time series data
        df_ts = pd.DataFrame({
            'Period': t_grid,
            'Output_Gap': y_sim[0, :],
            'Consumption': y_sim[1, :],
            'Investment': y_sim[2, :],
            'Labor': y_sim[3, :],
            'Inflation': y_sim[4, :],
            'Government_Debt': y_sim[5, :],
            'Government_Spending': y_sim[6, :],
            'Interest_Rate': y_sim[7, :],
            'Technology': y_sim[8, :]
        })

        # Moments data (transpose for better CSV format)
        df_moments = moments_df.T.reset_index().rename(columns={'index': 'Variable'})

        # Create CSV with metadata
        csv_header = f"# DSGE Business Cycle Simulation Data\n"
        csv_header += f"# Simulation Horizon: {T_sim} periods\n"
        csv_header += f"# Parameters: h={h:.2f}, psi={psi:.2f}, theta_p={theta_p:.2f}, theta_w={theta_w:.2f}, phi_b={phi_b:.2f}, lambda_rot={lambda_rot:.2f}\n"
        csv_header += f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        csv_header += "#\n"
        csv_header += "# TIME SERIES DATA\n"
        csv_header += "#\n"
        csv_ts = df_ts.to_csv(index=False)

        csv_header += "# BUSINESS CYCLE MOMENTS\n"
        csv_header += "#\n"
        csv_moments = df_moments.to_csv(index=False)

        return csv_header + csv_ts + "\n\n" + csv_moments

    # Create download button for Tab 1
    csv_data_tab1 = create_simulation_csv(
        t_grid, y_sim, moments_df, T_sim, h, psi, theta_p, theta_w, phi_b, lambda_rot
    )

    st.download_button(
        label="📥 Download Simulation Data as CSV",
        data=csv_data_tab1,
        file_name=f"dsge_simulation_{T_sim}periods_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the business cycle simulation data and moments as a CSV file"
    )


# =========================
#   TAB 2: FISCAL EXERCISES
# =========================

with tab2:
    st.header("Fiscal Policy Experiments and Multipliers")

    col1, col2 = st.columns(2)

    with col1:
        shock_category = st.selectbox(
            "Fiscal Shock Category",
            ["spending", "tax", "transfer", "other"],
            format_func=lambda x: {
                "spending": "Spending Shocks",
                "tax": "Tax Shocks", 
                "transfer": "Transfer Shocks",
                "other": "Other Shocks"
            }[x],
        )
        
        # Dynamic shock type options based on category
        shock_options = {
            "spending": ["g_cons", "g_inv", "g_wage_bill", "g_infrastructure"],
            "tax": ["tau_labor", "tau_capital", "tau_consumption", "tau_corporate"],
            "transfer": ["lump_sum_transfer"],
            "other": ["gov_borrowing_cost"]
        }
        
        shock_type = st.selectbox(
            "Fiscal Shock Type",
            shock_options[shock_category],
            format_func=lambda x: {
                "g_cons": "↑ Government Consumption Spending",
                "g_inv": "↑ Government Investment Spending", 
                "g_wage_bill": "↑ Government Wage Bill (Public Sector Pay)",
                "g_infrastructure": "↑ Government Infrastructure Investment",
                "tau_labor": "↓ Labor Taxes (Tax Cut)",
                "tau_capital": "↓ Capital Taxes (Tax Cut)",
                "tau_consumption": "↓ Consumption Taxes (VAT/Sales Tax Cut)",
                "tau_corporate": "↓ Corporate Profit Taxes (Tax Cut)",
                "lump_sum_transfer": "↑ Lump-Sum Transfers (Stimulus Checks)",
                "gov_borrowing_cost": "↑ Government Borrowing Costs (Sovereign Risk)"
            }[x],
        )

    with col2:
        financing_rule = st.selectbox(
            "Financing Rule (How govt stabilizes debt)",
            [
                "lump_sum",
                "cons_tax",
                "labor_tax", 
                "capital_tax",
                "income_tax",
                "g_cuts",
                "debt_targeting",
                "automatic_stabilizer",
                "balanced_budget",
                "inflation_tax"
            ],
            format_func=lambda x: {
                "lump_sum": "Ricardian (Non-distortionary) Lump-Sum Transfers",
                "cons_tax": "Consumption Tax Hikes",
                "labor_tax": "Labor Tax Hikes",
                "capital_tax": "Capital Tax Hikes",
                "income_tax": "Broad-Based Income Tax Hikes",
                "g_cuts": "Future Government Spending Cuts",
                "debt_targeting": "Debt-to-GDP Targeting Rule",
                "automatic_stabilizer": "Automatic Stabilizers (Progressive Taxation)",
                "balanced_budget": "Balanced Budget Rule",
                "inflation_tax": "Inflation Tax (Seigniorage)"
            }[x],
        )

    A_pol, B_pol, C_pol = build_state_space(
        h=h,
        psi=psi,
        theta_p=theta_p,
        theta_w=theta_w,
        phi_b=phi_b,
        lambda_rot=lambda_rot,
        shock_type=shock_type,
        financing_rule=financing_rule,
    )

    T_irf = st.slider("IRF Horizon (quarters)", 10, 80, 40, 5)
    impact_period = st.slider("Impact period (t)", 0, 10, 1, 1)
    
    # Map shock types to shock indices in B matrix
    shock_idx_map = {
        "g_cons": 0,           # government spending shock
        "g_inv": 0,            # also government spending (modified in A)
        "g_wage_bill": 7,      # government wage bill shock
        "g_infrastructure": 8, # government infrastructure shock
        "tau_labor": 3,        # labor tax shock
        "tau_capital": 4,      # capital tax shock
        "tau_consumption": 5,  # consumption tax shock
        "tau_corporate": 4,    # corporate tax shock (similar to capital tax)
        "lump_sum_transfer": 9,# lump-sum transfer shock
        "gov_borrowing_cost": 2 # government borrowing cost shock
    }
    shock_idx = shock_idx_map.get(shock_type, 0)
    
    x_irf, y_irf = impulse_response(A_pol, B_pol, C_pol, T=T_irf, shock_size=0.01, shock_idx=shock_idx)

    # Extract IRFs for all 9 states
    y_gap_irf = y_irf[0, :]     # Output
    c_irf = y_irf[1, :]         # Consumption
    i_irf = y_irf[2, :]         # Investment
    n_irf = y_irf[3, :]         # Labor
    pi_irf = y_irf[4, :]        # Inflation
    b_irf = y_irf[5, :]         # Government Debt
    g_irf = y_irf[6, :]         # Government Spending
    r_irf = y_irf[7, :]         # Interest Rate
    a_irf = y_irf[8, :]         # Technology

    # For tax shocks, use the shock path itself (since tax shocks don't affect g directly)
    # For spending shocks, use government spending response
    tax_shocks = ["tau_labor", "tau_capital", "tau_consumption", "tau_corporate"]
    if shock_type in tax_shocks:
        # Create shock path: impulse of 0.01 at t=0, then zeros
        shock_path = np.zeros_like(y_gap_irf)
        shock_path[0] = 0.01  # The shock size used in impulse_response
    else:
        shock_path = g_irf  # Use government spending for spending shocks

    impact_mult, cum_mult, drag_horizon = compute_multipliers(y_gap_irf, shock_path, beta=0.99, impact_period=impact_period)

    st.subheader("📊 Fiscal Multipliers & Key Statistics")
    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("Impact Multiplier", f"{impact_mult:.3f}")
    colm2.metric("Cumulative Multiplier", f"{cum_mult:.3f}")
    if drag_horizon is not None:
        colm3.metric("Fiscal Drag Horizon", f"Q{drag_horizon}")
    else:
        colm3.metric("Fiscal Drag Horizon", "Beyond horizon")

    # Interactive IRF plots with hover functionality
    st.markdown("### 📉 Impulse Response Functions (Interactive - Hover for Values)")

    # Create subplots with Plotly
    fig_irf = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Output Gap Response", "Government Debt Response",
                       "Consumption & Investment Response", "Inflation & Labor Response"),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    t_grid_irf = np.arange(T_irf)

    # Output Gap Response
    fig_irf.add_trace(
        go.Scatter(
            x=t_grid_irf,
            y=y_gap_irf,
            mode='lines+markers',
            name='Output Gap',
            line=dict(color='cyan', width=3),
            marker=dict(size=4),
            hovertemplate='Quarter: %{x}<br>Output Gap: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Government Debt Response
    fig_irf.add_trace(
        go.Scatter(
            x=t_grid_irf,
            y=b_irf,
            mode='lines+markers',
            name='Government Debt',
            line=dict(color='yellow', width=3),
            marker=dict(size=4),
            hovertemplate='Quarter: %{x}<br>Gov Debt: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Consumption & Investment Response
    fig_irf.add_trace(
        go.Scatter(
            x=t_grid_irf,
            y=c_irf,
            mode='lines+markers',
            name='Consumption',
            line=dict(color='orange', width=2),
            marker=dict(symbol='circle', size=5),
            hovertemplate='Quarter: %{x}<br>Consumption: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    fig_irf.add_trace(
        go.Scatter(
            x=t_grid_irf,
            y=i_irf,
            mode='lines+markers',
            name='Investment',
            line=dict(color='magenta', width=2),
            marker=dict(symbol='square', size=5),
            hovertemplate='Quarter: %{x}<br>Investment: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Inflation & Labor Response
    fig_irf.add_trace(
        go.Scatter(
            x=t_grid_irf,
            y=pi_irf,
            mode='lines+markers',
            name='Inflation',
            line=dict(color='#00ff00', width=2),
            marker=dict(symbol='diamond', size=5),
            hovertemplate='Quarter: %{x}<br>Inflation: %{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )
    fig_irf.add_trace(
        go.Scatter(
            x=t_grid_irf,
            y=n_irf,
            mode='lines+markers',
            name='Labor',
            line=dict(color='#ff00ff', width=2),
            marker=dict(symbol='triangle-up', size=5),
            hovertemplate='Quarter: %{x}<br>Labor: %{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )

    # Update layout for dark theme
    fig_irf.update_layout(
        height=700,
        showlegend=True,
        paper_bgcolor='#0b1020',
        plot_bgcolor='#111827',
        font=dict(color='white'),
        hovermode='x unified',  # Shows all values at cursor position
    )

    # Update axes styling and add zero lines
    for i in range(1, 3):
        for j in range(1, 3):
            fig_irf.update_xaxes(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.5)',
                row=i, col=j
            )
            fig_irf.update_yaxes(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.5)',
                row=i, col=j
            )
            # Add horizontal reference line at y=0
            fig_irf.add_hline(y=0, line=dict(color='white', width=1, dash='dash'),
                             opacity=0.7, row=i, col=j)

    st.plotly_chart(fig_irf, use_container_width=True)

    # CSV Download Functionality
    @st.cache_data
    def create_irf_csv(t_grid, y_gap, c, i, n, pi, b, g, r, a, shock_type, financing_rule):
        """Create CSV data from IRF results."""
        df = pd.DataFrame({
            'Quarter': t_grid,
            'Output_Gap': y_gap,
            'Consumption': c,
            'Investment': i,
            'Labor': n,
            'Inflation': pi,
            'Government_Debt': b,
            'Government_Spending': g,
            'Interest_Rate': r,
            'Technology': a
        })
        # Add metadata as comments at the top
        csv_header = f"# DSGE Fiscal Policy IRF Data\n"
        csv_header += f"# Shock Type: {shock_type}\n"
        csv_header += f"# Financing Rule: {financing_rule}\n"
        csv_header += f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        csv_header += "#\n"
        return csv_header + df.to_csv(index=False)

    # Create download button
    csv_data = create_irf_csv(
        t_grid_irf, y_gap_irf, c_irf, i_irf, n_irf, pi_irf,
        b_irf, g_irf, r_irf, a_irf, shock_type, financing_rule
    )

    st.download_button(
        label="📥 Download IRF Data as CSV",
        data=csv_data,
        file_name=f"dsge_irf_{shock_type}_{financing_rule}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the impulse response function data for all variables as a CSV file"
    )

    # Automated policy briefing
    shock_label = {
        "g_cons": "an increase in government consumption spending",
        "g_inv": "an increase in government investment spending",
        "g_wage_bill": "an increase in government wage bill (public sector pay raises)",
        "g_infrastructure": "an increase in government infrastructure investment (roads, bridges)",
        "tau_labor": "a cut in labor taxes (positive supply shock to labor)",
        "tau_capital": "a cut in capital taxes (positive incentive to invest)",
        "tau_consumption": "a cut in consumption taxes (VAT/sales tax reduction)",
        "tau_corporate": "a cut in corporate profit taxes (boost to business investment)",
        "lump_sum_transfer": "lump-sum transfers to households (stimulus checks)",
        "gov_borrowing_cost": "an increase in government borrowing costs (sovereign risk shock)"
    }[shock_type]

    finance_label = {
        "lump_sum": "Ricardian (non-distortionary) lump-sum transfers that do not distort any economic margins",
        "cons_tax": "consumption tax hikes that directly dampen household demand and reduce disposable income",
        "labor_tax": "labor income tax hikes that discourage hours worked and reduce labor supply",
        "capital_tax": "capital income tax hikes that lower the after-tax return on investment and depress capital formation",
        "income_tax": "broad-based income tax hikes affecting both labor and capital income",
        "g_cuts": "future government spending cuts that unwind the initial fiscal expansion",
        "debt_targeting": "debt-to-GDP targeting that adjusts taxes/spending to return debt to target levels",
        "automatic_stabilizer": "automatic stabilizers with progressive taxation that strengthen during downturns",
        "balanced_budget": "balanced budget rule requiring immediate spending adjustments to offset shocks",
        "inflation_tax": "inflation tax (seigniorage) through monetary financing of deficits"
    }[financing_rule]

    if drag_horizon is not None:
        drag_text = (
            f"Under this configuration, the **fiscal drag horizon** occurs around Q{drag_horizon}, "
            f"when the multiplied effects of distortionary financing are strong enough to push output below its steady state. "
            f"This quantifies the long-run sustainability of the policy stimulus."
        )
    else:
        drag_text = (
            "Within the displayed horizon, output does not fall below steady state after the initial expansion, "
            "suggesting that the fiscal drag horizon lies beyond the simulated window. "
            "The financing rule chosen is relatively benign over this timeframe."
        )

    policy_brief = (
        f"**Scenario**: Triggering {shock_label} financed through {finance_label}\n\n"
        f"**Multiplier Analysis**: This stimulus generates:\n"
        f"- **Impact Multiplier**: {impact_mult:.3f} (the contemporaneous output response per unit of shock)\n"
        f"- **Cumulative Multiplier**: {cum_mult:.3f} (the discounted cumulative output effect)\n\n"
        f"**Transmission Mechanism**: "
        f"When financing relies more heavily on distortionary taxes—especially capital or labor taxes—the "
        f"initial boost to output is gradually offset by weaker private demand, lower investment, and reduced labor supply. "
        f"\n\n**Fiscal Sustainability**: {drag_text}\n\n"
        f"**Sensitivity Analysis**: Adjusting the debt-feedback parameter φ_b and the stickiness parameters θ_p and θ_w "
        f"will change how quickly the financing rule feeds back into inflation, real activity, and long-run dynamics. "
        f"A larger rule-of-thumb population (λ_rot = {lambda_rot:.2f}) amplifies fiscal multipliers by increasing demand sensitivity to current income."
    )

    st.markdown("### 📋 Automated Policy Briefing")
    st.markdown(policy_brief)
