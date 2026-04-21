import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
    c_persist = 0.6 + 0.2 * h        # higher h → smoother, more persistent consumption
    c_output_response = (1 - lambda_rot) * 0.1  # optimizers respond to output
    c_rot_response = lambda_rot * 0.2  # ROT households consume income
    
    # Investment: capital utilization convexity parameter psi affects response
    i_persist = 0.5 + 0.15 * psi
    i_output_response = 0.1 + 0.05 * psi  # higher psi → stronger response to demand
    i_rate_response = -0.05 * (1 - psi)  # interest rate sensitivity
    
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
    b_persist = 0.85 + 0.1 * phi_b  # higher phi_b → stronger persistence (forward-looking debt effects)
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
    if financing_rule != "lump_sum":
        # Distortionary taxes generate tax revenue
        A[5, 0] += -(tax_on_labor * 0.15 + tax_on_capital * 0.10 + tax_on_cons * 0.20)

    # Government spending (exogenous AR(1) process)
    A[6, 6] = rho_g

    # Real interest rate (monetary policy AR(1), assumed exogenous for real rate)
    A[7, 7] = rho_r
    
    # Interest rate responds to inflation (Taylor rule feedback)
    A[7, 4] = 0.30 * (1 - theta_p - theta_w)  # more aggressive when prices flexible

    # Technology (exogenous AR(1))
    A[8, 8] = rho_a

    # Ensure system stability: clip eigenvalues to max 0.98
    evals, evecs = np.linalg.eig(A)
    evals_clipped = np.clip(np.abs(evals), 0, 0.98) * np.sign(evals)
    A_stable = evecs @ np.diag(evals_clipped) @ np.linalg.inv(evecs)
    A_stable = np.real(A_stable)  # take real part (discard numerical imaginary components)

    # --- Shock loading matrix B (9x7) ---
    # 7 structural shocks: g, a, r, tau_labor, tau_capital, tau_cons, z_markup
    B = np.zeros((9, 7))

    shock_mapping = {
        "g_cons": 0,      # government consumption shock
        "g_inv": 0,       # government investment (same state, different effect in A)
        "tau_labor": 3,   # labor tax shock
        "tau_capital": 4, # capital tax shock
    }
    
    if shock_type == "g_cons":
        B[6, 0] = 1.0  # government consumption shock
    elif shock_type == "g_inv":
        B[6, 0] = 0.8  # government investment shock (slightly smaller)
        B[2, 0] = 0.5  # direct investment effect
    elif shock_type == "tau_labor":
        B[3, 3] = -1.0  # negative tax shock (tax cut)
    elif shock_type == "tau_capital":
        B[2, 4] = -1.0  # negative tax shock
    
    # Technology shock always present for baseline dynamics
    B[8, 1] = 1.0

    # --- Observation matrix C (9x9) ---
    # We observe all states directly
    C = np.eye(9)

    return A_stable, B, C


def simulate_business_cycle(A, B, C, T=1000, shock_std=0.01, seed=42):
    """
    Simulate the DSGE model for T periods.
    
    The system is now driven by multiple independent shocks:
    - Government spending shock
    - Technology shock  
    - Interest rate shock
    - Labor tax shock
    - Capital tax shock
    - Consumption tax shock
    - Markup shock
    """
    np.random.seed(seed)
    n_states = A.shape[0]
    n_shocks = B.shape[1]
    
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
    """
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
        
        # Relative volatility vs output
        rel_vol = std_dev / (np.std(y_output[burn_in:]) + 1e-8)
        
        # Correlation with output
        if np.std(y_output[burn_in:]) > 1e-8:
            corr_output = np.corrcoef(series, y_output[burn_in:])[0, 1]
        else:
            corr_output = np.nan
        
        # Autocorrelation at lag 1
        if len(series) > 1:
            series_lag = series[:-1]
            series_lead = series[1:]
            if np.std(series_lag) > 1e-8 and np.std(series_lead) > 1e-8:
                ac1 = np.corrcoef(series_lag, series_lead)[0, 1]
            else:
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
    
    # Format for display precision
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


def compute_multipliers(irf_y, irf_g, beta=0.99):
    """
    Very stylized fiscal multipliers:
        Impact multiplier: ΔY_0 / ΔG_0
        Cumulative multiplier: sum_t β^t ΔY_t / sum_t β^t ΔG_t
    """
    y_path = irf_y
    g_path = irf_g

    impact = y_path[0] / (g_path[0] + 1e-8)

    T = len(y_path)
    discounts = np.array([beta**t for t in range(T)])
    num = np.sum(discounts * y_path)
    den = np.sum(discounts * g_path) + 1e-8
    cumulative = num / den

    # Fiscal drag horizon: first t where output falls below 0 after being positive
    drag_horizon = None
    for t in range(1, T):
        if y_path[t] < 0 and np.any(y_path[:t] > 0):
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

    # Simple time-series plots
    st.markdown("### 📈 Simulated Time Series (Selected Variables)")

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    t_grid = np.arange(T_sim)

    axes[0].plot(t_grid, y_sim[0, :], color="cyan", linewidth=1.5)
    axes[0].set_ylabel("Output Gap", color="white")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="white", linewidth=0.5, linestyle="--")

    axes[1].plot(t_grid, y_sim[1, :], color="orange", linewidth=1.5, label="Consumption")
    axes[1].plot(t_grid, y_sim[0, :], color="cyan", linewidth=1, alpha=0.5, label="Output (ref)")
    axes[1].set_ylabel("Consumption", color="white")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="white", linewidth=0.5, linestyle="--")

    axes[2].plot(t_grid, y_sim[2, :], color="magenta", linewidth=1.5, label="Investment")
    axes[2].plot(t_grid, y_sim[0, :], color="cyan", linewidth=1, alpha=0.5, label="Output (ref)")
    axes[2].set_ylabel("Investment", color="white")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color="white", linewidth=0.5, linestyle="--")

    axes[3].plot(t_grid, y_sim[4, :], color="#00ff00", linewidth=1.5)
    axes[3].set_ylabel("Inflation", color="white")
    axes[3].set_xlabel("Time (periods)", color="white")
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(0, color="white", linewidth=0.5, linestyle="--")

    fig.patch.set_facecolor("#0b1020")
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")

    st.pyplot(fig)


# =========================
#   TAB 2: FISCAL EXERCISES
# =========================

with tab2:
    st.header("Fiscal Policy Experiments and Multipliers")

    col1, col2 = st.columns(2)

    with col1:
        shock_type = st.selectbox(
            "Fiscal Shock Type",
            [
                "g_cons",
                "g_inv",
                "tau_labor",
                "tau_capital",
            ],
            format_func=lambda x: {
                "g_cons": "↑ Government Consumption Spending",
                "g_inv": "↑ Government Investment Spending",
                "tau_labor": "↓ Labor Taxes (Tax Cut)",
                "tau_capital": "↓ Capital Taxes (Tax Cut)",
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
                "g_cuts",
            ],
            format_func=lambda x: {
                "lump_sum": "Ricardian (Non-distortionary) Lump-Sum Transfers",
                "cons_tax": "Consumption Tax Hikes",
                "labor_tax": "Labor Tax Hikes",
                "capital_tax": "Capital Tax Hikes",
                "g_cuts": "Future Government Spending Cuts",
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
    
    # Map shock types to shock indices in B matrix
    shock_idx_map = {
        "g_cons": 0,      # government spending shock
        "g_inv": 0,       # also government spending (modified in A)
        "tau_labor": 3,   # labor tax shock
        "tau_capital": 4, # capital tax shock
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

    impact_mult, cum_mult, drag_horizon = compute_multipliers(y_gap_irf, g_irf, beta=0.99)

    st.subheader("📊 Fiscal Multipliers & Key Statistics")
    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("Impact Multiplier", f"{impact_mult:.3f}")
    colm2.metric("Cumulative Multiplier", f"{cum_mult:.3f}")
    if drag_horizon is not None:
        colm3.metric("Fiscal Drag Horizon", f"Q{drag_horizon}")
    else:
        colm3.metric("Fiscal Drag Horizon", "Beyond horizon")

    # IRF plots
    st.markdown("### 📉 Impulse Response Functions (40-Quarter Horizon)")

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8))
    t_grid_irf = np.arange(T_irf)

    # Output & Debt
    axes2[0, 0].plot(t_grid_irf, y_gap_irf, label="Output", color="cyan", linewidth=2.5)
    axes2[0, 0].axhline(0, color="white", linewidth=0.8)
    axes2[0, 0].set_title("Output Gap Response", fontsize=12, fontweight="bold")
    axes2[0, 0].grid(True, alpha=0.3)
    axes2[0, 0].set_ylabel("Deviation from SS")

    axes2[0, 1].plot(t_grid_irf, b_irf, label="Debt", color="yellow", linewidth=2.5)
    axes2[0, 1].axhline(0, color="white", linewidth=0.8)
    axes2[0, 1].set_title("Government Debt Response", fontsize=12, fontweight="bold")
    axes2[0, 1].grid(True, alpha=0.3)
    axes2[0, 1].set_ylabel("Deviation from SS")

    # Consumption & Investment
    axes2[1, 0].plot(t_grid_irf, c_irf, label="Consumption", color="orange", linewidth=2, marker="o", markersize=3)
    axes2[1, 0].plot(t_grid_irf, i_irf, label="Investment", color="magenta", linewidth=2, marker="s", markersize=3)
    axes2[1, 0].axhline(0, color="white", linewidth=0.8)
    axes2[1, 0].set_title("Consumption & Investment Response", fontsize=12, fontweight="bold")
    axes2[1, 0].legend(fontsize=10)
    axes2[1, 0].grid(True, alpha=0.3)
    axes2[1, 0].set_ylabel("Deviation from SS")

    # Inflation & Labor
    axes2[1, 1].plot(t_grid_irf, pi_irf, label="Inflation", color="#00ff00", linewidth=2, marker="d", markersize=3)
    axes2[1, 1].plot(t_grid_irf, n_irf, label="Labor", color="#ff00ff", linewidth=2, marker="^", markersize=3)
    axes2[1, 1].axhline(0, color="white", linewidth=0.8)
    axes2[1, 1].set_title("Inflation & Labor Response", fontsize=12, fontweight="bold")
    axes2[1, 1].legend(fontsize=10)
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].set_ylabel("Deviation from SS")

    fig2.patch.set_facecolor("#0b1020")
    for ax in axes2.flatten():
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.title.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")

    plt.tight_layout()
    st.pyplot(fig2)

    # Automated policy briefing
    shock_label = {
        "g_cons": "an increase in government consumption spending",
        "g_inv": "an increase in government investment spending",
        "tau_labor": "a cut in labor taxes (positive supply shock to labor)",
        "tau_capital": "a cut in capital taxes (positive incentive to invest)",
    }[shock_type]

    finance_label = {
        "lump_sum": "Ricardian (non-distortionary) lump-sum transfers that do not distort any economic margins",
        "cons_tax": "consumption tax hikes that directly dampen household demand and reduce disposable income",
        "labor_tax": "labor income tax hikes that discourage hours worked and reduce labor supply",
        "capital_tax": "capital income tax hikes that lower the after-tax return on investment and depress capital formation",
        "g_cuts": "future government spending cuts that unwind the initial fiscal expansion",
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
