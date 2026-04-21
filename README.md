# AI_Project_3
Quick Start 

pip install -r requirements.txt

streamlit run app.py

# Install dependencies
pip install streamlit numpy pandas scipy matplotlib

# Run the app
streamlit run app.py

# Access at: http://localhost:8501

# DSGE Fiscal Policy Dashboard

A comprehensive interactive DSGE (Dynamic Stochastic General Equilibrium) model for analyzing fiscal policy in a modern macroeconomic framework. Built for ECO 317 assignment with rule-of-thumb households, multiple structural shocks, and sophisticated fiscal multiplier analysis.

## 📊 Overview

This Streamlit application implements a state-of-the-art DSGE model featuring:
- **9-state linearized system** with Blanchard-Kahn conditions
- **Rule-of-thumb households** alongside optimizing agents
- **Multiple fiscal shocks** (government spending, tax cuts)
- **Flexible financing rules** (lump-sum, distortionary taxes, spending cuts)
- **Real-time impulse response functions** and fiscal multipliers
- **10+ business cycle moments** with economic interpretation

## 🎯 Key Features

### Model Components
- **Habit formation** in consumption (parameter h)
- **Variable capital utilization** (parameter ψ)
- **Calvo sticky prices and wages** (θ_p, θ_w)
- **Debt-feedback mechanism** (φ_b)
- **Rule-of-thumb households** (λ_rot)

### Interactive Analysis
- **Parameter sliders** for real-time model adjustment
- **Business cycle moments** table with AI-generated interpretation
- **Impulse response functions** for policy shocks
- **Fiscal multiplier calculations** (impact and cumulative)
- **Multiple shock types** and financing rules

### Professional UI
- Dark navy theme with Garamond typography
- Comprehensive parameter documentation
- Economic insights and policy briefings
- Responsive design with professional styling

## 🛠️ Requirements

- **Python 3.8+** (tested with Python 3.14.2)
- **Streamlit 1.55.0+**
- **NumPy 2.4.2+**
- **Pandas 2.3.3+**
- **SciPy 1.17.0+**
- **Matplotlib 3.10.8+**

## 📦 Installation

### Option 1: Direct Run (Recommended)
```bash
# Clone or download the repository
cd your-project-directory

# Install dependencies (if needed)
pip install streamlit numpy pandas scipy matplotlib

# Run the application
streamlit run app.py
```

### Option 2: Using requirements.txt
```bash
# Create requirements.txt with:
streamlit==1.55.0
numpy==2.4.2
pandas==2.3.3
scipy==1.17.0
matplotlib==3.10.8

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 3: Using Conda/Miniconda
```bash
# Create environment
conda create -n dsge-env python=3.14
conda activate dsge-env

# Install packages
conda install -c conda-forge streamlit numpy pandas scipy matplotlib

# Run the application
streamlit run app.py
```

## 🚀 Usage

### Basic Operation
1. **Launch the app**: `streamlit run app.py`
2. **Access in browser**: Navigate to `http://localhost:8501`
3. **Adjust parameters**: Use sidebar sliders to modify model parameters
4. **Explore tabs**:
   - **Tab 1**: View unconditional business cycle moments
   - **Tab 2**: Analyze conditional fiscal policy experiments

### Parameter Guide

| Parameter | Range | Description |
|-----------|-------|-------------|
| **h** | 0.0 - 0.9 | Habit formation in consumption (higher = smoother consumption) |
| **ψ** | 0.0 - 1.0 | Capital utilization convexity (higher = more elastic investment) |
| **θ_p** | 0.0 - 0.95 | Price stickiness (Calvo probability, higher = stickier prices) |
| **θ_w** | 0.0 - 0.95 | Wage stickiness (Calvo probability, higher = stickier wages) |
| **φ_b** | 0.0 - 1.0 | Debt-feedback aggressiveness (higher = stronger fiscal stress effects) |
| **λ_rot** | 0.0 - 0.6 | Rule-of-thumb households share (higher = more income-sensitive demand) |

### Fiscal Policy Experiments

#### Available Shocks
- **Government Consumption** (↑ G_cons): Direct spending increase
- **Government Investment** (↑ G_inv): Infrastructure/capital spending
- **Labor Tax Cut** (↓ τ_labor): Supply-side stimulus
- **Capital Tax Cut** (↓ τ_capital): Investment incentive

#### Financing Rules
- **Lump-sum**: Non-distortionary transfers
- **Consumption Tax**: VAT-style increases
- **Labor Tax**: Income tax increases
- **Capital Tax**: Corporate tax increases
- **Spending Cuts**: Future austerity

## 📈 Model Details

### State-Space Representation
The model is solved using linearization around steady state:

```
x_{t+1} = A x_t + B ε_t
y_t = C x_t
```

Where:
- **x**: 9 states (output gap, consumption, investment, labor, inflation, debt, spending, interest rate, technology)
- **ε**: 7 structural shocks
- **A**: State transition matrix (9×9)
- **B**: Shock loading matrix (9×7)
- **C**: Observation matrix (9×9, identity)

### Key Mechanisms

#### Rule-of-Thumb Households
Consumption = (1-λ_rot) × C_optimizers + λ_rot × Y_after_tax

This creates income-sensitive demand that amplifies fiscal multipliers.

#### Habit Formation
Consumption persistence: c_{t+1} = (0.6 + 0.2h) × c_t + ...

Higher habit formation (h) reduces contemporaneous consumption responses.

#### Sticky Prices/Wages
Phillips curve: π_{t+1} = 0.7 × π_t + κ × (y_gap) × (1-θ_p)(1-θ_w)

Flatter curve when prices/wages are stickier.

#### Debt Feedback
Automatic stabilizers: Tax revenue increases with output growth.

### Stability Conditions
- All eigenvalues of A matrix < 1 (stationarity)
- Debt persistence clipped to ≤ 0.99
- Pseudoinverse used for numerical stability

## 📊 Output Interpretation

### Business Cycle Moments
- **Std Dev**: Volatility relative to steady state
- **Rel. Volatility**: Ratio to output volatility
- **Corr w/ Output**: Contemporaneous correlation
- **Autocorr(1)**: First-order persistence

### Fiscal Multipliers
- **Impact**: ΔY₀ / ΔG₀ (immediate response)
- **Cumulative**: Discounted sum of effects
- **Drag Horizon**: When output turns negative (if applicable)

## 🎓 Academic Context

This application was developed for **ECO 317: Macroeconomics** to demonstrate:
- DSGE model construction and solution
- Fiscal policy transmission mechanisms
- Role of household heterogeneity (Ricardian vs. rule-of-thumb)
- Quantitative impact of parameter choices
- Real-world policy analysis tools

### Assignment Requirements Met
✅ Rule-of-thumb households (λ_rot parameter)  
✅ 10+ business cycle moments  
✅ Multiple structural shocks  
✅ Various financing rules  
✅ Impulse response functions  
✅ Fiscal multiplier calculations  
✅ Interactive parameter exploration  
✅ Economic interpretation  

## 🔧 Technical Notes

### Numerical Stability
- Eigenvalue clipping prevents explosive dynamics
- Pseudoinverse (pinv) handles near-singular matrices
- Robust error handling for edge cases
- Burn-in periods for accurate statistics

### Performance
- Cached state-space construction
- Efficient matrix operations
- Real-time parameter updates
- Memory-optimized simulations

## 📝 License

This project is developed for educational purposes as part of ECO 317 coursework.

## 🤝 Contributing

For academic use or improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with documentation

## 📞 Support

For questions about the model implementation or economic interpretation:
- Review the inline documentation in `app.py`
- Check parameter tooltips in the application
- Reference standard DSGE literature (Smets & Wouters, 2007)

---

**Built with ❤️ for macroeconomic education**</content>
<parameter name="filePath">C:\Users\paris\Downloads\README.md
