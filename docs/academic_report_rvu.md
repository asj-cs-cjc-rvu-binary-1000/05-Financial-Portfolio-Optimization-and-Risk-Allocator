![RVU LOGO](docs/screenshots/rvu-logo.png)

## LA2133 THEORY AND PRACTICE OF LINEAR ALGEBRA
## Mini Project Report
### Financial Portfolio Optimization & Risk Allocator: An Application of Linear Algebra

**Submitted By**
Bibek Panigrahi (1RUA25CSE0121)
Aryan S Jain (1RUA25CSE0087)

**Under the guidance of:**
Prof. Manjunath S

School of Computer Science and Engineering
RV University, Bangalore
2025–26

---

![RVU LOGO](docs/screenshots/rvu-logo.png)

## School of Computer Science and Engineering
### CERTIFICATE

Certified that the LA2133 Theory and Practice of Linear Algebra Mini Project work titled **Financial Portfolio Optimization & Risk Allocator** is carried out by **Bibek Panigrahi (1RUA25CSE0121)** and **Aryan S Jain (1RUA25CSE0087)**, who are bonafide students of the School of Computer Science and Engineering, RV University, Bengaluru, during the year 2025–26. It is certified that all corrections/suggestions from all the continuous internal evaluations have been incorporated into the project and in this report.

Prof. Manjunath S ________________
Faculty Guide

Program Director ________________

---

# 1. Problem Statement

The central problem in quantitative finance is allocating capital across a set of assets to maximize expected return while minimizing risk. This is the **Markowitz Mean-Variance Optimization problem**. At its core, this is a constrained quadratic programming problem: minimizing the portfolio variance \(w^T \Sigma w\) subject to a budget constraint \(1^T w = 1\) and long-only bounds \(0 \le w_i \le 1\). This project builds a complete software pipeline—from raw market data to an interactive web dashboard—to solve this problem for a 7-asset US equity universe (AAPL, MSFT, JPM, JNJ, PG, XOM, AMZN) using only the principles of Linear Algebra (covariance matrices, eigendecomposition) and numerical optimization (SLSQP). By doing so, we demonstrate how foundational linear algebra concepts power real-world financial risk management.

# 2. Introduction

Portfolio theory relies entirely on the linear algebra of covariance. The risk of a portfolio is not merely the sum of individual asset risks; it is the quadratic form \(\sigma_p^2 = w^T \Sigma w\), where \(\Sigma\) is the covariance matrix of returns. The covariance matrix \(\Sigma\) is symmetric and positive semi-definite (PSD). Because it is PSD, it can be decomposed via the spectral theorem into \(\Sigma = V \Lambda V^T\), where \(V\) contains orthogonal eigenvectors and \(\Lambda\) contains non-negative eigenvalues. 

The core optimization problem—Markowitz Mean-Variance—requires minimizing this quadratic form under linear constraints. For the unconstrained case (allowing shorts), the closed-form solution is \(w^* = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}\), which relies on matrix inversion. However, with real-world constraints (long-only, budget), we must solve a convex quadratic program (QP) using numerical methods like Sequential Least Squares Programming (SLSQP).

Beyond optimization, measuring risk involves concepts like Value-at-Risk (VaR) and Maximum Drawdown (MDD). While these are more statistical than purely algebraic, the foundation remains the covariance structure. The project constructs an "Efficient Frontier"—the set of optimal portfolios that offer the highest expected return for a given level of volatility—by sweeping target returns and solving the QP iteratively. The final deliverable is a Streamlit dashboard that renders these complex linear algebra computations interactively.

# 3. System Architecture & Mathematical Diagram

*(Insert the Efficient Frontier chart and Dashboard Architecture image here from `docs/screenshots/`)*

**Diagram 1: Efficient Frontier (Markowitz Bullet)**
![Efficient Frontier](docs/screenshots/efficient_frontier.png)
*Figure 1: The efficient frontier computed from 7 US equities (2018–2026). The solid blue line is the efficient branch (50 points). The dashed gray line is the inefficient branch. The stars mark the Max-Sharpe and Min-Variance portfolios.*

**Diagram 2: Dashboard Architecture**
*(Insert `docs/screenshots/dashboard_optimization.png` here)*
*Figure 2: The Streamlit dashboard showing the optimization results.*

**Mathematical Data Flow:**
`Raw Prices (yfinance)` \(\to\) `Daily Returns \(X\)` \(\to\) `Covariance \(\Sigma = X^T X / (T-1)\)` \(\to\) `Optimization \(\min w^T \Sigma w\)` \(\to\) `Risk Metrics (VaR, CVaR)`.

# 4. Mathematical Formulation & Configuration

*(Equivalent to "Configuration setup")*

### 4.1 Data & Covariance Configuration
- **Universe:** 7 assets (AAPL, MSFT, JPM, JNJ, PG, XOM, AMZN).
- **Date Range:** 2018-01-02 to 2026-11-13 (2230 trading days).
- **Return Convention:** Simple returns (\(r_t = P_t/P_{t-1} - 1\)).
- **Annualization:** 252 trading days per year.

### 4.2 Linear Algebra Primitives (Phase 3)
- **Covariance Matrix:** \(\Sigma = \tilde{X}^T \tilde{X} / (T-1)\) (using Bessel's correction).
- **Eigendecomposition:** \(\Sigma = V \Lambda V^T\), solved using `numpy.linalg.eigh` (Hermitian solver).
- **Quadratic Form:** Portfolio variance \(\sigma_p^2 = w^T \Sigma w\).

### 4.3 Optimization Models (Phase 4)
- **Min-Variance:** \(\min w^T \Sigma w\) s.t. \(1^T w = 1, 0 \le w_i \le 1\).
- **Max-Sharpe:** \(\max (w^T \mu - r_f)/\sqrt{w^T \Sigma w}\) s.t. same constraints.
- **Target-Return:** \(\min w^T \Sigma w\) s.t. \(w^T \mu = \text{target}\).
- **Solver:** SciPy `optimize.minimize` (SLSQP), \(tol=10^{-10}\), \(w_0 = 1/N \cdot \mathbf{1}\).

### 4.4 Risk Metrics (Phase 5)
- **VaR (95%):** \(-Q_{0.05}(R_p)\).
- **CVaR:** \(-\mathbb{E}[R_p \mid R_p \le Q_{0.05}(R_p)]\).
- **MDD:** \(\min_t (C_t / \max_{s \le t} C_s - 1)\).

# 5. Results

*(Minimum 3 proofs — Insert the requested screenshots)*

**Proof 1: Optimization Comparison Table**
![Optimization Table](docs/screenshots/dashboard_optimization.png)
*Figure 3: Dashboard showing the comparison table for the 7 assets. EqualWeight gives 19.88% return (18.43% vol). MinVolatility gives 14.79% return (15.87% vol). MaxSharpe gives 22.36% return (20.18% vol) with the highest Sharpe ratio (1.009).*

**Proof 2: Risk Analysis Dashboard**
![Risk Dashboard](docs/screenshots/dashboard_risk.png)
*Figure 4: Risk metrics for the Max-Sharpe portfolio. VaR (95%) is 1.84%, CVaR is 2.99%, and Maximum Drawdown is -31.42%. The charts show the historical drawdown and return distribution.*

**Proof 3: Correlation & Eigenvalue Analysis**
![Correlation Heatmap](docs/screenshots/corr_heatmap.png)
*Figure 5: The correlation matrix. The first 3 principal components explain ~77% of the variance (market mode).*

**Proof 4: CLI Test Output**
252 passed in 5.16s
EqualWeight 0.1988 0.1843 0.0340 0.9701 True
MinVolatility 0.1479 0.1587 0.0252 0.8054 True
MaxSharpe 0.2236 0.2018 0.0407 1.0088 True

# 6. Conclusion

In this project, we successfully implemented a complete Markowitz Mean-Variance Portfolio Optimization system from scratch using Linear Algebra. We derived the covariance matrix \(\Sigma = \tilde{X}^T \tilde{X} / (T-1)\), verified its positive semi-definiteness via eigendecomposition, and formulated the portfolio optimization problem as a constrained quadratic program. By implementing the SLSQP solver, we generated the efficient frontier and compared Min-Variance, Max-Sharpe, and Equal-Weight portfolios on real historical data (2018-2026).

The mathematical framework proved highly effective: the Max-Sharpe portfolio achieved an annualized return of 22.36% with a Sharpe ratio of 1.009, outperforming the naive equal-weight benchmark (0.970). The system is packaged into an interactive Streamlit dashboard, allowing users to visually explore the risk-return tradeoff. 

**Future extensions:** We can extend this project by incorporating **shrinkage estimators** for the covariance matrix (Ledoit-Wolf) to combat estimation error, introducing **transaction costs** into the optimization objective, or employing **robust optimization** techniques. Additionally, we could apply this linear algebra engine to other domains, such as resource allocation in cloud computing or factor modeling in genomics.