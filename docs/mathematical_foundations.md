# Mathematical Foundations — Covariance, Eigen Analysis, and Markowitz Optimization

> This document demonstrates meaningful applications of Linear Algebra for Financial Portfolio Optimization.
> Code lives in `src/linear_algebra/` and `src/optimization/`; this doc contains only math.
> Sections 1–10 cover Phase 3 (covariance, eigendecomposition, correlation). Section 11 covers Phase 4 (Markowitz optimization).

---

## 1. Notation

| Symbol | Space / Size | Meaning |
| :--- | :--- | :--- |
| $T$ | $\mathbb{N}$ | Number of trading days (observations) |
| $N$ | $\mathbb{N}$ | Number of assets (tickers), $N=7$ in this project |
| $X \in \mathbb{R}^{T \times N}$ | $T \times N$ | Daily returns matrix, $X_{t,i}=r_{t,i}$ |
| $r_{:,i}$ | $\mathbb{R}^T$ | Return time series of asset $i$ (column $i$ of $X$) |
| $\mu \in \mathbb{R}^N$ | $N$ | Column means $\mu_i = \frac{1}{T}\sum_{t=1}^T r_{t,i}$ |
| $\mathbf{1}_T \in \mathbb{R}^T$ | $T$ | Vector of ones, length $T$ |
| $\tilde{X} \in \mathbb{R}^{T \times N}$ | $T \times N$ | Centered returns $\tilde{X}=X - \mathbf{1}_T \mu^\top$ |
| $\Sigma \in \mathbb{R}^{N \times N}$ | $N \times N$ | Covariance matrix, daily or annualized |
| $\Sigma_{\text{daily}}$ | $N \times N$ | $\tilde{X}^\top \tilde{X}/(T-1)$ |
| $\Sigma_{\text{annual}}$ | $N \times N$ | $252 \cdot \Sigma_{\text{daily}}$ |
| $D = \text{diag}(\sigma_i)$ | $N \times N$ | Diagonal matrix of volatilities |
| $\sigma \in \mathbb{R}^N$ | $N$ | $\sigma_i = \sqrt{\Sigma_{ii}}$ |
| $\text{Corr} \in \mathbb{R}^{N \times N}$ | $N \times N$ | Correlation matrix, unit-free |
| $w \in \mathbb{R}^N$ | $N$ | Portfolio weights |
| $V \in \mathbb{R}^{N \times N}$ | $N \times N$ | Orthogonal eigenvector matrix, $V^\top V = I$ |
| $\Lambda = \text{diag}(\lambda_i)$ | $N \times N$ | Eigenvalue matrix, $\lambda_1 \ge \dots \ge \lambda_N \ge 0$ |
| $\lambda_i$ | $\mathbb{R}_{\ge 0}$ | Eigenvalue = variance along eigenvector $v_i$ |
| $r_f$ | $\mathbb{R}$ | Risk-free rate, annualized decimal ($0.02$ = 2%/yr) |
| $S_p$ | $\mathbb{R}$ | Sharpe ratio, $(w^\top \mu - r_f)/\sqrt{w^\top \Sigma w}$ |
| $\mathbf{1}_N \in \mathbb{R}^N$ | $N$ | Vector of ones, length $N$ (used as $1^\top w = 1$) |

---

## 2. Covariance Derivation

### 2.1 Population definition

For two random returns $X_i, X_j$:

$$
\text{Cov}(X_i, X_j) = \mathbb{E}\left[(X_i - \mu_i)(X_j - \mu_j)\right]
$$

where $\mu_i = \mathbb{E}[X_i]$.

In particular $\text{Var}(X_i)=\text{Cov}(X_i,X_i)$.

### 2.2 Sample estimator

Given $T$ observations $r_{t,i}$, the unbiased sample estimator uses Bessel's correction $T-1$:

$$
\hat{\mu}_i = \frac{1}{T}\sum_{t=1}^T r_{t,i}
$$

$$
\hat{\Sigma}_{ij} = \frac{1}{T-1}\sum_{t=1}^T (r_{t,i}-\hat{\mu}_i)(r_{t,j}-\hat{\mu}_j)
$$

$$
\hat{\Sigma}_{ii} = \frac{1}{T-1}\sum_{t=1}^T (r_{t,i}-\hat{\mu}_i)^2
$$

Division by $T-1$ not $T$ makes $\mathbb{E}[\hat{\Sigma}] = \Sigma$ unbiased.

### 2.3 Matrix form $\Sigma = \frac{1}{T-1}\tilde{X}^\top \tilde{X}$

Define the centered matrix:

$$
\tilde{X} = X - \mathbf{1}_T \mu^\top, \quad \tilde{X}_{t,i}=r_{t,i}-\mu_i, \quad \frac{1}{T}\sum_t \tilde{X}_{t,i}=0
$$

Then:

$$
\tilde{X}^\top \tilde{X} \in \mathbb{R}^{N \times N}, \quad (\tilde{X}^\top \tilde{X})_{ij}=\sum_{t=1}^T (r_{t,i}-\mu_i)(r_{t,j}-\mu_j)
$$

Shape contract: $(N \times T) \cdot (T \times N) = N \times N$.

Therefore:

$$
\Sigma_{\text{daily}} = \frac{1}{T-1}\tilde{X}^\top \tilde{X}
$$

$$
\Sigma_{\text{annual}} = 252 \cdot \Sigma_{\text{daily}} \quad \text{(i.i.d. approximation, see §7)}
$$

Implementation uses Task 3.1 primitives: `center_columns`, `transpose`, `matrix_multiply` — no `DataFrame.cov()`.

---

## 3. Symmetry $\Sigma = \Sigma^\top$

From definition:

$$
\Sigma_{ij} = \frac{1}{T-1}\sum_t (r_{t,i}-\mu_i)(r_{t,j}-\mu_j) = \frac{1}{T-1}\sum_t (r_{t,j}-\mu_j)(r_{t,i}-\mu_i) = \Sigma_{ji}
$$

by commutativity of scalar multiplication. Hence $\Sigma = \Sigma^\top$. Float asymmetry $||\Sigma - \Sigma^\top||_{\max} \le 10^{-10}$ is round-off only.

Symmetry enables the Hermitian solver `eigh` — real eigenvalues, orthogonal eigenvectors.

---

## 4. Positive Semi-Definiteness $w^\top \Sigma w \ge 0$

For any $w \in \mathbb{R}^N$:

$$
w^\top \Sigma w = w^\top \left(\frac{1}{T-1}\tilde{X}^\top \tilde{X}\right) w = \frac{1}{T-1} (\tilde{X}w)^\top (\tilde{X}w) = \frac{1}{T-1} \|\tilde{X}w\|_2^2 \ge 0
$$

Proof in two lines:

1. $\Sigma = \frac{1}{T-1}\tilde{X}^\top \tilde{X}$ is a Gram matrix.
2. $w^\top \tilde{X}^\top \tilde{X} w = \|\tilde{X}w\|^2 \ge 0$, with equality only if $\tilde{X}w=0$.

Consequences:
- All eigenvalues $\lambda_i \ge 0$ (Task 3.3 checks via `eigvalsh`)
- Determinant $>0$ is necessary but not sufficient — eigenvalue test is required
- $\Sigma$ is invertible iff strictly PD ($\lambda_{\min}>0$); PSD with $\lambda_{\min}\approx 0$ indicates near-collinear tickers

Condition number $\kappa = \lambda_{\max}/\lambda_{\min}$: 10–200 moderate for 7 tickers over 5y; $>10^6$ warns of multicollinearity.

---

## 5. Eigendecomposition $\Sigma = V \Lambda V^\top$

### 5.1 Spectral theorem for symmetric matrices

For symmetric $\Sigma$, there exists orthogonal $V$ ($V^\top V = I$, $V^{-1}=V^\top$) and real diagonal $\Lambda$:

$$
\Sigma = V \Lambda V^\top = \sum_{k=1}^N \lambda_k v_k v_k^\top
$$

where $v_k = V_{:,k}$ is eigenvector, $\lambda_k$ eigenvalue, $\Lambda = \text{diag}(\lambda_1,\dots,\lambda_N)$, sorted $\lambda_1 \ge \dots \ge \lambda_N$ (descending convention for variance-explained).

- $V$ orthogonal, $\|v_k\|=1$, $v_i^\top v_j = 0$ for $i \ne j$
- $\lambda_i \in \mathbb{R}$ guaranteed by symmetry; $\lambda_i \ge 0$ guaranteed by PSD
- Sign convention: `eigh` returns arbitrary sign; we enforce largest-magnitude component positive for interpretability (does not change math but changes heatmap colors)

### 5.2 Variance explained

$$
\text{Tr}(\Sigma) = \sum_i \Sigma_{ii} = \sum_k \lambda_k = \text{total variance}
$$

$$
\text{explained}_k = \frac{\lambda_k}{\sum_j \lambda_j}, \quad \text{cumulative}_k = \sum_{j=1}^k \text{explained}_j
$$

First PC explains $\lambda_1 / \text{Tr}(\Sigma)$ of total variance. For 7-stock equity, first 3 PCs typically explain 70–85%, indicating effectively 3 independent risk factors. First eigenvector often has all same sign — the "market mode".

In eigenbasis $a = V^\top r$, components are uncorrelated:

$$
\text{Var}(a_k) = \lambda_k, \quad \text{Cov}(a_k,a_j)=0 \ (k \ne j)
$$

Reconstruction identity (must hold atol $10^{-10}$):

$$
V \Lambda V^\top \approx \Sigma, \quad \max_{ij} | (V\Lambda V^\top)_{ij} - \Sigma_{ij} | < 10^{-10}
$$

---

## 6. Correlation as Diagonal Scaling

### 6.1 Derivation

Let $\sigma_i = \sqrt{\Sigma_{ii}}$.

$$
\text{Corr}_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}
$$

Matrix form with $D = \text{diag}(\sigma_1,\dots,\sigma_N)$:

$$
\text{Corr} = D^{-1} \Sigma D^{-1}
$$

Steps:
1. $\sigma = \sqrt{\text{diag}(\Sigma)}$
2. $D^{-1} = \text{diag}(1/\sigma_i)$
3. $\text{Corr} = D^{-1} \Sigma D^{-1}$

### 6.2 Properties

- $\text{diag}(\text{Corr}) = 1$
- $\text{Corr}_{ij} \in [-1,1]$
- Symmetric, PSD
- Unit-free — measures linear co-movement independent of scale, so a tech stock at $\$300$ and a staple at $\$40$ are comparable.

### 6.3 Eigenvalues of Corr vs Σ

$\text{Corr}$ is covariance of standardized returns $\tilde{X} D^{-1}$. Different eigenproblem from $\Sigma$:
- $\Sigma$ eigenvalues: variances in return units (scale matters for $w^\top \Sigma w$)
- $\text{Corr}$ eigenvalues: variances of standardized returns (unit-free, used for the heatmap)

Phase 4 optimization uses $\Sigma$; the heatmap uses Corr.

---

## 7. Annualization Conventions

Assume i.i.d. daily returns (approximation).

| Quantity | Daily | Annual | Reason |
| :--- | :--- | :--- | :--- |
| Mean $\mu$ | $\mu_{\text{daily}} = \frac{1}{T}\sum r_t$ | $\mu_{\text{annual}} = 252 \cdot \mu_{\text{daily}}$ | Linearity of expectation, $T$ days |
| Variance $\Sigma_{ii}$ | $\Sigma_{\text{daily}}$ | $\Sigma_{\text{annual}} = 252 \cdot \Sigma_{\text{daily}}$ | $\text{Var}(\sum_{t=1}^{252} r_t) = 252 \cdot \text{Var}(r_t)$ under i.i.d. |
| Volatility $\sigma$ | $\sigma_{\text{daily}} = \sqrt{\Sigma_{\text{daily},ii}}$ | $\sigma_{\text{annual}} = \sqrt{252}\cdot \sigma_{\text{daily}} = \sqrt{\text{diag}(\Sigma_{\text{annual}})}$ | $\sqrt{252}$ scaling, contract $\sqrt{\text{diag}(\Sigma_{\text{annual}})} = \sigma_{\text{annual}}$ |
| Covariance $\Sigma_{ij}$ | $\Sigma_{\text{daily},ij}$ | $252 \cdot \Sigma_{\text{daily},ij}$ | Same as variance |
| Correlation $\text{Corr}$ | $D^{-1}\Sigma_{\text{daily}} D^{-1}$ | Same — $252$ cancels: $\frac{252\Sigma_{ij}}{\sqrt{252}\sigma_i \sqrt{252}\sigma_j} = \frac{\Sigma_{ij}}{\sigma_i\sigma_j}$ | Scale-invariant |

Check: $\sigma_{\text{annual}}$ from Phase 2 `compute_volatility` must equal $\sqrt{\text{diag}(\Sigma_{\text{annual}})}$ within $10^{-10}$ (Task 3.3 `verify_covariance_properties` hard requirement).

---

## 8. Why Covariance Structure Matters for Finance

Portfolio variance is exactly the quadratic form:

$$
\sigma_p^2 = w^\top \Sigma w = \sum_{i,j} w_i w_j \Sigma_{ij}
$$

Properties used:

- Symmetry ensures $w^\top \Sigma w$ real
- PSD ensures $\sigma_p^2 \ge 0$ for any $w$
- Eigen decomposition diagonalizes risk: in basis $a=V^\top w$, $\sigma_p^2 = \sum_k \lambda_k a_k^2$

- Minimizing $w^\top \Sigma w$ is minimizing portfolio variance (quadratic form)
- Eigenvalues tell the risk budget: steep drop-off = low-rank covariance = diversification limited to few factors
- Off-diagonal $\text{Corr}_{ij} \approx 1$ → near-collinear → small eigenvalue → near-zero variance direction → ill-conditioned $\Sigma$ → optimization unstable
- Condition number $\kappa(\Sigma)$ signals when to regularize (shrinkage, factor model)

The heatmap visualizes diversification potential: values near 0 good, near ±1 bad. The eigenvalue spectrum visualizes the effective number of risk factors.

---

## 9. Implementation Map (Phase 3)

| Formula | Module | Function | Prohibited Call |
| :--- | :--- | :--- | :--- |
| $\tilde{X}=X-\mathbf{1}\mu^\top$ | `linear_algebra/matrix_ops.py` | `center_columns` | — |
| $\Sigma = \tilde{X}^\top \tilde{X}/(T-1)$ | `linear_algebra/covariance.py` | `compute_covariance_matrix` | `DataFrame.cov()` |
| $\text{Corr}=D^{-1}\Sigma D^{-1}$ | `linear_algebra/covariance.py` | `compute_correlation_matrix` | `DataFrame.corr()` |
| $\Sigma = V\Lambda V^\top$ | `linear_algebra/eigen_analysis.py` | `eigen_decompose` | `DataFrame.eigen*` |
| $w^\top \Sigma w$ | `linear_algebra/matrix_ops.py` | `quadratic_form` | — |
| Symmetry/PSD/$\sigma$ check | `linear_algebra/covariance.py` | `verify_covariance_properties` | — |
| Variance-explained table | `linear_algebra/eigen_analysis.py` | `eigen_summary_table`, `first_principal_direction` | — |
| Heatmap + spectrum | `visualization/charts.py` | `plot_correlation_heatmap`, `plot_eigenvalue_spectrum` | `seaborn`, `matplotlib` |

Tests in `tests/test_covariance.py` lock identities against Pandas/NumPy oracles and internal consistency.

---

## 10. References for Grader (Phase 3)

- Task 3.2: $\Sigma = X^\top X/(T-1)$ shape $(N,T)(T,N)=N\times N$, Bessel's correction
- Task 3.3: Symmetry $\Sigma_{ij}=\Sigma_{ji}$ by scalar commutativity; PSD $w^\top\Sigma w = \frac{1}{T-1}\|\tilde{X}w\|^2 \ge 0$; $\sqrt{\text{diag}(\Sigma_{\text{annual}})} = \sigma_{\text{annual}}$ within $10^{-10}$
- Task 3.4: $\text{Corr}_{ij}=\Sigma_{ij}/(\sigma_i\sigma_j)$, $\text{Corr}=D^{-1}\Sigma D^{-1}$, diag 1, $\in[-1,1]$
- Task 3.5: $\Sigma=V\Lambda V^\top$, `eigh` Hermitian solver, descending, $V^\top V=I$, sign convention largest component positive, reconstruction atol $10^{-10}$
- Task 3.6: Explained $\lambda_k/\text{Tr}(\Sigma)$, cumulative, first PC market mode all same sign for equity universe
- Task 3.7: Heatmap RdBu centered 0 range [-1,1] with 2-decimal annotations; eigenvalue bar + cumulative line secondary y-axis ends 1.0
- Task 3.8: Offline pytest 17 tests prove identities vs Pandas/NumPy oracles

---

## 11. Markowitz Mean–Variance Optimization (Phase 4)

Phase 3 established that portfolio variance is the quadratic form $\sigma_p^2(w) = w^\top \Sigma w$.
Phase 4 constructs $w$ by **minimizing or constrained-optimizing this quadratic form** under
the project's feasible set:

$$
\mathcal{F} = \left\{ w \in \mathbb{R}^N \;\middle|\; \mathbf{1}_N^\top w = 1,\; 0 \le w_i \le 1 \;\forall i \right\}.
$$

$\mathcal{F}$ is the standard $N$-simplex: **fully invested** and **long-only**. No shorts, no
leverage, no individual weight caps.

### 11.1 Minimum-variance problem

$$
\begin{aligned}
\min_{w \in \mathcal{F}} \quad & w^\top \Sigma w \\
\text{s.t.} \quad & \mathbf{1}_N^\top w = 1, \\
& 0 \le w_i \le 1 \quad \forall i \in \{1,\dots,N\}.
\end{aligned}
$$

Because $\Sigma$ is PSD (§4), the objective is convex in $w$. Because $\mathcal{F}$ is a
polytope (intersection of linear constraints and a box), the feasible set is convex. A
convex objective over a convex set has **any local minimum = global minimum**; hence the
SLSQP solver in Task 4.6 either finds the global minimum or honestly reports failure.

### 11.2 Lagrangian — equality-only case (analytic oracle)

Temporarily drop $w_i \ge 0$ (allow shorts) but keep the budget constraint. Define

$$
\mathcal{L}(w, \lambda) = w^\top \Sigma w - \lambda (\mathbf{1}_N^\top w - 1).
$$

First-order condition on $w$:

$$
\frac{\partial \mathcal{L}}{\partial w} = 2 \Sigma w - \lambda \mathbf{1}_N = 0
\quad\Longleftrightarrow\quad
\Sigma w = \frac{\lambda}{2} \mathbf{1}_N.
$$

$\Sigma$ is strictly PD under the project's data (min eigenvalue $\approx 7.3 \times 10^{-5}$ from
Task 3.3), so $\Sigma^{-1}$ exists and

$$
w^* = \frac{\lambda}{2} \, \Sigma^{-1} \mathbf{1}_N.
$$

Apply the budget constraint $\mathbf{1}_N^\top w^* = 1$:

$$
\frac{\lambda}{2} \mathbf{1}_N^\top \Sigma^{-1} \mathbf{1}_N = 1
\quad\Longleftrightarrow\quad
\frac{\lambda}{2} = \frac{1}{\mathbf{1}_N^\top \Sigma^{-1} \mathbf{1}_N}.
$$

Therefore the **closed-form unconstrained min-variance weights** are

$$
\boxed{ \; w^*_{\text{min-var}} = \frac{\Sigma^{-1} \mathbf{1}_N}{\mathbf{1}_N^\top \Sigma^{-1} \mathbf{1}_N} \; }
$$

This is the **oracle** against which the Task 4.6 test
(`test_matches_analytic_2asset_oracle`) validates the SLSQP output. On the analytic
2-asset case, $w^* \approx (0.7273, 0.2727)$ — both positive, so the long-only constraint
does not bind and the SLSQP result must match the closed form to within $10^{-5}$.

### 11.3 Long-only case (KKT, no closed form)

When the long-only constraint $0 \le w_i \le 1$ is active, the closed form in §11.2 may
produce negative components (i.e., short positions), which $\mathcal{F}$ forbids. There is
**no closed-form solution** in general. The KKT conditions are:

$$
\begin{aligned}
2 \Sigma w - \lambda \mathbf{1}_N - \mu + \nu &= 0, \\
\mu_i \, w_i &= 0, \quad \nu_i \, (1 - w_i) = 0, \quad \forall i, \\
\mu_i \ge 0, \quad \nu_i &\ge 0, \quad \forall i,
\end{aligned}
$$

combined with $\mathbf{1}_N^\top w = 1$ and $0 \le w_i \le 1$. The active-set structure
(which of the $2N$ bound constraints are active) is not known a priori. In practice, SLSQP
solves this quadratic program directly. Task 4.6 verifies SLSQP against the §11.2 closed
form *only on universes where the closed form happens to be long-only-feasible*, and
against a hand-verifiable 2-asset corner solution otherwise.

### 11.4 Maximum-Sharpe problem

The Sharpe ratio of a portfolio is

$$
S_p(w) = \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}},
$$

where $r_f$ is the annualized risk-free rate (project convention: decimal, $0.02$ = 2%/yr).
The maximum-Sharpe problem is

$$
\begin{aligned}
\max_{w \in \mathcal{F}} \quad & S_p(w) \\
\text{s.t.} \quad & \mathbf{1}_N^\top w = 1, \\
& 0 \le w_i \le 1 \quad \forall i.
\end{aligned}
$$

Three structural facts:

1. **Not convex in $w$.** $S_p$ is a ratio of a linear function to a convex function
   ($\sqrt{w^\top \Sigma w}$). Minimizing $-S_p$ is not a convex problem. SLSQP is a
   **local** solver, so in principle it may land on a local (sub-optimal) stationary point.
   In practice for $N = 7$ equities, SLSQP with the equal-weight initial guess reliably
   finds the global maximum. The Task 4.7 test `test_max_sharpe_beats_equal_weight`
   enforces the weaker but rigorous guarantee $S_p(w^*_{\text{max-Sharpe}}) \ge S_p(w_{\text{eq}})$.

2. **Tangency-portfolio closed form only without long-only.** If we relax the box
   constraints, the tangency (max-Sharpe) portfolio has the known closed form
   $$
   w_{\text{tan}} \propto \Sigma^{-1} (\mu - r_f \mathbf{1}_N),
   $$
   normalized so $\mathbf{1}_N^\top w_{\text{tan}} = 1$. With long-only, this closed form
   is **not** used — SLSQP must solve the constrained problem directly. The Task 4.7
   implementation follows the constrained path.

3. **Negative Sharpe is a valid answer.** If every portfolio in $\mathcal{F}$ has
   $S_p(w) < 0$ (no long-only combination beats $r_f$), the maximum is still well-defined
   (it is the least-negative value), and Task 4.7 returns it with `success=True` and a
   message explaining the situation. Spec Section 5.1, Q3, option A.

### 11.5 Target-return problem (frontier primitive)

Fix a target expected return $\bar{\mu} \in [\mu_{\min}, \mu_{\max}]$ and solve

$$
\begin{aligned}
\min_{w \in \mathcal{F}} \quad & w^\top \Sigma w \\
\text{s.t.} \quad & \mathbf{1}_N^\top w = 1, \\
& w^\top \mu = \bar{\mu}, \\
& 0 \le w_i \le 1 \quad \forall i.
\end{aligned}
$$

Two linear equalities and one quadratic objective — this is a **quadratic program**.
The feasible set is non-empty for $\bar{\mu} \in [\mu_{\min}, \mu_{\max}]$, with:

- $\bar{\mu} = \mu_{\min}$ → the corner solution $w = e_{i_{\min}}$ (all-in on the lowest-μ asset), *if long-only permits it*; otherwise the lowest-feasible-return portfolio.
- $\bar{\mu} = \mu_{\max}$ → the corner solution $w = e_{i_{\max}}$ (all-in on the highest-μ asset), *if long-only permits it*.
- $\bar{\mu} = \mu_{\text{min-var}}$ (the return of the §11.1 solution) → the SLSQP output matches the minimum-variance portfolio to within $10^{-5}$.

**Efficient frontier:** As $\bar{\mu}$ sweeps from $\mu_{\text{min-var}}$ to $\mu_{\max}$, the
target-return solutions trace the **upper branch** of the minimum-variance envelope. This
is the classic efficient frontier. The lower branch ($\bar{\mu} < \mu_{\text{min-var}}$) is
**inefficient** — dominated by a higher-return portfolio at the same or lower variance —
and Phase 5 restricts its sweep to $[\mu_{\text{min-var}}, \mu_{\max}]$.

Feasibility is not equivalent to $\bar{\mu} \in [\mu_{\min}, \mu_{\max}]$: the long-only box
can make some interior targets unreachable, but for $N = 7$ US equities with these
tickers, all interior targets are feasible. Task 4.8 fails loudly (spec Section 4.1–4.3)
when $\bar{\mu} \notin [\mu_{\min}, \mu_{\max}]$; the long-only interaction is not
special-cased.

### 11.6 Equal-weight diversification and the "1/N" benchmark

The equal-weight portfolio

$$
w_{\text{eq}} = \frac{1}{N} \mathbf{1}_N
$$

has variance

$$
\sigma_{\text{eq}}^2 = w_{\text{eq}}^\top \Sigma w_{\text{eq}}
= \frac{1}{N^2} \mathbf{1}_N^\top \Sigma \mathbf{1}_N
= \frac{1}{N^2} \sum_{i,j} \Sigma_{ij}.
$$

Split the sum into diagonal and off-diagonal parts:

$$
\sigma_{\text{eq}}^2 = \frac{1}{N^2} \sum_{i=1}^N \Sigma_{ii} + \frac{1}{N^2} \sum_{i \ne j} \Sigma_{ij}.
$$

Let $\bar{\sigma}^2 = \frac{1}{N}\sum_i \Sigma_{ii}$ (average variance) and
$\bar{\rho}\,\bar{\sigma}^2$ the average off-diagonal covariance (with $\bar{\rho}$ the
average correlation). Then

$$
\sigma_{\text{eq}}^2
= \frac{\bar{\sigma}^2}{N} + \frac{N(N-1)}{N^2} \cdot \bar{\rho}\,\bar{\sigma}^2
= \bar{\sigma}^2 \left[ \frac{1}{N} + \left(1 - \frac{1}{N}\right) \bar{\rho} \right].
$$

**As $N \to \infty$:**

$$
\sigma_{\text{eq}}^2 \to \bar{\sigma}^2 \cdot \bar{\rho}.
$$

Interpretation:

- If assets are uncorrelated ($\bar{\rho} = 0$), the equal-weight portfolio variance
  vanishes as $N \to \infty$ — naive diversification eliminates idiosyncratic risk.
- If assets are perfectly correlated ($\bar{\rho} = 1$), the equal-weight variance equals
  $\bar{\sigma}^2$ for every $N$ — diversification cannot help, risk is purely systemic.
- In practice $\bar{\rho} \approx 0.3$–$0.6$ for US equities, so
  $\sigma_{\text{eq}}^2 \to \bar{\sigma}^2 \bar{\rho}$ — a substantial reduction from
  $\bar{\sigma}^2$ but not zero. This is the mathematical statement of
  "diversification reduces but cannot eliminate portfolio risk."

The equal-weight portfolio is the **benchmark** against which every optimizer is measured.
Any long-only optimizer at the same or higher expected return with the same or lower
volatility dominates $w_{\text{eq}}$; if it does not, either the optimizer found a local
optimum (spec Section 8) or the data is degenerate.

### 11.7 Constraint feasibility and the numerical solver

The project feasible set $\mathcal{F}$ is enforced in code by three reusable helpers in
`src/optimization/constraints.py`:

- `budget_constraint()` → scipy equality: $f(w) = \mathbf{1}^\top w - 1 = 0$.
- `long_only_bounds(n)` → scipy box: `[(0.0, 1.0)] * n`.
- `target_return_constraint(mu, target)` → scipy equality: $f(w) = w^\top \mu - \bar{\mu} = 0$.

Each optimizer in `src/optimization/optimizer.py` (Tasks 4.6, 4.7, 4.8) passes these
objects directly to `scipy.optimize.minimize(..., method="SLSQP")`. The post-optimization
gate `validate_weights` re-checks $\mathbf{1}^\top w \approx 1$ and $0 \le w_i \le 1$ within
$10^{-6}$, and downgrades `success` to `False` if any check fails (spec Section 5.3).

The default initial guess is $w_0 = w_{\text{eq}} = \frac{1}{N} \mathbf{1}_N$ — strictly
interior to $\mathcal{F}$, avoids the zero-gradient degeneracy at $w = \mathbf{0}$, and
matches the benchmark so SLSQP cannot trivially improve by starting at a boundary point.

### 11.8 Implementation Map (Phase 4)

| Formula | Module | Function | Prohibited Call |
| :--- | :--- | :--- | :--- |
| $R_p = w^\top \mu$ | `optimization/portfolio_metrics.py` | `portfolio_return` | — |
| $\sigma_p^2 = w^\top \Sigma w$ | `optimization/portfolio_metrics.py` | `portfolio_variance` | inline $w \Sigma w$ (must route via `quadratic_form`) |
| $\sigma_p = \sqrt{w^\top \Sigma w}$ | `optimization/portfolio_metrics.py` | `portfolio_volatility` | — |
| $S_p = (R_p - r_f)/\sigma_p$ | `optimization/portfolio_metrics.py` | `portfolio_sharpe` | — |
| Feasible set $\mathcal{F}$ | `optimization/constraints.py` | `validate_weights`, `budget_constraint`, `long_only_bounds`, `target_return_constraint` | — |
| $w_{\text{eq}} = \frac{1}{N}\mathbf{1}_N$ | `optimization/benchmarks.py` | `equal_weights`, `equal_weight_portfolio` | — |
| $\min_{w \in \mathcal{F}} w^\top \Sigma w$ | `optimization/optimizer.py` | `minimum_variance_portfolio` | `PyPortfolioOpt`, `cvxpy` |
| $\max_{w \in \mathcal{F}} S_p(w)$ | `optimization/optimizer.py` | `maximum_sharpe_portfolio` | `PyPortfolioOpt` |
| $\min_{w \in \mathcal{F}} w^\top \Sigma w$ s.t. $w^\top\mu = \bar{\mu}$ | `optimization/optimizer.py` | `target_return_portfolio` | `PyPortfolioOpt` |
| Comparison table | `optimization/optimization_pipeline.py` | `compare_portfolios` | — |

Tests in `tests/test_optimizer.py` (45) and `tests/test_optimization_pipeline.py` (13)
lock the identities against the §11.2 analytic oracle, the constraint helpers, and the
mathematical guarantees of §11.6.

### 11.9 References for Grader (Phase 4)

- Task 4.1: Verified inputs — $\mu$ as `pd.Series`, $\Sigma$ as `pd.DataFrame`, both label-aligned on ticker order; 252-day annualization; 37-test baseline reproduced
- Task 4.2: Optimization specification — decisions Q1–Q7 documented in `docs/optimization_specification.md`
- Task 4.3: Four metric primitives; `portfolio_variance` routes through `quadratic_form` (design-invariant test)
- Task 4.4: Four constraint helpers; 30 offline tests including boundary cases
- Task 4.5: Equal-weight benchmark; `n_iter = 0`, `success = True`; matches primitives exactly (design-invariant test)
- Task 4.6: Min-variance — matches §11.2 closed form to $10^{-5}$ on the analytic 2-asset case; 45 optimizer tests including solver-failure and design-invariant checks
- Task 4.7: Max-Sharpe — beats equal-weight on the synthetic 3-asset case; Q3 negative-Sharpe returns `success=True` with a message; Q4 `ZeroDivisionError` → sentinel $10^{10}$
- Task 4.8: Target-return — matches min-variance at $\bar\mu = \mu_{\text{min-var}}$, corner at $\bar\mu = \mu_{\min}$ and $\mu_{\max}$; Q1/Q2 out-of-range fails loudly; spec-4.3 tolerance downgrades over-target solutions
- Task 4.9: `compare_portfolios` — fixed row/column order; EqualWeight row matches `equal_weight_portfolio` (design-invariant test); 13 pipeline tests
- Task 4.10: CLI + notebook + docs — `python -m src.optimization.optimization_pipeline` runs end-to-end on 7-ticker real data, writes `docs/optimization_summary.md`

This doc renders with LaTeX on GitHub (MathJax). No code blocks in the math sections — math only. Code blocks appear only in the "Implementation Map" tables where the reference to a function is clearer as inline code.