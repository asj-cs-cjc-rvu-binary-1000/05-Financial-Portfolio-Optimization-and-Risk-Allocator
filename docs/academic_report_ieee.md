# Financial Portfolio Optimization & Risk Allocator: A Linear Algebra Approach

**Bibek Panigrahi**  
*School of Computer Science and Engineering*  
*RV University*  
Bangalore, India  
bibekpanigrahibtech25@rvu.edu.in

**Aryan S Jain**  
*School of Computer Science and Engineering*  
*RV University*  
Bangalore, India  
aryansjainbtech25@rvu.edu.in

---

**Abstract**—This paper presents a comprehensive computational framework for Markowitz Mean-Variance Portfolio Optimization, implemented from first principles using Linear Algebra. We construct the covariance matrix from historical returns via the matrix identity \(\Sigma = \tilde{X}^T \tilde{X} / (T-1)\), verify its structural properties (symmetry, positive semi-definiteness), and perform eigendecomposition to analyze risk factors. The optimization engine solves the constrained quadratic program using Sequential Least Squares Programming (SLSQP), generating Minimum-Variance, Maximum-Sharpe, and Target-Return portfolios. We also implement historical Value-at-Risk (VaR), Conditional Value-at-Risk (CVaR), and Maximum Drawdown metrics. Applied to a 7-asset US equity portfolio (2018–2026), the Maximum-Sharpe strategy achieves an annualized return of 22.36% and a Sharpe ratio of 1.009, outperforming the equal-weight benchmark (0.970). All computations are exposed via an interactive Streamlit dashboard.

**Keywords**—Markowitz Optimization, Covariance Matrix, Quadratic Programming, Eigenvalue Decomposition, Value-at-Risk.

---

## I. Introduction

Modern Portfolio Theory, introduced by Harry Markowitz in 1952, formalizes the trade-off between risk and return in asset allocation. The core insight is that portfolio risk is not the sum of individual asset risks, but a function of their co-movements, captured mathematically by the covariance matrix \(\Sigma\). 

Portfolio variance is the quadratic form \(\sigma_p^2 = w^T \Sigma w\), where \(w\) is the vector of asset weights. Minimizing this variance subject to a budget constraint constitutes a quadratic program (QP). This paper demonstrates a full end-to-end implementation of this linear-algebraic pipeline: from data ingestion and matrix construction to numerical optimization and risk analysis, culminating in a user-facing dashboard.

## II. Mathematical Foundations

### A. Covariance and Correlation

Given a \(T \times N\) matrix of daily returns \(X\), the mean-centered returns are \(\tilde{X} = X - \mathbf{1}_T \mu^T\). The sample covariance matrix is:
\[
 \Sigma = \frac{1}{T-1} \tilde{X}^T \tilde{X} 
\]
This is a Gram matrix, ensuring it is positive semi-definite (PSD). The correlation matrix is obtained by diagonal scaling: \(\text{Corr} = D^{-1} \Sigma D^{-1}\), where \(D = \text{diag}(\sigma_i)\).

### B. Eigendecomposition

By the spectral theorem, a symmetric covariance matrix can be decomposed as:
\[
 \Sigma = V \Lambda V^T 
\]
where \(V\) is an orthogonal matrix of eigenvectors and \(\Lambda\) is a diagonal matrix of non-negative eigenvalues \(\lambda_1 \ge \dots \ge \lambda_N \ge 0\). The eigenvalue \(\lambda_k\) represents the variance of the portfolio in the direction of eigenvector \(v_k\).

### C. Markowitz Optimization

The Minimum-Variance portfolio solves:
\[
 \min_w \; w^T \Sigma w \quad \text{s.t.} \quad \mathbf{1}^T w = 1, \; 0 \le w_i \le 1 
\]
Without the long-only constraint, the closed form is \(w^* = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}\). With the long-only constraint, we solve a convex QP via SLSQP.

The Maximum-Sharpe portfolio maximizes the risk-adjusted return:
\[
 \max_w \; \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}} 
\]
This is a quasi-convex problem, solved numerically.

## III. Implementation and Results

### A. Data and Covariance

We evaluated a 7-asset US equity portfolio (AAPL, MSFT, JPM, JNJ, PG, XOM, AMZN) from 2018-01-02 to 2026-11-13 (2230 trading days). The annualized covariance matrix was constructed, and its eigenvalues analyzed. The first 3 principal components explained approximately 77% of total variance, indicating that risk is largely driven by a single "market mode."

### B. Optimization Engine

Using SciPy's SLSQP solver with \(w_0 = 1/N \mathbf{1}\), we computed three portfolios:

**Table I: Portfolio Comparison (r_f = 2%)**
| Strategy | Return | Volatility | Variance | Sharpe |
|---|---|---|---|---|
| EqualWeight | 19.88% | 18.43% | 3.40% | 0.970 |
| MinVolatility | 14.79% | 15.87% | 2.52% | 0.805 |
| MaxSharpe | 22.36% | 20.18% | 4.07% | 1.009 |

The Maximum-Sharpe portfolio concentrated 56% of its weight in high-return tech stocks (AAPL and MSFT), while the Min-Variance portfolio heavily weighted defensive sectors (JNJ and PG at 69%).

### C. Risk Metrics

We calculated historical Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) at a 95% confidence level, as well as Maximum Drawdown (MDD). For the Max-Sharpe portfolio, VaR (95%) was 1.84%, CVaR was 2.99%, and MDD was -31.42%. The Min-Variance portfolio exhibited the lowest tail risk (VaR 1.39%, MDD -27.48%).

## IV. Conclusion

We successfully implemented the Markowitz Mean-Variance framework using pure Linear Algebra and numerical optimization. The system correctly constructs the covariance matrix, performs eigendecomposition for risk analysis, and generates efficient portfolios. The interactive dashboard allows users to visualize the efficient frontier and risk metrics in real time.

Future work will focus on incorporating covariance shrinkage (Ledoit-Wolf) to improve out-of-sample robustness and integrating transaction costs into the optimization objective.

## References

[1] H. Markowitz, "Portfolio Selection," *The Journal of Finance*, vol. 7, no. 1, pp. 77–91, 1952.  
[2] G. Strang, *Introduction to Linear Algebra*, 5th ed. Wellesley-Cambridge Press, 2016.  
[3] S. Boyd and L. Vandenberghe, *Convex Optimization*. Cambridge University Press, 2004.