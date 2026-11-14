# Project Presentation Outline
**LA2133: Theory and Practice of Linear Algebra**
**Financial Portfolio Optimization & Risk Allocator**

## Slide 1: Title Slide
- Project Title, Team Names, Course Code, Guide: Prof. Manjunath S.

## Slide 2: The Problem
- The Markowitz Portfolio Problem: How to allocate capital to maximize return and minimize risk?
- Relevance: Foundation of quantitative finance. Pure application of Linear Algebra.

## Slide 3: Mathematical Foundation (Covariance)
- \( \sigma_p^2 = w^T \Sigma w \) (Quadratic Form).
- Covariance Matrix \( \Sigma = \frac{1}{T-1} \tilde{X}^T \tilde{X} \).
- Show the Correlation Matrix (Heatmap).

## Slide 4: Eigendecomposition
- \( \Sigma = V \Lambda V^T \) (Spectral Theorem).
- PSD matrix -> \( \lambda_i \ge 0 \).
- Show Eigenvalue Spectrum chart (first 3 PCs explain 77% of risk).

## Slide 5: Optimization Engine
- Min-Variance: \( \min w^T \Sigma w \).
- Max-Sharpe: \( \max (w^T \mu - r_f) / \sqrt{w^T \Sigma w} \).
- Solver: SLSQP with long-only constraints.
- Show the Efficient Frontier plot (Bullet shape).

## Slide 6: Real-Data Results
- Table of 7 assets: EqualWeight, MinVol, MaxSharpe.
- Key insight: MaxSharpe (1.009) beats EqualWeight (0.970).

## Slide 7: Risk Metrics
- VaR (95%), CVaR (95%), Max Drawdown.
- Show the Risk Dashboard screenshot.

## Slide 8: The Interactive Dashboard
- Built with Streamlit.
- Live demo / screenshot of the UI.
- User can select tickers, adjust risk-free rate, view frontier.

## Slide 9: Conclusion & Future Work
- What we did: From-scratch implementation of Markowitz on 7 stocks.
- What's next: Covariance shrinkage, transaction costs, live trading integration.

## Slide 10: Q&A