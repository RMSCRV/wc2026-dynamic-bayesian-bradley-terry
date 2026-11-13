# Bradley-Terry Model: Technical Note

This note explains the Bradley-Terry (BT) model used in `WC2026_BT_vs_Elo.Rmd`.
It gives the maths, the fitting method, the extensions and the results.

Rendered report: [WC2026_BT_vs_Elo on RPubs](https://rpubs.com/PRC/1475935)

## 1. Problem

Input: results of paired contests between teams.
Output: one strength value for each team.
Use: predict the result of any future match, including pairs that never met.

Raw tables (wins, goals) do not work well. Teams play different opponents.
A win against a strong team must count more than a win against a weak team.
BT solves this. It rates all teams on one scale from all results together.

## 2. Model

Each team *i* has a positive strength $\pi_i$. Write $\pi_i = e^{\beta_i}$.

$$
P(i \text{ beats } j) = \frac{\pi_i}{\pi_i + \pi_j} = \frac{1}{1 + e^{-(\beta_i - \beta_j)}} = \mathrm{logit}^{-1}(\beta_i - \beta_j)
$$

Key points:

- The probability depends only on the difference $\beta_i - \beta_j$.
- Equal strength gives probability 0.5.
- BT is a binomial GLM with a logit link. Each match has a design row with +1 for team *i*, -1 for team *j*, and no intercept.

## 3. Estimation

Let $y_{ij} = 1$ if *i* wins, else 0. Let $w_{ij}$ be the weight of the match.

**Log-likelihood**

$$
\ell(\beta) = \sum_{\text{matches}} w_{ij}\Big[\, y_{ij}\log p_{ij} + (1-y_{ij})\log(1-p_{ij}) \Big], \qquad p_{ij} = \mathrm{logit}^{-1}(\beta_i-\beta_j)
$$

**Score equations.** Set $\partial \ell / \partial \beta_i = 0$:

$$
\sum_{j} w_{ij}\,(y_{ij} - p_{ij}) = 0 \quad \text{for each team } i
$$

Meaning: the weighted wins of each team equal its weighted expected wins.

**Identifiability.** Adding a constant to all $\beta_i$ changes nothing.
Fix one reference team at $\beta = 0$ (or force $\sum \beta_i = 0$).
All abilities are then relative to the reference team.

**Algorithm.** `BradleyTerry2::BTm` fits the model by iteratively reweighted least squares (IRLS), as for any GLM.
An older option is the MM algorithm (Zermelo, Hunter):

$$
\pi_i^{\text{new}} = \frac{W_i}{\sum_{j} \dfrac{n_{ij}}{\pi_i + \pi_j}}
$$

$W_i$ is the number of wins of team *i*. $n_{ij}$ is the number of matches between *i* and *j*.
Rescale the $\pi_i$ after each step.

**Existence of the estimate.** The maximum likelihood estimate (MLE) is finite only if
the comparison graph is strongly connected. That means: every team can reach every other team
through a chain of wins. Two problems occur:

| Problem | Cause | Effect |
|---|---|---|
| Perfect record | A team never loses | $\hat\beta_i \to +\infty$ |
| Disconnected graph | Groups of teams never meet | Abilities cannot be compared across groups |

Fixes: Firth bias reduction (`br = TRUE` in `BTm`), a prior on $\beta$, or a different reference team per component.

## 4. Extensions

### 4.1 Time weights

Old results are less informative. Give each match an exponential weight:

$$
w = e^{-\xi\,\Delta t} \cdot c_{\text{comp}}, \qquad \xi = \frac{\ln 2}{912\ \text{days}}
$$

$\Delta t$ is the age of the match. The half-life is 912 days (about 2.5 years).
$c_{\text{comp}}$ is the competition weight (World Cup 3.0, friendly 0.5).
Weights go into the log-likelihood. Scale them to a maximum of 1.

### 4.2 Draws

Plain BT has two outcomes. Football has three. Options:

1. **Drop draws.** Simple, but it removes about 25% of matches and loses information.
2. **Half-wins (used here).** A draw becomes two rows with half weight: one win for each team. The fit stays a standard BT fit.
3. **Rao-Kupper** (threshold $\theta > 1$):

$$
P(i\text{ wins}) = \frac{\pi_i}{\pi_i + \theta\pi_j}, \qquad
P(\text{draw}) = \frac{(\theta^2-1)\,\pi_i\pi_j}{(\pi_i+\theta\pi_j)(\theta\pi_i+\pi_j)}
$$

4. **Davidson** (parameter $\nu > 0$):

$$
P(i\text{ wins}) = \frac{\pi_i}{\pi_i + \pi_j + \nu\sqrt{\pi_i\pi_j}}, \qquad
P(\text{draw}) = \frac{\nu\sqrt{\pi_i\pi_j}}{\pi_i + \pi_j + \nu\sqrt{\pi_i\pi_j}}
$$

With the half-win fit, $p$ is an expected score (win = 1, draw = 0.5). The code then sets:

$$
P(\text{draw}) = \bar d \cdot 4p(1-p), \qquad P(\text{win}) = p - \tfrac{1}{2}P(\text{draw}), \qquad P(\text{loss}) = 1 - p - \tfrac{1}{2}P(\text{draw})
$$

$\bar d$ is the weighted draw rate in the training data. Draws are most likely when $p = 0.5$.

### 4.3 Home advantage

Add a constant *h* when team *i* plays at home:

$$
P(i \text{ beats } j) = \mathrm{logit}^{-1}(\beta_i - \beta_j + h)
$$

In `BTm`, use a covariate `at.home` in the formula. This project does not include *h* in BT.
It is estimated in the Poisson model instead. The back-test predicts at a neutral venue.

### 4.4 Covariates

Replace $\beta_i$ by $\beta_i = \sum_k \gamma_k x_{ik} + u_i$. Example: a league-strength score as covariate (the STAT-479 repo does this for club teams).

### 4.5 Dynamic Bayesian model (the headline model of this project)

The static fit gives one number per team and cannot say how fast a team changes.
The dynamic Bayesian model fixes both points.

**Time.** Split time into half-year periods $t = 1, \dots, T$. Team *j* has strength $\beta_{j,t}$.
The strength follows a Gaussian random walk:

$$
\beta_{j,1} = \sigma_0 z_j, \qquad \beta_{j,t} = \beta_{j,t-1} + \sigma_{\text{drift}}\,\varepsilon_{j,t}, \qquad z_j,\ \varepsilon_{j,t} \sim N(0,1)
$$

$\sigma_{\text{drift}}$ is learned from the data. A large value means ratings change fast.

**Likelihood (Davidson form with home advantage).** Let $d = \beta_{i,t} - \beta_{j,t} + h\cdot\text{home}$.

$$
P(i\text{ wins}) = \frac{e^{d/2}}{e^{d/2} + e^{-d/2} + \nu}, \quad
P(\text{draw}) = \frac{\nu}{e^{d/2} + e^{-d/2} + \nu}, \quad
P(j\text{ wins}) = \frac{e^{-d/2}}{e^{d/2} + e^{-d/2} + \nu}
$$

**Priors.**

| Parameter | Prior |
|---|---|
| $\sigma_0$ | half-normal(0, 1) |
| $\sigma_{\text{drift}}$ | half-normal(0, 0.3) |
| $h$ | normal(0, 0.5) |
| $\log \nu$ | normal(0, 1) |

**Posterior.** The target is $p(\beta, \sigma_0, \sigma_{\text{drift}}, h, \nu \mid y)$.
It has no closed form. The code samples it with Hamiltonian Monte Carlo (HMC), using the No-U-Turn sampler (NUTS) in Stan (`dynamic_bt.stan`).

- HMC adds a momentum variable $p$ and simulates the Hamiltonian dynamics of $H(\theta, p) = -\log \pi(\theta \mid y) + \tfrac{1}{2}p^\top M^{-1}p$ with the leapfrog integrator.
- NUTS chooses the path length automatically. Stan tunes the step size and mass matrix in the warm-up phase.
- The innovations $\varepsilon$ are sampled as standard normals and scaled by $\sigma_{\text{drift}}$ (non-centred form). This removes the funnel-shaped geometry that stops HMC from moving.

**Sampler checks.** Report R-hat (target below 1.01), effective sample size, divergent transitions (target 0), E-BFMI (target above 0.3) and trace plots.

**Posterior predictive checks.** Simulate replicated results $y^{\text{rep}}$ from the posterior. Compare features of $y^{\text{rep}}$ with the real data:

$$
\text{ppp} = P\big(T(y^{\text{rep}}) \ge T(y) \mid y\big)
$$

A ppp near 0 or 1 means the model misses that feature. The Rmd checks: result frequencies, draw rate by strength gap, home-win share over time, and points per game by team.

**Forecasts.** To predict a match *k* periods ahead, add drift to the last fitted strengths:
$\beta_{j,T+k} = \beta_{j,T} + \sqrt{k}\,\sigma_{\text{drift}}\,\varepsilon$. Average the match probabilities over the posterior draws.
The tournament simulation uses a new posterior draw in every simulated tournament, so parameter uncertainty reaches the champion probabilities.

Related models in the repos: Serie A repo (Gibbs sampler with Gamma priors, static), STAT-479 repo (BT with a league-strength covariate, static).

## 5. Link to Elo

Elo is BT with an online update. The expected score is:

$$
E_i = \frac{1}{1 + 10^{-(R_i - R_j)/400}}
$$

This equals BT with $\beta = R \ln 10 / 400$. The update after each match is:

$$
R_i \leftarrow R_i + K\,(S_i - E_i)
$$

$S_i$ is 1, 0.5 or 0. This is one stochastic gradient step on the BT log-likelihood.
Differences from the BT fit:

| | Bradley-Terry fit | Elo |
|---|---|---|
| Method | Maximum likelihood on all data at once | One update per match, in date order |
| Time effect | Explicit weights | Implicit (recent matches move the rating most) |
| Draws | Half-wins | Score 0.5 |
| Uncertainty | Standard errors | None |

This project uses K = 20 to 60 by competition and a margin multiplier $\log(|gd|+1)+1$.

## 6. Results

Back-test: fit on matches before 2022-01-01, score on 1,645 later matches.
Lower is better for all columns except accuracy. RPS is the ranked probability score.

| Model | RPS | Log-loss | Brier | Accuracy |
|---|---|---|---|---|
| Baseline (outcome frequencies) | 0.2281 | 1.0588 | 0.6391 | 46.6% |
| Static BT, decisive matches only (MLE) | 0.2009 | 1.0231 | 0.5938 | 56.4% |
| Static BT, draws as half-wins (MLE) | 0.1888 | 0.9557 | 0.5639 | 56.7% |
| Elo | 0.1933 | 0.9682 | 0.5725 | 55.6% |
| Bivariate Poisson | 0.1894 | 0.9433 | 0.5594 | 55.0% |
| **Dynamic Bayesian BT (HMC)** | **0.1857** | **0.9313** | **0.5515** | **56.9%** |

Findings:

1. The dynamic Bayesian model is best on all four columns.
2. Keeping draws improves static BT (RPS 0.2009 to 0.1888). Letting strengths drift improves it further (0.1857).
3. The Bayesian model beats Elo on RPS by 0.0076.
4. The Poisson shared-goal rate is estimated at 0, so that model is independent Poisson here.

**Sampler (final fit, 4 chains x 1,000 draws).** 0 divergences, max R-hat 1.009, min ESS 893, min E-BFMI 0.81, no tree-depth hits.
The drift volatility mixes slowest (R-hat 1.025, ESS 211). Posterior mean $\sigma_{\text{drift}} = 0.17$ per half-year (95% interval 0.06 to 0.26),
home advantage $h = 0.50$ (0.40 to 0.60), draw parameter $\nu = 0.93$.

**Posterior predictive checks.**

| Check | Result |
|---|---|
| Overall win / draw / loss shares | Good (ppp 0.49 to 0.53) |
| Draw rate by strength gap | **Misfit.** Largest-gap quintile: observed 8.8% against 13.0% replicated (ppp 1.00). The Davidson form gives mismatched teams too many draws |
| First-listed win share over time | One period unusual (ppp 0.01). The home-advantage term may vary over time |
| Points per game by team | 0 of 121 teams outside the 90% band (about 10% expected). The check is conservative, because the data were used to fit the model |

Champion probability (10,000 simulations per model, seed 2026, Monte Carlo SE at most 0.5 points).
The Bayesian column uses a new posterior draw in every simulated tournament:

| Team | Dynamic Bayesian BT | Poisson | Elo |
|---|---|---|---|
| Argentina | 15.2% | 6.4% | 19.8% |
| Colombia | 13.5% | 5.9% | 5.2% |
| Spain | 12.5% | 11.7% | 34.4% |
| Brazil | 8.3% | 6.4% | 3.1% |
| France | 7.5% | 6.8% | 7.6% |

Elo gives the most concentrated forecast (top team 34.4%, about 5.7 effective contenders).
The Bayesian model is in between (top team 15.2%, 12.4 effective contenders). Poisson is the flattest (18.8).
The three models disagree on the favourite: Argentina, Spain and Spain. The credible intervals of the top teams' strengths overlap, so the data cannot separate them.

## 7. Limits

- Data ends on 2025-07-06. CONMEBOL qualifiers are not in the source.
- The 48 teams are the strongest by rating, not the real qualifiers.
- The knockout bracket is random. There is no host advantage.
- Static BT and Elo have no home term. The back-test is a single time split.
- The Bayesian model has no competition weights, and venue is only approximated (tournament finals are treated as neutral).
- The Davidson draw model overpredicts draws between mismatched teams (see the posterior predictive checks).
- The posterior is conditional on the model. The checks show where the model does and does not describe the data.
- Results show model behaviour on this data. They are not betting advice.

## 8. How to run

1. Install Rtools 4.5 (C++ compiler for Stan): https://cran.r-project.org/bin/windows/Rtools/rtools45/rtools.html
2. Close RStudio. In plain R, update `dplyr` to 1.2.0 or newer and install the packages once:
   `install.packages(c("dplyr", "vctrs", "gt", "glmnet", "rstan"))`.
3. Open `WC2026_BT_vs_Elo.Rmd` in RStudio. Keep `dynamic_bt.stan` and the `.rds` cache files in the same folder.
4. Click Knit. The first run with `refit: true` (or with no cache) compiles the Stan model (1 to 3 minutes) and samples (several minutes). Normal knits read the cached posterior from `bt_dyn_train.rds` and `bt_dyn_final.rds`.
5. Use Knit with Parameters to change `n_sims`, `seed`, `cutoff`, `refit`, `chains`, `iter`, `adapt_delta`.

## 9. References

- Bradley, R. A., Terry, M. E. (1952). Rank analysis of incomplete block designs. *Biometrika* 39.
- Zermelo, E. (1929). Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem.
- Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models. *Annals of Statistics* 32.
- Rao, P. V., Kupper, L. L. (1967). Ties in paired-comparison experiments. *JASA* 62.
- Davidson, R. R. (1970). On extending the Bradley-Terry model to accommodate ties. *JASA* 65.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika* 80.
- Turner, H., Firth, D. (2012). Bradley-Terry models in R: the BradleyTerry2 package. *J. Stat. Software* 48.
