// Dynamic Bayesian Bradley-Terry model with draws (Davidson form).
// Team strengths follow a Gaussian random walk over time periods.
//
//   beta[j, 1] = sigma0 * z0[j]
//   beta[j, t] = beta[j, t-1] + sigma_drift * eps[j, t-1]
//
//   d = beta[team1, period] - beta[team2, period] + h * home
//   P(team1 wins) = exp(d/2) / (exp(d/2) + exp(-d/2) + nu)
//   P(draw)       = nu       / (exp(d/2) + exp(-d/2) + nu)
//   P(team2 wins) = exp(-d/2)/ (exp(d/2) + exp(-d/2) + nu)
//
// y = 1: team2 wins, 2: draw, 3: team1 wins.

data {
  int<lower=1> N;                         // matches
  int<lower=1> J;                         // teams
  int<lower=2> T;                         // time periods
  array[N] int<lower=1, upper=J> team1;
  array[N] int<lower=1, upper=J> team2;
  array[N] int<lower=1, upper=T> period;
  array[N] int<lower=1, upper=3> y;
  vector<lower=0, upper=1>[N] home;       // 1 if team1 plays at home, else 0 (neutral)
}

parameters {
  vector[J] z0;                           // initial strengths (standardised)
  matrix[J, T - 1] eps;                   // random-walk innovations (standardised)
  real<lower=0> sigma0;                   // spread of initial strengths
  real<lower=0> sigma_drift;              // drift per period
  real h;                                 // home advantage (logit scale)
  real log_nu;                            // draw parameter
}

transformed parameters {
  matrix[J, T] beta;
  real<lower=0> nu = exp(log_nu);
  beta[, 1] = sigma0 * z0;
  for (t in 2:T) {
    beta[, t] = beta[, t - 1] + sigma_drift * eps[, t - 1];
  }
}

model {
  z0 ~ std_normal();
  to_vector(eps) ~ std_normal();
  sigma0 ~ normal(0, 1);                  // half-normal (lower bound 0)
  sigma_drift ~ normal(0, 0.3);           // half-normal
  h ~ normal(0, 0.5);
  log_nu ~ normal(0, 1);

  for (n in 1:N) {
    real d = beta[team1[n], period[n]] - beta[team2[n], period[n]] + h * home[n];
    y[n] ~ categorical_logit([-d / 2, log_nu, d / 2]');
  }
}

generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    real d = beta[team1[n], period[n]] - beta[team2[n], period[n]] + h * home[n];
    vector[3] lg = [-d / 2, log_nu, d / 2]';
    y_rep[n] = categorical_logit_rng(lg);
    log_lik[n] = categorical_logit_lpmf(y[n] | lg);
  }
}
