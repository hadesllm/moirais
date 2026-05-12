# SPDX-License-Identifier: GPL-2.0-only
#' Ghosal Bayesian-nonparametrics suite (R parity)
#'
#' Twenty callables implementing the Ghosal & van der Vaart (2017)
#' "Fundamentals of Nonparametric Bayesian Inference" formula index:
#' Dirichlet- and beta-process posteriors, stick-breaking, posterior
#' consistency / contraction, GP regression, semiparametric BvM,
#' spike-and-slab wavelets, etc.  Each function returns a named list
#' with the same keys as its Python sibling in `morie.fn.<name>`.
#'
#' Where possible we use base-R + recommended packages (mvtnorm).
#' Heavy dependencies (`dirichletprocess`, `BNPdensity`, `MCMCpack`)
#' are NOT required — small implementations are inlined.
#'
#' @references
#' Ghosal, S. & van der Vaart, A. W. (2017). Fundamentals of
#'   Nonparametric Bayesian Inference. Cambridge University Press.
#' @name ghosal_bnp
NULL


# --- helpers -----------------------------------------------------------

.gh_have <- function(pkg) requireNamespace(pkg, quietly = TRUE)


# --- ghdir: DP posterior ----------------------------------------------

#' Dirichlet-process posterior (conjugate update)
#'
#' Posterior of `G | X_{1:n}` for `G ~ DP(alpha, G0)` with
#' `G0 = N(base_mean, base_sd^2)`.  Returns the posterior-mean CDF
#' evaluated on a grid plus the headline `estimate` at `mean(x)`.
#' @param x numeric vector.
#' @param alpha concentration.
#' @param base_mean,base_sd base measure (N).
#' @param grid optional grid (default: 51 pts spanning x).
#' @return named list with `estimate`, `cdf_grid`, `cdf_post`,
#'   `cdf_var`, `alpha_post`, `n`, `method`.
#' @export
ghosal_dirichlet_posterior <- function(x, alpha = 1.0, base_mean = 0,
                                         base_sd = 1, grid = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (is.null(grid)) {
    if (n == 0) grid <- seq(base_mean - 3*base_sd, base_mean + 3*base_sd, length.out = 51)
    else {
      pad <- max(1e-6, 0.1 * (max(x) - min(x) + 1))
      grid <- seq(min(x) - pad, max(x) + pad, length.out = 51)
    }
  }
  alpha_post <- alpha + n
  G0_t <- stats::pnorm(grid, mean = base_mean, sd = base_sd)
  emp_t <- if (n > 0) sapply(grid, function(t) sum(x <= t)) else rep(0, length(grid))
  F_post <- (alpha * G0_t + emp_t) / alpha_post
  var_post <- F_post * (1 - F_post) / (alpha_post + 1)
  t0 <- if (n > 0) mean(x) else base_mean
  G0_t0 <- stats::pnorm(t0, mean = base_mean, sd = base_sd)
  emp_t0 <- if (n > 0) sum(x <= t0) else 0
  est <- (alpha * G0_t0 + emp_t0) / alpha_post
  list(estimate = est, alpha_post = alpha_post, n = n,
       cdf_grid = grid, cdf_post = F_post, cdf_var = var_post,
       method = "Dirichlet process posterior (conjugate)")
}


# --- ghdpm: DP-mixture density via collapsed Gibbs --------------------

#' DP mixture density estimate (Neal 2000 algorithm 3)
#' @param x numeric vector
#' @param alpha,sigma DP and within-cluster sd (sigma defaults to Silverman bw)
#' @param grid evaluation grid
#' @param n_iter,burn,seed Gibbs settings
#' @return named list with `estimate`, `grid`, `density`, `k_post`, `n`
#' @export
ghosal_dpmixture_density <- function(x, alpha = 1.0, sigma = NULL,
                                       grid = NULL, n_iter = 120, burn = 40,
                                       seed = 0) {
  set.seed(seed)
  x <- as.numeric(x); n <- length(x)
  if (n == 0) return(list(estimate = NA_real_, n = 0,
                          method = "DP-mixture density (empty input)"))
  if (is.null(sigma)) {
    s <- if (n > 1) sd(x) else 1
    sigma <- 1.06 * max(s, 1e-6) * n^(-1/5)
  }
  sigma <- max(sigma, 1e-6)
  m0 <- mean(x); s0 <- if (n > 1) sd(x) else 1
  s0 <- max(s0, 1e-3)
  if (is.null(grid)) grid <- seq(min(x) - s0, max(x) + s0, length.out = 51)
  labels <- rep(0L, n)
  k_chain <- integer(0); f_chain <- list()
  new_log <- function(xi) stats::dnorm(xi, mean = m0, sd = sqrt(sigma^2 + s0^2), log = TRUE)
  cluster_post <- function(xs) {
    nk <- length(xs); v <- 1 / (1/s0^2 + nk/sigma^2)
    m <- v * (m0/s0^2 + sum(xs)/sigma^2)
    list(m = m, v = v)
  }
  in_log <- function(xi, xs) {
    cp <- cluster_post(xs)
    stats::dnorm(xi, mean = cp$m, sd = sqrt(cp$v + sigma^2), log = TRUE)
  }
  for (it in seq_len(n_iter)) {
    for (i in seq_len(n)) {
      old <- labels[i]; labels[i] <- -1L
      uniq <- sort(unique(labels[labels >= 0]))
      lps <- numeric(length(uniq) + 1L)
      for (j in seq_along(uniq)) {
        xs <- x[labels == uniq[j]]
        lps[j] <- log(length(xs)) + in_log(x[i], xs)
      }
      lps[length(uniq) + 1L] <- log(alpha) + new_log(x[i])
      lps <- lps - max(lps); probs <- exp(lps); probs <- probs/sum(probs)
      choice <- sample.int(length(probs), 1, prob = probs)
      if (choice == length(uniq) + 1L) {
        labels[i] <- if (length(uniq)) max(uniq) + 1L else 0L
      } else {
        labels[i] <- uniq[choice]
      }
    }
    if (it > burn) {
      uniq <- sort(unique(labels))
      f <- numeric(length(grid))
      for (k in uniq) {
        xs <- x[labels == k]; cp <- cluster_post(xs)
        f <- f + (length(xs) / (alpha + n)) *
          stats::dnorm(grid, mean = cp$m, sd = sqrt(cp$v + sigma^2))
      }
      f <- f + (alpha / (alpha + n)) *
        stats::dnorm(grid, mean = m0, sd = sqrt(sigma^2 + s0^2))
      f_chain[[length(f_chain) + 1L]] <- f
      k_chain <- c(k_chain, length(uniq))
    }
  }
  density <- Reduce("+", f_chain) / length(f_chain)
  num <- sum(diff(grid) * (head(density * grid, -1) + tail(density * grid, -1))) / 2
  den <- sum(diff(grid) * (head(density, -1) + tail(density, -1))) / 2
  est <- num / max(den, 1e-12)
  list(estimate = est, grid = grid, density = density,
       k_post = mean(k_chain), n = n, alpha = alpha, sigma = sigma,
       method = "DP-mixture density via collapsed Gibbs (Neal 2000 Alg 3)")
}


# --- ghstk: truncated stick-breaking ----------------------------------

#' Truncated stick-breaking representation of DP(alpha, G0).
#' @export
ghosal_stick_breaking_trunc <- function(x, alpha = 1.0, K = 50, seed = 0,
                                          base_mean = NULL, base_sd = NULL) {
  set.seed(seed)
  x <- as.numeric(x); n <- length(x)
  if (is.null(base_mean)) base_mean <- if (n) mean(x) else 0
  if (is.null(base_sd))   base_sd   <- if (n > 1) sd(x) else 1
  base_sd <- max(base_sd, 1e-6)
  V <- stats::rbeta(K, 1, alpha)
  log_cum <- c(0, cumsum(log1p(-V[-K])))
  w <- V * exp(log_cum)
  theta <- stats::rnorm(K, mean = base_mean, sd = base_sd)
  t0 <- if (n) mean(x) else base_mean
  est <- sum(w * (theta <= t0))
  trunc_bound <- (alpha / (alpha + 1))^K
  list(estimate = est, weights = w, atoms = theta,
       effective_K = sum(w > 1e-3),
       trunc_err_bound = trunc_bound, n = n,
       method = "Truncated stick-breaking DP draw (Sethuraman 1994)")
}


# --- ghcon: posterior consistency ------------------------------------

#' Schwartz posterior-consistency diagnostic (Bayesian bootstrap).
#' @export
ghosal_posterior_consistency <- function(x, ref_loc = NULL, ref_scale = NULL,
                                           eps = 0.1, K = 200, seed = 0) {
  set.seed(seed)
  x <- as.numeric(x); n <- length(x)
  if (n == 0) return(list(estimate = NA_real_, n = 0,
                          method = "Schwartz consistency (empty)"))
  xs <- sort(x)
  grid <- seq(xs[1] - 1, xs[n] + 1, length.out = 200)
  if (is.null(ref_loc) || is.null(ref_scale)) {
    F_ref <- sapply(grid, function(t) sum(xs <= t)) / n
  } else {
    F_ref <- stats::pnorm(grid, ref_loc, ref_scale)
  }
  ks <- numeric(K)
  for (k in seq_len(K)) {
    if (.gh_have("MCMCpack")) {
      u <- as.numeric(MCMCpack::rdirichlet(1, rep(1, n)))
    } else {
      g <- stats::rgamma(n, shape = 1, rate = 1); u <- g / sum(g)
    }
    cdf <- cumsum(u)  # already in sorted order since xs is sorted
    idx <- findInterval(grid, xs)
    F_draw <- ifelse(idx == 0, 0, cdf[pmin(pmax(idx, 1L), n)])
    ks[k] <- max(abs(F_draw - F_ref))
  }
  list(estimate = mean(ks > eps), ks_mean = mean(ks),
       ks_se = sd(ks) / sqrt(K),
       schwartz_bound = exp(-2 * n * eps^2),
       n = n, eps = eps,
       method = "Schwartz consistency (Bayesian-bootstrap proxy)")
}


# --- ghcrt: contraction rate ----------------------------------------

#' Minimax posterior-contraction rate eps_n = n^{-beta/(2beta+d)}.
#' @export
ghosal_contraction_rate <- function(x, beta = 1.0, d = 1) {
  n <- length(x)
  if (n <= 1) return(list(estimate = NA_real_, n = n,
                          method = "Contraction rate (n too small)"))
  eps_n <- n^(-beta / (2 * beta + d))
  list(estimate = eps_n,
       log_rate_correction = (log(n))^(beta/(2*beta+d)) * eps_n,
       parametric_rate = n^(-0.5),
       n = n, beta = beta, d = d,
       method = "Minimax contraction rate n^{-beta/(2beta+d)}")
}


# --- ghgps / ghgpm: GP regression -----------------------------------

.gh_pairwise_sq <- function(a, b = a) {
  outer(rowSums(a^2), rowSums(b^2), "+") - 2 * a %*% t(b)
}

#' GP posterior mean with squared-exponential kernel.
#' @export
ghosal_gp_squared_exponential <- function(x, y, length_scale = NULL,
                                            sigma_f = 1.0, noise = NULL,
                                            x_star = NULL) {
  if (is.vector(x)) x <- matrix(as.numeric(x), ncol = 1L) else x <- as.matrix(x)
  y <- as.numeric(y); n <- nrow(x)
  if (is.null(x_star)) {
    x_star <- x
  } else if (is.vector(x_star)) {
    x_star <- matrix(as.numeric(x_star), ncol = 1L)
  } else {
    x_star <- as.matrix(x_star)
  }
  sq <- pmax(.gh_pairwise_sq(x), 0)
  if (is.null(length_scale)) {
    d <- sqrt(sq[upper.tri(sq)])
    length_scale <- if (length(d)) max(stats::median(d[d > 0]), 1e-3) else 1
  }
  if (is.null(noise)) noise <- max(0.1 * stats::sd(y), 1e-3)
  kernel <- function(a, b) {
    sq_ab <- pmax(.gh_pairwise_sq(a, b), 0)
    sigma_f^2 * exp(-sq_ab / (2 * length_scale^2))
  }
  K <- kernel(x, x) + noise^2 * diag(n)
  K_s <- kernel(x_star, x)
  K_ss_diag <- rep(sigma_f^2, nrow(x_star))
  L <- chol(K + 1e-8 * diag(n))
  alpha_ <- backsolve(L, forwardsolve(t(L), y))
  mu <- as.numeric(K_s %*% alpha_)
  v <- forwardsolve(t(L), t(K_s))
  var <- K_ss_diag - colSums(v^2)
  sd_ <- sqrt(pmax(var, 0))
  list(estimate = mean(mu), se = mean(sd_), mu = mu, sd = sd_,
       length_scale = length_scale, noise = noise, n = n,
       method = "GP regression (squared-exponential kernel)")
}

#' GP posterior mean with Matern kernel.
#' @export
ghosal_gp_matern <- function(x, y, nu = 1.5, length_scale = NULL,
                              sigma_f = 1.0, noise = NULL, x_star = NULL) {
  if (is.vector(x)) x <- matrix(as.numeric(x), ncol = 1L) else x <- as.matrix(x)
  y <- as.numeric(y); n <- nrow(x)
  if (is.null(x_star)) {
    x_star <- x
  } else if (is.vector(x_star)) {
    x_star <- matrix(as.numeric(x_star), ncol = 1L)
  } else {
    x_star <- as.matrix(x_star)
  }
  sq <- pmax(.gh_pairwise_sq(x), 0)
  if (is.null(length_scale)) {
    d <- sqrt(sq[upper.tri(sq)])
    length_scale <- if (length(d)) max(stats::median(d[d > 0]), 1e-3) else 1
  }
  if (is.null(noise)) noise <- max(0.1 * stats::sd(y), 1e-3)
  kernel <- function(a, b) {
    sq_ab <- pmax(.gh_pairwise_sq(a, b), 0)
    r <- sqrt(sq_ab)
    if (isTRUE(all.equal(nu, 0.5)))  return(sigma_f^2 * exp(-r / length_scale))
    if (isTRUE(all.equal(nu, 1.5))) {
      t <- sqrt(3) * r / length_scale
      return(sigma_f^2 * (1 + t) * exp(-t))
    }
    if (isTRUE(all.equal(nu, 2.5))) {
      t <- sqrt(5) * r / length_scale
      return(sigma_f^2 * (1 + t + t^2/3) * exp(-t))
    }
    rr <- pmax(r, 1e-12); z <- sqrt(2*nu) * rr / length_scale
    coef <- sigma_f^2 * 2^(1 - nu) / gamma(nu)
    K <- coef * z^nu * besselK(z, nu)
    K[r < 1e-12] <- sigma_f^2
    K
  }
  K <- kernel(x, x) + noise^2 * diag(n)
  K_s <- kernel(x_star, x)
  K_ss_diag <- rep(sigma_f^2, nrow(x_star))
  L <- chol(K + 1e-8 * diag(n))
  alpha_ <- backsolve(L, forwardsolve(t(L), y))
  mu <- as.numeric(K_s %*% alpha_)
  v <- forwardsolve(t(L), t(K_s))
  var <- K_ss_diag - colSums(v^2)
  sd_ <- sqrt(pmax(var, 0))
  list(estimate = mean(mu), se = mean(sd_), mu = mu, sd = sd_,
       length_scale = length_scale, nu = nu, noise = noise, n = n,
       method = "GP regression (Matern kernel)")
}


# --- ghsve: Bernstein-polynomial sieve ------------------------------

.gh_bernstein <- function(u, K) {
  u <- pmin(pmax(u, 1e-12), 1 - 1e-12)
  B <- matrix(0, nrow = length(u), ncol = K)
  for (k in seq_len(K)) {
    B[, k] <- stats::dbeta(u, k, K - k + 1)
  }
  B
}

#' Bernstein-polynomial sieve density estimator (Petrone 1999).
#' @export
ghosal_sieve_prior <- function(x, K = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 3) return(list(estimate = NA_real_, n = n,
                         method = "Bernstein sieve (n<3)"))
  lo <- min(x) - 1e-6; hi <- max(x) + 1e-6
  u <- (x - lo) / (hi - lo)
  if (is.null(K)) K <- max(2, round(n^(1/3)))
  B <- .gh_bernstein(u, K)
  w <- rep(1/K, K)
  for (it in seq_len(60)) {
    num <- sweep(B, 2, w, "*")
    denom <- pmax(rowSums(num), 1e-12)
    gamma <- num / denom
    w_new <- colMeans(gamma); w_new <- w_new/sum(w_new)
    if (max(abs(w_new - w)) < 1e-8) { w <- w_new; break }
    w <- w_new
  }
  log_lik <- mean(log(pmax(B %*% w, 1e-12)))
  u_bar <- (mean(x) - lo) / (hi - lo)
  B_bar <- .gh_bernstein(u_bar, K)
  f_bar <- as.numeric((B_bar %*% w) / (hi - lo))
  list(estimate = f_bar, log_lik_per_obs = log_lik, weights = w, K = K, n = n,
       method = "Bernstein-polynomial sieve density (Petrone 1999, Ghosal 2001)")
}


# --- ghadp: adaptive contraction rates ------------------------------

#' Adaptive contraction rates over a smoothness grid.
#' @export
ghosal_adaptation <- function(x, betas = NULL, d = 1) {
  n <- length(x)
  if (is.null(betas)) betas <- seq(0.5, 3.0, length.out = 11)
  rates <- n^(-betas / (2*betas + d))
  best <- which.min(rates)
  list(estimate = rates[best], betas = betas, rates = rates,
       best_beta = betas[best], n = n, d = d,
       method = "Adaptive posterior contraction over Holder grid")
}


# --- ghbvm: Bernstein-von Mises -------------------------------------

#' BvM diagnostic for the mean functional under a DP prior.
#' @export
ghosal_bernstein_von_mises <- function(x, theta0 = NULL, B = 500, seed = 0) {
  set.seed(seed)
  x <- as.numeric(x); n <- length(x)
  if (n < 2) return(list(estimate = NA_real_, se = NA_real_, n = n,
                          method = "BvM (n<2)"))
  theta_hat <- mean(x); s <- sd(x)
  draws <- numeric(B)
  for (b in seq_len(B)) {
    g <- stats::rgamma(n, 1, 1); u <- g / sum(g)
    draws[b] <- sum(u * x)
  }
  z <- (draws - theta_hat) * sqrt(n) / max(s, 1e-12)
  ks <- suppressWarnings(stats::ks.test(z, "pnorm"))
  if (!is.null(theta0)) {
    sd_d <- sd(draws); wald <- (mean(draws) - theta0) / max(sd_d, 1e-12)
    wald_p <- 2 * (1 - stats::pnorm(abs(wald)))
  } else {
    wald <- NA_real_; wald_p <- NA_real_
  }
  list(estimate = mean(draws), se = sd(draws), theta_hat = theta_hat,
       z_ks_stat = unname(ks$statistic), z_ks_pvalue = ks$p.value,
       wald = wald, wald_pvalue = wald_p, n = n, B = B,
       method = "BvM for mean functional (Bayesian bootstrap)")
}


# --- ghreg / ghcls: GP regression / classification ------------------

#' GP nonparametric regression (wraps `ghosal_gp_squared_exponential`).
#' @export
ghosal_np_regression <- function(x, y, length_scale = NULL,
                                   sigma_f = 1.0, noise = NULL) {
  gp <- ghosal_gp_squared_exponential(x, y, length_scale = length_scale,
                                       sigma_f = sigma_f, noise = noise)
  yv <- as.numeric(y); mu <- gp$mu; sd_ <- gp$sd
  ss_tot <- sum((yv - mean(yv))^2); ss_res <- sum((yv - mu)^2)
  r2 <- 1 - ss_res / max(ss_tot, 1e-12)
  var_pred <- sd_^2 + gp$noise^2
  log_marg <- -0.5 * sum((yv - mu)^2 / var_pred + log(2*pi*var_pred))
  list(estimate = mean(mu), se = mean(sd_), mu = mu, sd = sd_,
       ci_lower = mu - 1.96 * sqrt(var_pred),
       ci_upper = mu + 1.96 * sqrt(var_pred),
       r2 = r2, log_marginal = log_marg,
       length_scale = gp$length_scale, noise = gp$noise, n = length(yv),
       method = "GP regression posterior")
}

#' Probit-GP classifier (Laplace approximation).
#' @export
ghosal_np_classification <- function(x, y, length_scale = NULL,
                                       sigma_f = 1.0, n_iter = 300, seed = 0) {
  set.seed(seed)
  x <- as.matrix(x); y <- as.numeric(y); n <- nrow(x)
  y_pm <- 2 * y - 1
  sq <- pmax(.gh_pairwise_sq(x), 0)
  if (is.null(length_scale)) {
    d <- sqrt(sq[upper.tri(sq)])
    length_scale <- if (length(d)) max(stats::median(d[d > 0]), 1e-3) else 1
  }
  K <- sigma_f^2 * exp(-sq / (2*length_scale^2)) + 1e-6 * diag(n)
  f <- rep(0, n)
  for (it in seq_len(n_iter)) {
    z <- y_pm * f
    phi <- stats::dnorm(z); Phi <- pmin(pmax(stats::pnorm(z), 1e-12), 1-1e-12)
    grad_ll <- y_pm * phi / Phi
    W <- pmax((phi/Phi) * (phi/Phi + z), 1e-8)
    sW <- sqrt(W)
    # B = I + diag(sW) %*% K %*% diag(sW)
    B <- diag(n) + (sW %o% sW) * K
    Lf <- tryCatch(chol(B), error = function(e) NULL)
    if (is.null(Lf)) break
    b <- W * f + grad_ll
    a <- b - sW * backsolve(Lf, forwardsolve(t(Lf), sW * (K %*% b)))
    f_new <- as.numeric(K %*% a)
    if (max(abs(f_new - f)) < 1e-6) { f <- f_new; break }
    f <- f_new
  }
  p_hat <- stats::pnorm(f); pred <- as.integer(p_hat >= 0.5)
  accuracy <- mean(pred == y)
  list(estimate = mean(p_hat), p_hat = p_hat, accuracy = accuracy,
       length_scale = length_scale, n = n,
       method = "Probit-link GP classifier (Laplace)")
}


# --- ghsrv / ghntr: beta-process / NTR survival ---------------------

.gh_surv_post <- function(t, ev, c, lam0) {
  t <- as.numeric(t); n <- length(t)
  if (n == 0) return(NULL)
  if (is.null(ev)) ev <- rep(1L, n)
  if (is.null(lam0)) lam0 <- 1 / max(mean(t), 1e-6)
  uniq <- sort(unique(t))
  Y <- sapply(uniq, function(tk) sum(t >= tk))
  dN <- sapply(uniq, function(tk) sum(t == tk & ev == 1))
  dH0 <- diff(c(0, uniq)) * lam0
  dHp <- (c * dH0 + dN) / (c + Y)
  S <- cumprod(1 - pmin(dHp, 1 - 1e-12))
  list(times = uniq, S = S, H = cumsum(dHp), dH = dHp, lam0 = lam0)
}

#' Beta-process posterior survival (Hjort 1990).
#' @export
ghosal_survival_beta_process <- function(time, event = NULL, c = 1.0,
                                          lam0 = NULL) {
  s <- .gh_surv_post(time, event, c, lam0)
  if (is.null(s)) return(list(estimate = NA_real_, n = 0,
                              method = "Beta-process survival (empty)"))
  t_med <- stats::median(time)
  idx <- findInterval(t_med, s$times)
  est <- if (idx >= 1) s$S[idx] else 1
  list(estimate = est, times = s$times, S_post = s$S, H_post = s$H,
       c = c, lam0 = s$lam0, n = length(time),
       method = "Beta-process posterior survival (Hjort 1990)")
}

#' Neutral-to-the-right posterior survival (Doksum 1974).
#' @export
ghosal_neutral_right <- function(time, event = NULL, c = 1.0, lam0 = NULL) {
  s <- .gh_surv_post(time, event, c, lam0)
  if (is.null(s)) return(list(estimate = NA_real_, n = 0,
                              method = "NTR process (empty)"))
  t_med <- stats::median(time)
  idx <- findInterval(t_med, s$times)
  est <- if (idx >= 1) s$S[idx] else 1
  list(estimate = est, times = s$times, S_post = s$S, H_post = s$H,
       c = c, lam0 = s$lam0, n = length(time),
       method = "Neutral-to-the-right posterior (Doksum 1974)")
}


# --- ghebp: empirical Bayes alpha (Antoniak 1974) -------------------

#' Empirical-Bayes alpha MLE for a DP, given the observed K_n.
#' @export
ghosal_empirical_bayes <- function(x, alpha_grid = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2) return(list(estimate = NA_real_, n = n,
                          method = "Empirical Bayes (n<2)"))
  K_n <- length(unique(x))
  if (K_n == n) K_n <- max(2, ceiling(log2(n) + 1))
  neg_ll <- function(a) -(K_n * log(a) + lgamma(a) - lgamma(a + n))
  if (is.null(alpha_grid)) {
    opt <- stats::optimize(neg_ll, interval = c(1e-3, 1e3))
    a_hat <- opt$minimum; ll <- -opt$objective
  } else {
    ll_grid <- -sapply(alpha_grid, neg_ll)
    idx <- which.max(ll_grid); a_hat <- alpha_grid[idx]; ll <- ll_grid[idx]
  }
  list(estimate = a_hat, K_n = K_n, log_lik_at_estimate = ll, n = n,
       method = "Empirical-Bayes alpha for DP (Antoniak 1974 MLE)")
}


# --- ghhbp: hierarchical Bayes -------------------------------------

#' Escobar–West augmentation for alpha | K_n with a Gamma(a, b) hyperprior.
#' @export
ghosal_hierarchical_bayes <- function(x, a_prior = 1.0, b_prior = 1.0,
                                        M = 400, seed = 0) {
  set.seed(seed)
  x <- as.numeric(x); n <- length(x)
  if (n < 2) return(list(estimate = NA_real_, n = n,
                          method = "Hierarchical NP-Bayes (n<2)"))
  K_n <- length(unique(x))
  if (K_n == n) K_n <- max(2, ceiling(log2(n) + 1))
  a <- a_prior; b <- b_prior; alpha <- 1
  draws <- numeric(M)
  for (m in seq_len(M)) {
    eta <- stats::rbeta(1, alpha + 1, n)
    w1 <- a + K_n - 1; w2 <- n * (b - log(eta))
    p_eta <- w1 / (w1 + w2)
    if (stats::runif(1) < p_eta) {
      alpha <- stats::rgamma(1, shape = a + K_n, rate = b - log(eta))
    } else {
      alpha <- stats::rgamma(1, shape = a + K_n - 1, rate = b - log(eta))
    }
    draws[m] <- alpha
  }
  burn <- M %/% 4; chain <- draws[(burn + 1):M]
  list(estimate = mean(chain), alpha_se = sd(chain), alpha_draws = chain,
       K_n = K_n, n = n,
       method = "Escobar-West augmentation for alpha | K_n")
}


# --- ghtst: Polya-tree Bayes factor --------------------------------

#' Polya-tree Bayes factor for H0: F = N(loc, scale^2).
#' @export
ghosal_np_testing <- function(x, ref_loc = 0, ref_scale = 1, depth = 6,
                                c = 1.0) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2) return(list(statistic = NA_real_, p_value = NA_real_, n = n,
                          method = "Polya-tree BF (n<2)"))
  u <- stats::pnorm(x, mean = ref_loc, sd = ref_scale)
  log_bf <- 0
  for (m in seq_len(depth)) {
    nbins <- 2^m
    edges <- seq(0, 1, length.out = nbins + 1)
    bin <- findInterval(u, edges, rightmost.closed = TRUE)
    bin <- pmin(pmax(bin, 1L), nbins)
    counts <- tabulate(bin, nbins = nbins)
    alpha <- c * m * m
    n0 <- counts[seq(1, nbins, by = 2)]; n1 <- counts[seq(2, nbins, by = 2)]
    log_bf <- log_bf + sum(lbeta(alpha + n0, alpha + n1) - lbeta(alpha, alpha))
  }
  BF10 <- exp(log_bf)
  p_value <- if (BF10 > 1) 1 / (1 + BF10) else 0.5
  list(statistic = log_bf, p_value = p_value, BF10 = BF10,
       log_BF10 = log_bf, n = n, depth = depth,
       method = "Polya-tree Bayes-factor test (Berger-Guglielmi)")
}


# --- ghmmt: moment matching ----------------------------------------

#' Posterior mean / variance of G(A) for DP(alpha, G0) and A=(lo, hi].
#' @export
ghosal_moment_matching <- function(x, alpha = 1.0, A_lower = NULL,
                                     A_upper = NULL, base_mean = 0,
                                     base_sd = 1) {
  x <- as.numeric(x); n <- length(x)
  if (is.null(A_lower)) A_lower <- -Inf
  if (is.null(A_upper)) A_upper <- if (n) mean(x) else 0
  G0_A <- max(0, min(1, stats::pnorm(A_upper, base_mean, base_sd)
                         - stats::pnorm(A_lower, base_mean, base_sd)))
  prior_mean <- G0_A
  prior_var <- G0_A * (1 - G0_A) / (alpha + 1)
  n_A <- if (n) sum(x > A_lower & x <= A_upper) else 0L
  post_mean <- (alpha * G0_A + n_A) / (alpha + n)
  post_var  <- post_mean * (1 - post_mean) / (alpha + n + 1)
  list(estimate = post_mean, se = sqrt(max(post_var, 0)),
       prior_mean = prior_mean, prior_var = prior_var,
       n_A = as.integer(n_A), n = n, alpha = alpha,
       method = "DP moment-matching (Ferguson 1973)")
}


# --- ghlgd: log-density via monomial expansion ---------------------

#' Log-spline density estimator (Stone 1990, Ghosal Ch 8).
#' @export
ghosal_log_density <- function(x, K = 5, grid = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5) return(list(estimate = NA_real_, n = n,
                          method = "Log-density (n<5)"))
  m <- mean(x); s <- max(sd(x), 1e-6); z <- (x - m) / s
  if (is.null(grid)) gz <- seq(min(z) - 1, max(z) + 1, length.out = 401)
  else gz <- (grid - m) / s
  basis <- function(u) sapply(seq_len(K), function(k) u^k)
  Bx <- basis(z); Bg <- basis(gz)
  neg_ll <- function(theta) {
    eta_x <- Bx %*% theta; eta_g <- Bg %*% theta
    M <- max(eta_g)
    Z <- M + log(sum(diff(gz) * (head(exp(eta_g - M), -1) +
                                  tail(exp(eta_g - M), -1))) / 2)
    -(sum(eta_x) - n * Z) + 1e-4 * sum(theta^2)
  }
  opt <- stats::optim(rep(0, K), neg_ll, method = "BFGS")
  theta <- opt$par
  eta_g <- Bg %*% theta; M <- max(eta_g)
  logZ <- M + log(sum(diff(gz) * (head(exp(eta_g - M), -1) +
                                   tail(exp(eta_g - M), -1))) / 2)
  log_density <- as.numeric(eta_g - logZ - log(s))
  eta0 <- as.numeric(basis(0) %*% theta)
  est <- eta0 - logZ - log(s)
  list(estimate = est, theta = theta,
       log_lik = -(opt$value - 1e-4 * sum(theta^2)),
       grid = gz * s + m, log_density = log_density, K = K, n = n,
       method = "Log-spline density (Stone 1990)")
}


# --- ghwav: Haar wavelet spike-and-slab ----------------------------

.gh_haar_dwt <- function(y) {
  L <- 1L; while (L < length(y)) L <- 2L * L
  if (L > length(y)) y <- c(y, rep(0, L - length(y)))
  coeffs <- list(); cur <- y
  while (length(cur) > 1L) {
    a <- (cur[seq(1, length(cur), by = 2)] + cur[seq(2, length(cur), by = 2)]) / sqrt(2)
    d <- (cur[seq(1, length(cur), by = 2)] - cur[seq(2, length(cur), by = 2)]) / sqrt(2)
    coeffs[[length(coeffs) + 1L]] <- d
    cur <- a
  }
  coeffs[[length(coeffs) + 1L]] <- cur
  list(coeffs = coeffs, L = L)
}

#' Haar-wavelet spike-and-slab BayesThresh estimator (Abramovich 1998).
#' @export
ghosal_wavelet_prior <- function(x, pi = 0.5, sigma = NULL, noise = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 4) return(list(estimate = if (n) mean(x) else NA_real_,
                          fitted = x, n = n, method = "Wavelet prior (n<4)"))
  dw <- .gh_haar_dwt(x); coeffs <- dw$coeffs; L <- dw$L
  finest <- coeffs[[1]]
  if (is.null(noise)) noise <- max(stats::mad(finest) / 0.6745, 1e-6)
  if (is.null(sigma)) {
    all_d <- unlist(coeffs[-length(coeffs)])
    sigma <- sqrt(max(var(all_d) - noise^2, 1e-6))
  }
  sigma <- max(sigma, 1e-6)
  incl <- c(); new_coeffs <- list()
  for (i in seq_along(coeffs[-length(coeffs)])) {
    d <- coeffs[[i]]
    var_slab <- sigma^2 + noise^2
    log_slab  <- stats::dnorm(d, 0, sqrt(var_slab), log = TRUE)
    log_spike <- stats::dnorm(d, 0, noise, log = TRUE)
    a <- log(pi) + log_slab; b <- log(1 - pi) + log_spike
    mm <- pmax(a, b)
    w <- exp(a - mm) / (exp(a - mm) + exp(b - mm))
    shrink <- sigma^2 / var_slab
    new_coeffs[[i]] <- w * shrink * d
    incl <- c(incl, w)
  }
  new_coeffs[[length(new_coeffs) + 1L]] <- coeffs[[length(coeffs)]]
  # Inverse DWT
  cur <- new_coeffs[[length(new_coeffs)]]
  for (i in (length(new_coeffs) - 1L):1L) {
    d <- new_coeffs[[i]]
    out <- numeric(2 * length(cur))
    out[seq(1, length(out), by = 2)] <- (cur + d) / sqrt(2)
    out[seq(2, length(out), by = 2)] <- (cur - d) / sqrt(2)
    cur <- out
  }
  fitted <- cur[seq_len(n)]
  list(estimate = mean(fitted), fitted = fitted, noise = noise,
       sigma = sigma, inclusion = mean(incl), n = n,
       method = "Haar-wavelet spike-and-slab BayesThresh")
}


# CANONICAL TEST
# set.seed(0); x <- rnorm(50)
# stopifnot(0 < ghosal_dirichlet_posterior(x, alpha = 2)$estimate)
# stopifnot(0 < ghosal_stick_breaking_trunc(x, K = 200, seed = 0)$estimate)
# stopifnot(abs(ghosal_contraction_rate(1:100, 1, 1)$estimate - 100^(-1/3)) < 1e-9)
