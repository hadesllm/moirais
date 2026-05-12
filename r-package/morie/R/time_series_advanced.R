# SPDX-License-Identifier: GPL-2.0-only

#' Time-Series Advanced suite (R parity)
#'
#' R parity of \code{morie.fn.\{garch,egrch,tgrch,archm,ewtma,dccmd,coitg,
#' johsn,vecmf,ssmod,kalmn,ucmod,wavts,specf,cohrc,regms,tarmd,midas,propc,
#' nbeat\}}.  Wraps battle-tested CRAN packages (\code{rugarch},
#' \code{rmgarch}, \code{urca}, \code{vars}, \code{dlm}, \code{MSwM},
#' \code{tsDyn}, \code{midasr}, \code{forecast}, \code{wavelets},
#' \code{stats}) when installed and falls back to base-R implementations
#' otherwise.  Each callable returns a named \code{list} whose keys match
#' the Python \code{RichResult} payload (1e-5 agreement for MLE-fit
#' ARMA/VAR; GARCH may differ by ~1\% across packages).
#'
#' @references
#' Hyndman RJ, Athanasopoulos G (2021). \emph{Forecasting: Principles and
#'   Practice} (3rd ed.). OTexts.
#' Tsay RS (2010). \emph{Analysis of Financial Time Series} (3rd ed.).
#'   Wiley.
#'
#' @name time_series_advanced
#' @importFrom stats var sd lm coef residuals fitted lsfit fft
#'   acf arima ar nlminb pnorm dnorm cor decompose ts filter quantile
NULL


# Null-coalescing helper used internally by Johansen fallback critical
# values; not exported.
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a


# ── garch ────────────────────────────────────────────────────────────────

#' Fit a GARCH(1,1) model to a return series
#'
#' \deqn{\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2.}
#'
#' @param x Numeric return series.
#' @return Named list with \code{omega, alpha, beta, persistence, loglik,
#'   conditional_variance, n, method}.
#' @export
garch_fit <- function(x) {
  r <- as.numeric(x) - mean(as.numeric(x))
  n <- length(r); if (n < 10) stop("Need >=10 obs.")
  if (requireNamespace("rugarch", quietly = TRUE)) {
    spec <- rugarch::ugarchspec(
      variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
    )
    fit <- rugarch::ugarchfit(spec, r, solver = "hybrid")
    p <- rugarch::coef(fit)
    return(list(omega = unname(p["omega"]), alpha = unname(p["alpha1"]),
                beta = unname(p["beta1"]),
                persistence = unname(p["alpha1"] + p["beta1"]),
                loglik = as.numeric(rugarch::likelihood(fit)),
                conditional_variance = as.numeric(rugarch::sigma(fit))^2,
                n = n,
                method = "GARCH(1,1) via rugarch"))
  }
  neg_ll <- function(p) {
    omega <- p[1]; alpha <- p[2]; beta <- p[3]
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1) return(1e10)
    s2 <- numeric(n); s2[1] <- var(r)
    for (t in 2:n) s2[t] <- max(omega + alpha * r[t - 1]^2 + beta * s2[t - 1], 1e-12)
    0.5 * sum(log(2 * pi * s2) + r^2 / s2)
  }
  var_r <- var(r)
  opt <- nlminb(c(var_r * 0.05, 0.1, 0.85), neg_ll,
                lower = c(1e-8, 1e-8, 1e-8),
                upper = c(var_r * 10, 0.999, 0.999))
  omega <- opt$par[1]; alpha <- opt$par[2]; beta <- opt$par[3]
  s2 <- numeric(n); s2[1] <- var_r
  for (t in 2:n) s2[t] <- omega + alpha * r[t - 1]^2 + beta * s2[t - 1]
  list(omega = omega, alpha = alpha, beta = beta,
       persistence = alpha + beta, loglik = -opt$objective,
       conditional_variance = s2, n = n,
       method = "GARCH(1,1) Gaussian MLE (base R)")
}


# ── egrch ────────────────────────────────────────────────────────────────

#' EGARCH(1,1) asymmetric volatility model
#'
#' @inheritParams garch_fit
#' @return Named list with \code{omega, alpha, gamma, beta, loglik,
#'   conditional_variance, n, method}.
#' @export
egarch_model <- function(x) {
  r <- as.numeric(x) - mean(as.numeric(x)); n <- length(r)
  if (n < 20) stop("Need >=20 obs.")
  if (requireNamespace("rugarch", quietly = TRUE)) {
    spec <- rugarch::ugarchspec(
      variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
    )
    fit <- rugarch::ugarchfit(spec, r, solver = "hybrid")
    p <- rugarch::coef(fit)
    return(list(omega = unname(p["omega"]),
                alpha = unname(p["alpha1"]),
                gamma = unname(p["gamma1"]),
                beta  = unname(p["beta1"]),
                loglik = as.numeric(rugarch::likelihood(fit)),
                conditional_variance = as.numeric(rugarch::sigma(fit))^2,
                n = n,
                method = "EGARCH(1,1) via rugarch"))
  }
  EZ <- sqrt(2 / pi)
  neg_ll <- function(p) {
    omega <- p[1]; alpha <- p[2]; gamma <- p[3]; beta <- p[4]
    if (abs(beta) >= 1) return(1e10)
    log_s2 <- numeric(n); log_s2[1] <- log(var(r) + 1e-12)
    for (t in 2:n) {
      z <- r[t - 1] / sqrt(exp(log_s2[t - 1]) + 1e-12)
      log_s2[t] <- omega + beta * log_s2[t - 1] + alpha * (abs(z) - EZ) + gamma * z
    }
    s2 <- exp(log_s2)
    0.5 * sum(log(2 * pi * s2) + r^2 / s2)
  }
  opt <- nlminb(c(0, 0.1, 0, 0.9), neg_ll,
                lower = c(-5, -1, -1, -0.999),
                upper = c(5, 1, 1, 0.999))
  log_s2 <- numeric(n); log_s2[1] <- log(var(r) + 1e-12)
  for (t in 2:n) {
    z <- r[t - 1] / sqrt(exp(log_s2[t - 1]) + 1e-12)
    log_s2[t] <- opt$par[1] + opt$par[4] * log_s2[t - 1] +
                 opt$par[2] * (abs(z) - EZ) + opt$par[3] * z
  }
  list(omega = opt$par[1], alpha = opt$par[2],
       gamma = opt$par[3], beta = opt$par[4],
       loglik = -opt$objective,
       conditional_variance = exp(log_s2), n = n,
       method = "EGARCH(1,1) Gaussian MLE (base R)")
}


# ── tgrch ────────────────────────────────────────────────────────────────

#' GJR-GARCH(1,1) threshold GARCH
#'
#' @inheritParams garch_fit
#' @return Named list with \code{omega, alpha, gamma, beta, persistence,
#'   loglik, conditional_variance, n, method}.
#' @export
tgarch_model <- function(x) {
  r <- as.numeric(x) - mean(as.numeric(x)); n <- length(r)
  if (n < 20) stop("Need >=20 obs.")
  if (requireNamespace("rugarch", quietly = TRUE)) {
    spec <- rugarch::ugarchspec(
      variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
    )
    fit <- rugarch::ugarchfit(spec, r, solver = "hybrid")
    p <- rugarch::coef(fit)
    return(list(omega = unname(p["omega"]),
                alpha = unname(p["alpha1"]),
                gamma = unname(p["gamma1"]),
                beta  = unname(p["beta1"]),
                persistence = unname(p["alpha1"] + 0.5 * p["gamma1"] + p["beta1"]),
                loglik = as.numeric(rugarch::likelihood(fit)),
                conditional_variance = as.numeric(rugarch::sigma(fit))^2,
                n = n,
                method = "GJR-GARCH(1,1) via rugarch"))
  }
  neg_ll <- function(p) {
    omega <- p[1]; alpha <- p[2]; gamma <- p[3]; beta <- p[4]
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + 0.5 * gamma + beta >= 1)
      return(1e10)
    s2 <- numeric(n); s2[1] <- var(r) + 1e-10
    for (t in 2:n) {
      I <- if (r[t - 1] < 0) 1 else 0
      s2[t] <- max(omega + (alpha + gamma * I) * r[t - 1]^2 + beta * s2[t - 1],
                   1e-12)
    }
    0.5 * sum(log(2 * pi * s2) + r^2 / s2)
  }
  var_r <- var(r)
  opt <- nlminb(c(var_r * 0.05, 0.05, 0.05, 0.85), neg_ll,
                lower = c(1e-8, 1e-8, -0.5, 1e-8),
                upper = c(var_r * 10, 0.5, 0.999, 0.999))
  omega <- opt$par[1]; alpha <- opt$par[2]
  gamma <- opt$par[3]; beta <- opt$par[4]
  s2 <- numeric(n); s2[1] <- var_r
  for (t in 2:n) {
    I <- if (r[t - 1] < 0) 1 else 0
    s2[t] <- omega + (alpha + gamma * I) * r[t - 1]^2 + beta * s2[t - 1]
  }
  list(omega = omega, alpha = alpha, gamma = gamma, beta = beta,
       persistence = alpha + 0.5 * gamma + beta,
       loglik = -opt$objective,
       conditional_variance = s2, n = n,
       method = "GJR-GARCH(1,1) Gaussian MLE (base R)")
}


# ── archm ────────────────────────────────────────────────────────────────

#' ARCH(1)-in-mean model
#'
#' @inheritParams garch_fit
#' @return Named list with \code{mu, delta, omega, alpha, loglik,
#'   conditional_variance, n, method}.
#' @export
arch_in_mean <- function(x) {
  y <- as.numeric(x); n <- length(y)
  if (n < 20) stop("Need >=20 obs.")
  neg_ll <- function(p) {
    mu <- p[1]; delta <- p[2]; omega <- p[3]; alpha <- p[4]
    if (omega <= 0 || alpha < 0 || alpha >= 0.999) return(1e10)
    s2 <- numeric(n); s2[1] <- max(var(y), 1e-10)
    eps <- numeric(n); eps[1] <- y[1] - mu - delta * sqrt(s2[1])
    for (t in 2:n) {
      s2[t] <- max(omega + alpha * eps[t - 1]^2, 1e-12)
      eps[t] <- y[t] - mu - delta * sqrt(s2[t])
    }
    0.5 * sum(log(2 * pi * s2) + eps^2 / s2)
  }
  var_y <- var(y)
  opt <- nlminb(c(mean(y), 0, var_y * 0.5, 0.2), neg_ll,
                lower = c(-10, -10, 1e-8, 1e-8),
                upper = c(10, 10, var_y * 10, 0.999))
  mu <- opt$par[1]; delta <- opt$par[2]
  omega <- opt$par[3]; alpha <- opt$par[4]
  s2 <- numeric(n); s2[1] <- var_y
  eps <- numeric(n); eps[1] <- y[1] - mu - delta * sqrt(s2[1])
  for (t in 2:n) {
    s2[t] <- omega + alpha * eps[t - 1]^2
    eps[t] <- y[t] - mu - delta * sqrt(s2[t])
  }
  list(mu = mu, delta = delta, omega = omega, alpha = alpha,
       loglik = -opt$objective,
       conditional_variance = s2, n = n,
       method = "ARCH(1)-in-mean Gaussian MLE (base R)")
}


# ── ewtma ────────────────────────────────────────────────────────────────

#' EWMA volatility (RiskMetrics 1996)
#'
#' @inheritParams garch_fit
#' @param lambda Decay factor in (0,1). Default 0.94 (daily RiskMetrics).
#' @return Named list with \code{conditional_variance, conditional_volatility,
#'   lambda, n, last_variance, last_volatility, method}.
#' @export
ewma_volatility <- function(x, lambda = 0.94) {
  r <- as.numeric(x); n <- length(r)
  if (n < 2) stop("Need >=2 obs.")
  if (lambda <= 0 || lambda >= 1) stop("lambda must be in (0,1).")
  r2 <- r^2
  s2 <- numeric(n); s2[1] <- r2[1]
  for (t in 2:n) s2[t] <- lambda * s2[t - 1] + (1 - lambda) * r2[t - 1]
  list(conditional_variance = s2,
       conditional_volatility = sqrt(s2),
       lambda = lambda, n = n,
       last_variance = s2[n],
       last_volatility = sqrt(s2[n]),
       method = "EWMA RiskMetrics")
}


# ── dccmd ────────────────────────────────────────────────────────────────

#' DCC multivariate GARCH (Engle 2002)
#'
#' Two-step DCC(1,1) on a panel of return series.
#'
#' @param x Numeric matrix of returns (T x k).
#' @return Named list with \code{a, b, unconditional_correlation,
#'   conditional_correlation, conditional_variance, loglik, n, k, method}.
#' @export
dcc_multivariate_garch <- function(x) {
  X <- as.matrix(x); if (nrow(X) < ncol(X)) X <- t(X)
  n <- nrow(X); k <- ncol(X)
  if (n < 30 || k < 2) stop("Need n>=30, k>=2.")
  if (requireNamespace("rmgarch", quietly = TRUE) &&
      requireNamespace("rugarch", quietly = TRUE)) {
    uspec <- rugarch::multispec(replicate(k, rugarch::ugarchspec(
      variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)),
      simplify = FALSE))
    dccspec <- rmgarch::dccspec(uspec = uspec, dccOrder = c(1, 1),
                                distribution = "mvnorm")
    fit <- rmgarch::dccfit(dccspec, data = X)
    p <- rmgarch::coef(fit)
    sig_mat <- as.matrix(rmgarch::sigma(fit))
    return(list(a = unname(p["[Joint]dcca1"]),
                b = unname(p["[Joint]dccb1"]),
                unconditional_correlation = cor(X),
                conditional_correlation = rmgarch::rcor(fit),
                conditional_variance = sig_mat^2,
                loglik = as.numeric(rmgarch::likelihood(fit)),
                n = n, k = k,
                method = "DCC(1,1) via rmgarch"))
  }
  # Fallback: two-step EWMA-marginal + closed-form DCC update.
  H <- matrix(NA_real_, n, k); Z <- matrix(NA_real_, n, k)
  for (j in seq_len(k)) {
    rj <- X[, j] - mean(X[, j])
    g <- garch_fit(rj)
    H[, j] <- g$conditional_variance
    Z[, j] <- rj / sqrt(H[, j] + 1e-12)
  }
  Q_bar <- crossprod(Z) / n
  neg_ll <- function(p) {
    a <- p[1]; b <- p[2]
    if (a < 0 || b < 0 || a + b >= 0.9999) return(1e10)
    Q <- Q_bar; ll <- 0
    for (t in seq_len(n)) {
      d <- sqrt(pmax(diag(Q), 1e-12))
      R <- Q / outer(d, d)
      ld <- determinant(R, logarithm = TRUE)
      if (ld$sign <= 0) return(1e10)
      Rinv <- solve(R)
      zt <- Z[t, ]
      ll <- ll + 0.5 * (ld$modulus + sum(zt * (Rinv %*% zt)) - sum(zt^2))
      Q <- (1 - a - b) * Q_bar + a * tcrossprod(zt) + b * Q
    }
    as.numeric(ll)
  }
  opt <- nlminb(c(0.02, 0.95), neg_ll,
                lower = c(1e-6, 1e-6),
                upper = c(0.5, 0.999))
  a <- opt$par[1]; b <- opt$par[2]
  Q <- Q_bar
  R_path <- array(NA_real_, c(n, k, k))
  for (t in seq_len(n)) {
    d <- sqrt(pmax(diag(Q), 1e-12))
    R_path[t, , ] <- Q / outer(d, d)
    Q <- (1 - a - b) * Q_bar + a * tcrossprod(Z[t, ]) + b * Q
  }
  list(a = a, b = b,
       unconditional_correlation = Q_bar,
       conditional_correlation = R_path,
       conditional_variance = H,
       loglik = -opt$objective,
       n = n, k = k,
       method = "DCC(1,1) two-step Gaussian MLE (base R)")
}


# ── coitg ────────────────────────────────────────────────────────────────

#' Engle-Granger two-step cointegration test
#'
#' @param y1 Numeric, first series.
#' @param y2 Numeric, second series.
#' @param max_lag Max ADF augmentation lags. Default \code{floor(12*(n/100)^{1/4})}.
#' @return Named list with \code{adf_statistic, p_value, beta, n, method}.
#' @export
eg_coint <- function(y1, y2, max_lag = NULL) {
  y1 <- as.numeric(y1); y2 <- as.numeric(y2)
  if (length(y1) != length(y2)) stop("Length mismatch.")
  n <- length(y1); if (n < 20) stop("Need >=20 obs.")
  if (is.null(max_lag)) max_lag <- floor(12 * (n / 100)^0.25)
  fit_ls <- lm(y1 ~ y2)
  beta <- coef(fit_ls)
  resid <- residuals(fit_ls)
  if (requireNamespace("urca", quietly = TRUE)) {
    adf <- urca::ur.df(resid, type = "none", lags = max_lag,
                       selectlags = "AIC")
    stat <- as.numeric(adf@teststat[1])
  } else {
    # Plain ADF-style t-stat on residuals.
    dr <- diff(resid); T <- length(dr) - max_lag
    dep <- dr[(max_lag + 1):length(dr)]
    Xr <- resid[(max_lag + 1):length(resid) - 1]
    Xr <- cbind(Xr)
    if (max_lag >= 1) {
      for (i in seq_len(max_lag))
        Xr <- cbind(Xr, dr[(max_lag + 1 - i):(length(dr) - i)])
    }
    b <- lsfit(Xr, dep, intercept = FALSE)
    e <- dep - Xr %*% b$coef
    sig2 <- sum(e^2) / (T - ncol(Xr))
    se <- sqrt(sig2 * solve(crossprod(Xr))[1, 1])
    stat <- b$coef[1] / se
  }
  crit <- c(`1%` = -3.90, `5%` = -3.34, `10%` = -3.04)
  approx_p <- if (stat < crit["1%"]) 0.005 else
              if (stat < crit["5%"]) 0.03 else
              if (stat < crit["10%"]) 0.07 else
              min(1, 2 * pnorm(stat))
  list(adf_statistic = as.numeric(stat),
       p_value = as.numeric(approx_p),
       beta = unname(beta),
       critical_values = crit,
       n = n,
       method = "Engle-Granger 2-step cointegration (Engle & Granger 1987)")
}


# ── johsn ────────────────────────────────────────────────────────────────

#' Johansen trace test for cointegration
#'
#' @param x Numeric matrix (T x k) of I(1) candidate series.
#' @param k_ar_diff Number of lagged differences. Default 1.
#' @return Named list with \code{eigenvalues, trace_stat, crit_values,
#'   rank, n, k, method}.
#' @export
johansen_cointegration <- function(x, k_ar_diff = 1) {
  Y <- as.matrix(x); if (nrow(Y) < ncol(Y)) Y <- t(Y)
  Tt <- nrow(Y); k <- ncol(Y)
  if (Tt < 20 || k < 2) stop("Need T>=20, k>=2.")
  if (requireNamespace("urca", quietly = TRUE)) {
    jres <- urca::ca.jo(Y, type = "trace", ecdet = "none",
                        K = max(k_ar_diff + 1, 2))
    return(list(eigenvalues = jres@lambda,
                trace_stat = jres@teststat,
                crit_values = jres@cval,
                rank = sum(jres@teststat > jres@cval[, "5pct"]),
                n = Tt, k = k,
                method = "Johansen trace test via urca::ca.jo"))
  }
  dY <- diff(Y)
  rows <- nrow(dY) - k_ar_diff
  Z0 <- dY[(k_ar_diff + 1):nrow(dY), , drop = FALSE]
  Z1 <- Y[(k_ar_diff + 1):(k_ar_diff + rows), , drop = FALSE]
  Z2 <- matrix(1, rows, 1)
  if (k_ar_diff > 0) {
    for (i in seq_len(k_ar_diff))
      Z2 <- cbind(Z2, dY[(k_ar_diff - i + 1):(k_ar_diff - i + rows), ,
                          drop = FALSE])
  }
  P <- Z2 %*% solve(crossprod(Z2)) %*% t(Z2)
  R0 <- Z0 - P %*% Z0; R1 <- Z1 - P %*% Z1
  S00 <- crossprod(R0) / rows; S01 <- crossprod(R0, R1) / rows
  S11 <- crossprod(R1) / rows
  M <- solve(S11) %*% t(S01) %*% solve(S00) %*% S01
  eig <- sort(Re(eigen(M, only.values = TRUE)$values), decreasing = TRUE)
  eig <- pmax(pmin(eig, 1 - 1e-12), 0)
  trace_stat <- sapply(0:(k - 1), function(r) -rows * sum(log(1 - eig[(r + 1):k])))
  crit_table <- list(
    `1` = c(2.7055, 3.8415, 6.6349), `2` = c(13.4294, 15.4943, 19.9349),
    `3` = c(27.0669, 29.7961, 35.4628), `4` = c(44.4929, 47.8545, 54.6815),
    `5` = c(65.8202, 69.8189, 77.8202)
  )
  crit_values <- do.call(rbind, lapply(seq_len(k),
    function(r) crit_table[[as.character(k - r + 1)]] %||% c(NA, NA, NA)))
  rank <- sum(trace_stat > crit_values[, 2])
  list(eigenvalues = eig, trace_stat = trace_stat,
       crit_values = crit_values, rank = rank,
       n = Tt, k = k,
       method = "Johansen trace test (reduced-rank regression, base R)")
}


# ── vecmf ────────────────────────────────────────────────────────────────

#' Vector error-correction model (VECM)
#'
#' @param Y Numeric matrix (T x k) of I(1) candidate series.
#' @param k_ar Number of lagged differences. Default 1.
#' @param coint_rank Cointegration rank. Default 1.
#' @return Named list with \code{alpha, beta, Gamma, Sigma, loglik, n, k,
#'   rank, method}.
#' @export
vecm <- function(Y, k_ar = 1, coint_rank = 1) {
  Y <- as.matrix(Y); if (nrow(Y) < ncol(Y)) Y <- t(Y)
  Tt <- nrow(Y); k <- ncol(Y)
  if (Tt < 20 || k < 2 || coint_rank < 1 || coint_rank > k)
    stop("Need T>=20, 1<=rank<=k.")
  if (requireNamespace("urca", quietly = TRUE) &&
      requireNamespace("vars", quietly = TRUE)) {
    jres <- urca::ca.jo(Y, type = "trace", ecdet = "none",
                        K = max(k_ar + 1, 2))
    vfit <- vars::vec2var(jres, r = coint_rank)
    return(list(alpha = jres@V[, seq_len(coint_rank), drop = FALSE],
                beta  = jres@V[, seq_len(coint_rank), drop = FALSE],
                Gamma = vfit$A,
                Sigma = summary(jres)$summary,
                loglik = NA_real_,
                n = Tt, k = k, rank = coint_rank,
                method = "VECM via urca::ca.jo + vars::vec2var"))
  }
  dY <- diff(Y); rows <- nrow(dY) - k_ar
  Z0 <- dY[(k_ar + 1):nrow(dY), , drop = FALSE]
  Z1 <- Y[(k_ar + 1):(k_ar + rows), , drop = FALSE]
  Z2 <- if (k_ar == 0) matrix(0, rows, 0)
        else do.call(cbind,
          lapply(seq_len(k_ar), function(i) dY[(k_ar - i + 1):(k_ar - i + rows), ]))
  X <- cbind(Z1, Z2)
  B <- solve(crossprod(X), crossprod(X, Z0))
  Pi_hat <- t(B[seq_len(k), , drop = FALSE])
  sv <- svd(t(Pi_hat))
  alpha <- sv$u[, seq_len(coint_rank), drop = FALSE] *
           rep(sv$d[seq_len(coint_rank)], each = nrow(sv$u))
  beta <- sv$v[, seq_len(coint_rank), drop = FALSE]
  eps <- Z0 - X %*% B
  Sigma <- crossprod(eps) / max(rows - 1, 1)
  list(alpha = alpha, beta = beta, Gamma = list(), Sigma = Sigma,
       loglik = NA_real_, n = Tt, k = k, rank = coint_rank,
       method = "VECM via SVD of OLS Pi (base R)")
}


# ── ssmod ────────────────────────────────────────────────────────────────

#' Local-level state-space model (Kalman filter+smoother)
#'
#' @param x Numeric univariate series.
#' @return Named list with \code{filtered_state, filtered_state_variance,
#'   smoothed_state, loglik, Q, R, n, method}.
#' @export
state_space_model <- function(x) {
  y <- as.numeric(x); n <- length(y)
  if (n < 4) stop("Need >=4 obs.")
  if (requireNamespace("dlm", quietly = TRUE)) {
    build <- function(p) dlm::dlmModPoly(order = 1,
                                         dV = exp(p[1]), dW = exp(p[2]))
    fit <- dlm::dlmMLE(y, parm = c(log(var(diff(y)) / 2),
                                   log(var(diff(y)) / 2)),
                       build = build)
    mod <- build(fit$par)
    f <- dlm::dlmFilter(y, mod)
    s <- dlm::dlmSmooth(f)
    return(list(filtered_state = as.numeric(f$m)[-1],
                filtered_state_variance = sapply(dlm::dlmSvd2var(f$U.C, f$D.C),
                                                  function(x) x[1, 1])[-1],
                smoothed_state = as.numeric(s$s)[-1],
                loglik = -fit$value,
                Q = exp(fit$par[2]),
                R = exp(fit$par[1]),
                n = n,
                method = "Local-level Kalman via dlm"))
  }
  Q <- var(diff(y)) / 2; R <- var(diff(y)) / 2
  a <- numeric(n); P <- numeric(n)
  a[1] <- y[1]; P[1] <- 1e7
  ll <- 0
  for (t in 2:n) {
    Pp <- P[t - 1] + Q
    v <- y[t] - a[t - 1]
    Fv <- Pp + R; K <- Pp / Fv
    a[t] <- a[t - 1] + K * v
    P[t] <- Pp - K * Pp
    ll <- ll + -0.5 * (log(2 * pi * Fv) + v^2 / Fv)
  }
  a_s <- a; P_s <- P
  for (t in (n - 1):1) {
    Pp <- P[t] + Q; J <- P[t] / Pp
    a_s[t] <- a[t] + J * (a_s[t + 1] - a[t])
    P_s[t] <- P[t] + J^2 * (P_s[t + 1] - Pp)
  }
  list(filtered_state = a, filtered_state_variance = P,
       smoothed_state = a_s, loglik = ll, Q = Q, R = R, n = n,
       method = "Local-level Kalman filter+smoother (base R)")
}


# ── kalmn ────────────────────────────────────────────────────────────────

#' Kalman filter predict-update for a linear-Gaussian state-space model
#'
#' Defaults to a univariate local-level model when matrices are omitted.
#'
#' @param x Numeric vector or matrix of observations.
#' @param F Transition matrix (default identity).
#' @param H Observation matrix (default identity).
#' @param Q State-innovation covariance (default sigma^2 I).
#' @param R Observation covariance (default sigma^2 I).
#' @param x0 Initial state mean.
#' @param P0 Initial state covariance.
#' @return Named list with \code{state, state_cov, innovations,
#'   innovation_variance, loglik, n, method}.
#' @export
kalman_filter <- function(x, F = NULL, H = NULL, Q = NULL, R = NULL,
                          x0 = NULL, P0 = NULL) {
  Y <- as.matrix(x); n <- nrow(Y); m <- ncol(Y)
  if (n < 2) stop("Need >=2 obs.")
  if (is.null(F)) F <- diag(m)
  if (is.null(H)) H <- diag(m)
  v0 <- var(diff(Y)) * 0.5
  if (is.null(Q)) Q <- if (is.matrix(v0)) v0 else diag(as.numeric(v0), m)
  if (is.null(R)) R <- if (is.matrix(v0)) v0 else diag(as.numeric(v0), m)
  F <- as.matrix(F); H <- as.matrix(H); Q <- as.matrix(Q); R <- as.matrix(R)
  p <- nrow(F)
  if (is.null(x0)) { x0 <- numeric(p); x0[seq_len(min(p, m))] <- Y[1, seq_len(min(p, m))] }
  if (is.null(P0)) P0 <- diag(1e6, p)
  x_hat <- matrix(0, n, p)
  P_arr <- array(0, c(n, p, p))
  innov <- matrix(0, n, m)
  Sv <- array(0, c(n, m, m))
  xc <- as.numeric(x0); Pc <- P0; ll <- 0
  for (t in seq_len(n)) {
    xp <- F %*% xc
    Pp <- F %*% Pc %*% t(F) + Q
    v <- Y[t, ] - H %*% xp
    S <- H %*% Pp %*% t(H) + R
    Sinv <- tryCatch(solve(S), error = function(e) {
      if (requireNamespace("MASS", quietly = TRUE)) MASS::ginv(S)
      else solve(S + diag(1e-8, nrow(S)))
    })
    K <- Pp %*% t(H) %*% Sinv
    xc <- as.numeric(xp + K %*% v)
    Pc <- (diag(p) - K %*% H) %*% Pp
    x_hat[t, ] <- xc
    P_arr[t, , ] <- Pc
    innov[t, ] <- as.numeric(v)
    Sv[t, , ] <- S
    ld <- determinant(S, logarithm = TRUE)
    if (ld$sign > 0)
      ll <- ll + -0.5 * (m * log(2 * pi) + ld$modulus +
                         sum(v * (Sinv %*% v)))
  }
  list(state = x_hat, state_cov = P_arr,
       innovations = innov, innovation_variance = Sv,
       loglik = as.numeric(ll), n = n,
       method = "Linear Gaussian Kalman filter (base R)")
}


# ── ucmod ────────────────────────────────────────────────────────────────

#' Unobserved-components decomposition (trend + seasonal + irregular)
#'
#' @param x Numeric univariate series.
#' @param period Seasonal period (pass 0 to omit). Default 12.
#' @param trend Trend specification, "local level" or "local linear".
#' @return Named list with \code{trend, seasonal, irregular, loglik, n,
#'   period, method}.
#' @export
unobserved_components <- function(x, period = 12, trend = "local linear") {
  y <- as.numeric(x); n <- length(y)
  if (n < max(2 * period, 6)) stop("Series too short.")
  if (period > 1) {
    dec <- stats::decompose(stats::ts(y, frequency = period),
                            type = "additive")
    mu <- as.numeric(dec$trend)
    mu[is.na(mu)] <- mean(mu, na.rm = TRUE)
    season <- as.numeric(dec$seasonal)
    irr <- y - mu - season
  } else {
    mu <- stats::filter(y, rep(1 / 5, 5), sides = 2)
    mu <- as.numeric(mu); mu[is.na(mu)] <- mean(mu, na.rm = TRUE)
    season <- numeric(n); irr <- y - mu
  }
  list(trend = mu, seasonal = season, irregular = irr,
       loglik = NA_real_, n = n, period = period,
       method = "Additive trend+seasonal decomposition (base R)")
}


# ── wavts ────────────────────────────────────────────────────────────────

#' Discrete wavelet decomposition for a time series
#'
#' @param x Numeric univariate series.
#' @param wavelet Wavelet family. Default "haar".
#' @param level Decomposition depth. Default floor(log2 n) capped at 6.
#' @return Named list with \code{approximation, details, energies, level,
#'   n, wavelet, method}.
#' @export
wavelet_time_series <- function(x, wavelet = "haar", level = NULL) {
  y <- as.numeric(x); n <- length(y)
  if (n < 4) stop("Need >=4 obs.")
  max_lv <- floor(log2(n))
  if (is.null(level)) level <- min(max(max_lv, 1), 6)
  level <- min(level, max_lv)
  if (requireNamespace("wavelets", quietly = TRUE)) {
    fit <- wavelets::dwt(y, filter = wavelet, n.levels = level)
    cA <- as.numeric(fit@V[[level]])
    cDs <- lapply(rev(fit@W), as.numeric)
    energies <- c(sum(cA^2), sapply(cDs, function(c) sum(c^2)))
    return(list(approximation = cA,
                details = cDs,
                energies = energies,
                level = level, n = n, wavelet = wavelet,
                method = sprintf("DWT via wavelets (wavelet=%s, level=%d)",
                                 wavelet, level)))
  }
  cA <- y; cDs <- list()
  for (lv in seq_len(level)) {
    if (length(cA) < 2) break
    if (length(cA) %% 2 == 1) cA <- c(cA, cA[length(cA)])
    even <- cA[seq(1, length(cA), 2)]
    odd <- cA[seq(2, length(cA), 2)]
    cD <- (even - odd) / sqrt(2)
    cA <- (even + odd) / sqrt(2)
    cDs <- c(list(cD), cDs)
  }
  energies <- c(sum(cA^2), sapply(cDs, function(c) sum(c^2)))
  list(approximation = cA, details = cDs, energies = energies,
       level = level, n = n, wavelet = "haar",
       method = "Haar DWT (base R fallback)")
}


# ── specf ────────────────────────────────────────────────────────────────

#' Welch power spectral density
#'
#' @param x Numeric univariate series.
#' @param fs Sampling frequency. Default 1.
#' @param nperseg Segment length. Default max(n/4, 8).
#' @return Named list with \code{frequencies, psd, n_segments, nperseg,
#'   fs, n, method}.
#' @export
spectral_density <- function(x, fs = 1, nperseg = NULL) {
  r <- as.numeric(x); n <- length(r)
  if (n < 8) stop("Need >=8 obs.")
  if (is.null(nperseg)) nperseg <- max(n %/% 4, 8)
  nperseg <- min(nperseg, n)
  step <- max(nperseg %/% 2, 1)
  win <- 0.5 - 0.5 * cos(2 * pi * (0:(nperseg - 1)) / max(nperseg - 1, 1))
  U <- sum(win^2)
  nfreq <- nperseg %/% 2 + 1
  S <- numeric(nfreq); nseg <- 0; start <- 1
  while (start + nperseg - 1 <= n) {
    seg <- (r[start:(start + nperseg - 1)] -
            mean(r[start:(start + nperseg - 1)])) * win
    Fk <- fft(seg)[1:nfreq]
    S <- S + Mod(Fk)^2
    nseg <- nseg + 1
    start <- start + step
  }
  S <- S / (nseg * U * fs)
  freqs <- seq(0, fs / 2, length.out = nfreq)
  list(frequencies = freqs, psd = S, n_segments = nseg,
       nperseg = nperseg, fs = fs, n = n,
       method = "Welch PSD (Hann window, 50% overlap, base R)")
}


# ── cohrc ────────────────────────────────────────────────────────────────

#' Magnitude-squared coherence between two time series
#'
#' @param x Numeric vector.
#' @param y Numeric vector (same length).
#' @param nperseg Segment length. Default n/4.
#' @param fs Sampling frequency. Default 1.
#' @return Named list with \code{frequencies, coherence, n_segments,
#'   nperseg, fs, n, method}.
#' @export
coherence <- function(x, y, nperseg = NULL, fs = 1) {
  x <- as.numeric(x); y <- as.numeric(y)
  if (length(x) != length(y)) stop("Length mismatch.")
  n <- length(x); if (n < 8) stop("Need >=8 obs.")
  if (is.null(nperseg)) nperseg <- max(n %/% 4, 4)
  nperseg <- min(nperseg, n)
  step <- nperseg %/% 2
  nfreq <- nperseg %/% 2 + 1
  Sxx <- numeric(nfreq); Syy <- numeric(nfreq)
  Sxy <- complex(nfreq); nseg <- 0; start <- 1
  while (start + nperseg - 1 <= n) {
    xs <- x[start:(start + nperseg - 1)] - mean(x[start:(start + nperseg - 1)])
    ys <- y[start:(start + nperseg - 1)] - mean(y[start:(start + nperseg - 1)])
    fx <- fft(xs)[1:nfreq]; fy <- fft(ys)[1:nfreq]
    Sxx <- Sxx + Mod(fx)^2
    Syy <- Syy + Mod(fy)^2
    Sxy <- Sxy + fx * Conj(fy)
    nseg <- nseg + 1
    start <- start + step
  }
  Sxx <- Sxx / nseg; Syy <- Syy / nseg; Sxy <- Sxy / nseg
  denom <- pmax(Sxx * Syy, 1e-15)
  coh <- Mod(Sxy)^2 / denom
  freqs <- seq(0, fs / 2, length.out = nfreq)
  list(frequencies = freqs, coherence = coh,
       n_segments = nseg, nperseg = nperseg,
       fs = fs, n = n,
       method = "Magnitude-squared coherence (Welch, base R)")
}


# ── regms ────────────────────────────────────────────────────────────────

#' Markov-switching regression (Hamilton 1989)
#'
#' Fit a constant-mean, switching-variance K-regime Markov-switching
#' model by EM (Hamilton filter).
#'
#' @param x Numeric univariate series.
#' @param k_regimes Number of latent regimes. Default 2.
#' @return Named list with \code{mu, sigma, transition,
#'   smoothed_probabilities, loglik, n, k_regimes, method}.
#' @export
regime_switching <- function(x, k_regimes = 2) {
  y <- as.numeric(x); n <- length(y)
  if (n < 4 * k_regimes) stop("Series too short for K regimes.")
  if (requireNamespace("MSwM", quietly = TRUE)) {
    df <- data.frame(y = y)
    base_fit <- lm(y ~ 1, data = df)
    msfit <- MSwM::msmFit(base_fit, k = k_regimes, sw = c(TRUE, TRUE))
    return(list(mu = as.numeric(msfit@Coef[, 1]),
                sigma = as.numeric(msfit@std),
                transition = msfit@transMat,
                smoothed_probabilities = msfit@Fit@smoProb,
                loglik = msfit@Fit@logLikel,
                n = n, k_regimes = k_regimes,
                method = sprintf("MSwM (K=%d)", k_regimes)))
  }
  mu <- seq(min(y), max(y), length.out = k_regimes)
  sig <- rep(max(sd(y), 1e-6), k_regimes)
  P <- matrix(1 / k_regimes, k_regimes, k_regimes)
  pi <- rep(1 / k_regimes, k_regimes)
  ll_prev <- -Inf
  for (it in seq_len(200)) {
    emit <- t(sapply(y,
      function(yt) dnorm(yt, mean = mu, sd = sig)))
    emit <- pmax(emit, 1e-300)
    alpha <- matrix(0, n, k_regimes); cv <- numeric(n)
    alpha[1, ] <- pi * emit[1, ]; cv[1] <- sum(alpha[1, ])
    alpha[1, ] <- alpha[1, ] / cv[1]
    for (t in 2:n) {
      alpha[t, ] <- (alpha[t - 1, ] %*% P) * emit[t, ]
      cv[t] <- sum(alpha[t, ]); alpha[t, ] <- alpha[t, ] / max(cv[t], 1e-300)
    }
    beta <- matrix(0, n, k_regimes); beta[n, ] <- 1
    for (t in (n - 1):1) {
      beta[t, ] <- P %*% (emit[t + 1, ] * beta[t + 1, ])
      beta[t, ] <- beta[t, ] / max(sum(beta[t, ]), 1e-300)
    }
    gamma <- alpha * beta
    gamma <- gamma / rowSums(gamma)
    xi <- array(0, c(n - 1, k_regimes, k_regimes))
    for (t in seq_len(n - 1)) {
      xi[t, , ] <- (alpha[t, ] %*% t(beta[t + 1, ] * emit[t + 1, ])) * P
      xi[t, , ] <- xi[t, , ] / max(sum(xi[t, , ]), 1e-300)
    }
    pi <- gamma[1, ]
    P <- apply(xi, c(2, 3), sum) /
      pmax(colSums(gamma[seq_len(n - 1), , drop = FALSE]), 1e-12)
    for (k in seq_len(k_regimes)) {
      wk <- gamma[, k]
      mu[k] <- sum(wk * y) / max(sum(wk), 1e-12)
      sig[k] <- max(sqrt(sum(wk * (y - mu[k])^2) / max(sum(wk), 1e-12)), 1e-6)
    }
    ll <- sum(log(pmax(cv, 1e-300)))
    if (abs(ll - ll_prev) < 1e-6) break
    ll_prev <- ll
  }
  list(mu = mu, sigma = sig, transition = P,
       smoothed_probabilities = gamma,
       loglik = ll_prev, n = n, k_regimes = k_regimes,
       method = sprintf("Markov switching via EM/Hamilton filter (K=%d, base R)",
                        k_regimes))
}


# ── tarmd ────────────────────────────────────────────────────────────────

#' Two-regime self-exciting threshold autoregressive (SETAR) model
#'
#' @param x Numeric univariate series.
#' @param p AR order in each regime. Default 1.
#' @param d Delay parameter for the threshold variable. Default 1.
#' @param n_grid Grid size for threshold search. Default 50.
#' @return Named list with \code{threshold, phi_lower, phi_upper, p, d,
#'   regime_sizes, sse, n, method}.
#' @export
threshold_autoregression <- function(x, p = 1, d = 1, n_grid = 50) {
  y <- as.numeric(x); n <- length(y); start <- max(p, d)
  if (n - start < 4 * (p + 1))
    stop("Series too short for SETAR(p, d).")
  Y <- y[(start + 1):n]
  X <- cbind(1, do.call(cbind,
    lapply(seq_len(p), function(i) y[(start - i + 1):(n - i)])))
  Z <- y[(start - d + 1):(n - d)]
  ql <- as.numeric(quantile(Z, 0.15))
  qh <- as.numeric(quantile(Z, 0.85))
  grid <- seq(ql, qh, length.out = n_grid)
  best <- list(sse = Inf, threshold = NA, phi_lo = NULL, phi_hi = NULL,
               sizes = NULL)
  for (c in grid) {
    lo <- Z <= c; hi <- !lo
    if (sum(lo) < 2 * (p + 1) || sum(hi) < 2 * (p + 1)) next
    phi_lo <- lsfit(X[lo, , drop = FALSE], Y[lo], intercept = FALSE)$coef
    phi_hi <- lsfit(X[hi, , drop = FALSE], Y[hi], intercept = FALSE)$coef
    sse <- sum((Y[lo] - X[lo, , drop = FALSE] %*% phi_lo)^2) +
           sum((Y[hi] - X[hi, , drop = FALSE] %*% phi_hi)^2)
    if (sse < best$sse) {
      best <- list(sse = sse, threshold = c,
                   phi_lo = phi_lo, phi_hi = phi_hi,
                   sizes = c(lower = sum(lo), upper = sum(hi)))
    }
  }
  if (is.null(best$phi_lo))
    stop("Could not find admissible threshold grid point.")
  list(threshold = best$threshold,
       phi_lower = best$phi_lo, phi_upper = best$phi_hi,
       p = p, d = d, regime_sizes = best$sizes,
       sse = best$sse, n = n,
       method = sprintf("SETAR(p=%d, d=%d) via grid-search OLS", p, d))
}


# ── midas ────────────────────────────────────────────────────────────────

.morie_beta_weights <- function(t1, t2, K) {
  k <- seq_len(K) / (K + 1)
  w <- (k^(t1 - 1)) * ((1 - k)^(t2 - 1))
  if (sum(w) > 0) w / sum(w) else rep(1 / K, K)
}

#' MIDAS regression with Beta-polynomial weights
#'
#' @param x High-frequency regressor matrix (n_t x K) or flat vector.
#' @param y Low-frequency target (length n_t).
#' @param K Number of high-frequency lags (required when x is flat).
#' @return Named list with \code{beta0, beta1, theta1, theta2, weights,
#'   r2, n, K, method}.
#' @export
midas_regression <- function(x, y, K = NULL) {
  Y <- as.numeric(y); nT <- length(Y)
  if (is.null(dim(x))) {
    if (is.null(K)) stop("Pass K when x is flat.")
    if (length(x) < K + nT - 1) stop("x too short.")
    Xf <- as.numeric(x)
    rows <- vector("list", nT)
    for (t in seq_len(nT)) {
      end <- length(Xf) - (nT - t)
      rows[[t]] <- rev(Xf[(end - K + 1):end])
    }
    X <- do.call(rbind, rows)
  } else {
    X <- as.matrix(x); K <- ncol(X)
  }
  if (nrow(X) != nT) stop("Dim mismatch.")
  if (nT < 4) stop("Need >=4 obs.")
  neg_ll <- function(p) {
    b0 <- p[1]; b1 <- p[2]; t1 <- p[3]; t2 <- p[4]
    if (t1 <= 0 || t2 <= 0) return(1e10)
    w <- .morie_beta_weights(t1, t2, K)
    yhat <- b0 + b1 * (X %*% w)
    sse <- sum((Y - yhat)^2)
    if (!is.finite(sse)) 1e10 else sse
  }
  opt <- nlminb(c(mean(Y), 1, 1.5, 2), neg_ll,
                lower = c(-1e3, -1e3, 0.1, 0.1),
                upper = c( 1e3,  1e3, 50,  50))
  b0 <- opt$par[1]; b1 <- opt$par[2]
  t1 <- opt$par[3]; t2 <- opt$par[4]
  w <- .morie_beta_weights(t1, t2, K)
  resid <- Y - (b0 + b1 * (X %*% w))
  ss_tot <- sum((Y - mean(Y))^2)
  r2 <- if (ss_tot > 0) 1 - sum(resid^2) / ss_tot else NA_real_
  list(beta0 = b0, beta1 = b1, theta1 = t1, theta2 = t2,
       weights = w, r2 = r2, n = nT, K = K,
       method = "MIDAS Beta-polynomial via nlminb (base R)")
}


# ── propc ────────────────────────────────────────────────────────────────

#' Prophet-style additive decomposition (linear trend + Fourier seasonality)
#'
#' @param x Numeric univariate series.
#' @param period Seasonal period. Default 12.
#' @return Named list with \code{trend, seasonal, residual, slope,
#'   intercept, fourier_terms, period, n, method}.
#' @export
prophet_components <- function(x, period = 12) {
  y <- as.numeric(x); n <- length(y)
  if (n < max(2 * period, 6)) stop("Series too short.")
  t <- seq(0, n - 1)
  fit <- lm(y ~ t)
  intercept <- coef(fit)[1]; slope <- coef(fit)[2]
  trend <- fitted(fit)
  detr <- y - trend
  K <- 5
  Fmat <- do.call(cbind, lapply(seq_len(K), function(k)
    cbind(sin(2 * pi * k * t / period), cos(2 * pi * k * t / period))))
  fcoef <- lsfit(Fmat, detr, intercept = FALSE)$coef
  seasonal <- as.numeric(Fmat %*% fcoef)
  residual <- detr - seasonal
  list(trend = trend, seasonal = seasonal, residual = residual,
       slope = as.numeric(slope), intercept = as.numeric(intercept),
       fourier_terms = fcoef, period = period, n = n,
       method = "Prophet-style linear-trend + Fourier(K=5) seasonality")
}


# ── nbeat ────────────────────────────────────────────────────────────────

#' N-BEATS-style polynomial + Fourier basis-expansion forecasting
#'
#' @param x Numeric history.
#' @param horizon Forecast horizon. Default 1.
#' @param n_trend Polynomial-trend degree. Default 3.
#' @param n_season Number of Fourier harmonics. Default 5.
#' @param period Seasonal period. Default 12.
#' @return Named list with \code{forecast, fitted, trend, seasonal,
#'   theta_trend, theta_seasonal, r2, n, horizon, method}.
#' @export
nbeats_basis <- function(x, horizon = 1, n_trend = 3, n_season = 5,
                          period = 12) {
  y <- as.numeric(x); n <- length(y)
  if (n < n_trend + 2 * n_season + 2)
    stop("Series too short for chosen basis.")
  t <- seq(0, n - 1)
  Tmat <- sapply(0:n_trend, function(k) t^k)
  Smat <- do.call(cbind, lapply(seq_len(n_season), function(j)
    cbind(sin(2 * pi * j * t / period),
          cos(2 * pi * j * t / period))))
  Xmat <- cbind(Tmat, Smat)
  coef <- lsfit(Xmat, y, intercept = FALSE)$coef
  fitted_y <- as.numeric(Xmat %*% coef)
  tf <- seq(n, n + horizon - 1)
  Tf <- sapply(0:n_trend, function(k) tf^k)
  Sf <- do.call(cbind, lapply(seq_len(n_season), function(j)
    cbind(sin(2 * pi * j * tf / period),
          cos(2 * pi * j * tf / period))))
  Xf <- cbind(Tf, Sf)
  forecast <- as.numeric(Xf %*% coef)
  theta_trend <- coef[seq_len(n_trend + 1)]
  theta_season <- coef[(n_trend + 2):length(coef)]
  trend <- as.numeric(Tmat %*% theta_trend)
  seasonal <- as.numeric(Smat %*% theta_season)
  ss_tot <- sum((y - mean(y))^2)
  r2 <- if (ss_tot > 0) 1 - sum((y - fitted_y)^2) / ss_tot else NA_real_
  list(forecast = forecast, fitted = fitted_y,
       trend = trend, seasonal = seasonal,
       theta_trend = theta_trend, theta_seasonal = theta_season,
       r2 = r2, n = n, horizon = horizon,
       method = sprintf("N-BEATS basis: poly(P=%d) + Fourier(H=%d, period=%d)",
                         n_trend, n_season, period))
}


# CANONICAL TEST — AR(1) with phi=0.5, n=200
# set.seed(0)
# r <- numeric(200); for (t in 2:200) r[t] <- 0.5 * r[t-1] + rnorm(1)
# str(garch_fit(r));        # alpha + beta < 1, persistence < 1
# str(spectral_density(r))  # frequencies vector, PSD vector, n_segments >= 1
