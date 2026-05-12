# SPDX-License-Identifier: GPL-2.0-only

#' Horowitz semiparametric econometrics suite (R parity)
#'
#' Implements the 20 callables in the morie Horowitz suite, matching the
#' Python module \code{morie.fn.hrz*}.  Each function returns a named
#' list with keys mirroring the Python \code{RichResult} payload.
#'
#' These functions are intentionally not exported through NAMESPACE (the
#' suite is consumed via the umbrella morie API on the Python side); call
#' them as \code{morie:::hrzk1}, etc.
#'
#' @references
#' Horowitz, J. L. (2009). \emph{Semiparametric and Nonparametric Methods
#'   in Econometrics}. Springer Series in Statistics.
#' @keywords internal
#' @name horowitz
NULL


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

.hrz_silverman <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  if (n < 2L) return(1.0)
  s <- stats::sd(x)
  iqr <- diff(stats::quantile(x, c(0.25, 0.75), na.rm = TRUE))
  sigma <- if (iqr > 0) min(s, iqr / 1.349) else s
  if (sigma <= 0) sigma <- max(s, 1e-6)
  unname(1.06 * sigma * n ^ (-1/5))
}

.hrz_R_K_gaussian <- 1.0 / (2.0 * sqrt(pi))

.hrz_gauss_kernel <- function(u) exp(-0.5 * u^2) / sqrt(2 * pi)


# ---------------------------------------------------------------------------
# hrzk1: kernel density estimator
# ---------------------------------------------------------------------------

#' Kernel density estimator (Rosenblatt-Parzen)
#' @keywords internal
hrzk1 <- function(x, bandwidth = NULL, sample = NULL) {
  if (is.null(sample)) {
    data <- as.numeric(x); grid <- data
  } else {
    data <- as.numeric(sample); grid <- as.numeric(x)
  }
  n <- length(data)
  if (n < 2) return(list(estimate = NA_real_, se = NA_real_, n = n,
                          method = "kernel-density (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(data) else as.numeric(bandwidth)
  if (h <= 0) h <- .hrz_silverman(data)
  diffs <- outer(grid, data, `-`) / h
  w <- exp(-0.5 * diffs^2) / sqrt(2 * pi)
  f_hat <- rowMeans(w) / h
  se <- sqrt(pmax(f_hat, 0) * .hrz_R_K_gaussian / (n * h))
  list(estimate = if (length(f_hat) == 1) as.numeric(f_hat) else f_hat,
       se = if (length(se) == 1) as.numeric(se) else se,
       bandwidth = h, n = n, kernel = "gaussian",
       method = "Rosenblatt-Parzen kernel density")
}


# ---------------------------------------------------------------------------
# hrzk2: Nadaraya-Watson regression
# ---------------------------------------------------------------------------

#' Nadaraya-Watson kernel regression
#' @keywords internal
hrzk2 <- function(x, y, bandwidth = NULL, grid = NULL) {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (n < 2 || length(y) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "NW (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(x) else as.numeric(bandwidth)
  if (h <= 0) h <- .hrz_silverman(x)
  g <- if (is.null(grid)) x else as.numeric(grid)
  u <- outer(g, x, `-`) / h
  w <- exp(-0.5 * u^2)
  wsum <- rowSums(w); safe <- ifelse(wsum > 0, wsum, 1)
  m_hat <- (w %*% y) / safe
  resid <- outer(rep(1, length(g)), y) - matrix(m_hat, length(g), n)
  sigma2 <- rowSums(w * resid^2) / safe
  f_hat <- wsum / (n * h * sqrt(2 * pi))
  se <- sqrt(pmax(sigma2, 0) * .hrz_R_K_gaussian / (n * h * pmax(f_hat, 1e-12)))
  list(estimate = as.numeric(m_hat), se = as.numeric(se),
       bandwidth = h, n = n,
       method = "Nadaraya-Watson kernel regression (Gaussian)")
}


# ---------------------------------------------------------------------------
# hrzk3: local-linear regression
# ---------------------------------------------------------------------------

#' Local-linear regression estimator
#' @keywords internal
hrzk3 <- function(x, y, bandwidth = NULL, grid = NULL) {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (n < 3 || length(y) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "local-linear (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(x) else as.numeric(bandwidth)
  if (h <= 0) h <- .hrz_silverman(x)
  g <- if (is.null(grid)) x else as.numeric(grid)
  m_hat <- numeric(length(g)); se <- numeric(length(g))
  for (i in seq_along(g)) {
    u <- (x - g[i]) / h
    w <- exp(-0.5 * u^2)
    if (sum(w) <= 1e-12) { m_hat[i] <- NA; se[i] <- NA; next }
    X <- cbind(1, x - g[i])
    WX <- X * w
    XtWX <- t(X) %*% WX
    beta <- tryCatch(solve(XtWX, t(WX) %*% y),
                     error = function(e) MASS::ginv(XtWX) %*% (t(WX) %*% y))
    m_hat[i] <- beta[1]
    r <- y - X %*% beta
    sigma2 <- sum(w * r^2) / max(sum(w), 1e-12)
    f_hat <- sum(w) / (n * h * sqrt(2 * pi))
    se[i] <- sqrt(max(sigma2, 0) * .hrz_R_K_gaussian / (n * h * max(f_hat, 1e-12)))
  }
  list(estimate = if (length(m_hat) == 1) m_hat[1] else m_hat,
       se = if (length(se) == 1) se[1] else se,
       bandwidth = h, n = n,
       method = "Local-linear regression (Gaussian kernel)")
}


# ---------------------------------------------------------------------------
# hrzp1: partially-linear regression (Robinson 1988)
# ---------------------------------------------------------------------------

.hrz_nw_loo <- function(z, y, h) {
  if (is.null(dim(z))) {
    u <- outer(z, z, `-`) / h
    w <- exp(-0.5 * u^2)
  } else {
    n <- nrow(z); w <- matrix(0, n, n)
    for (j in seq_len(ncol(z))) {
      u <- outer(z[, j], z[, j], `-`) / h
      w <- w + u^2
    }
    w <- exp(-0.5 * w)
  }
  diag(w) <- 0
  wsum <- rowSums(w); safe <- ifelse(wsum > 0, wsum, 1)
  as.numeric((w %*% y) / safe)
}

#' Robinson partially-linear regression
#' @keywords internal
hrzp1 <- function(x, y, z, bandwidth = NULL) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  Z <- if (is.null(dim(z))) matrix(z, ncol = 1) else as.matrix(z)
  n <- length(y)
  if (n < 5 || nrow(X) != n || nrow(Z) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "PLR (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(Z[, 1]) else as.numeric(bandwidth)
  if (h <= 0) h <- max(.hrz_silverman(Z[, 1]), 1e-6)
  Zs <- if (ncol(Z) == 1) Z[, 1] else Z
  mY <- .hrz_nw_loo(Zs, y, h)
  mX <- sapply(seq_len(ncol(X)), function(j) .hrz_nw_loo(Zs, X[, j], h))
  if (is.null(dim(mX))) mX <- matrix(mX, ncol = ncol(X))
  rY <- y - mY; rX <- X - mX
  beta <- tryCatch(MASS::ginv(t(rX) %*% rX) %*% (t(rX) %*% rY),
                   error = function(e) rep(NA_real_, ncol(X)))
  resid <- rY - rX %*% beta
  bread <- MASS::ginv(t(rX) %*% rX)
  meat <- t(rX) %*% (rX * as.numeric(resid)^2)
  cov_m <- bread %*% meat %*% bread
  se <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = if (length(beta) == 1) as.numeric(beta) else as.numeric(beta),
       se = if (length(se) == 1) as.numeric(se) else as.numeric(se),
       bandwidth = h, n = n,
       method = "Robinson (1988) partially-linear regression")
}


# ---------------------------------------------------------------------------
# hrzp2: PLR bandwidth (Silverman)
# ---------------------------------------------------------------------------

#' Silverman bandwidth selector for PLR
#' @keywords internal
hrzp2 <- function(x, y, c = 1.06) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5) return(list(estimate = NA_real_, n = n,
                          method = "plr-bandwidth (insufficient data)"))
  s <- stats::sd(x)
  iqr <- diff(stats::quantile(x, c(0.25, 0.75), na.rm = TRUE))
  sigma <- if (iqr > 0) min(s, iqr / 1.349) else s
  if (sigma <= 0) sigma <- max(s, 1e-6)
  h <- as.numeric(c * sigma * n ^ (-1/5))
  list(estimate = h, n = n, sigma = as.numeric(sigma), c = c,
       method = "Silverman h = c * sigma * n^(-1/5)")
}


# ---------------------------------------------------------------------------
# hrzi1: Ichimura single-index model
# ---------------------------------------------------------------------------

#' Ichimura (1993) single-index model
#' @keywords internal
hrzi1 <- function(x, y, bandwidth = NULL) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "single-index (insufficient data)"))
  beta0 <- as.numeric(stats::coef(stats::lm.fit(X, y)))
  nrm <- sqrt(sum(beta0^2))
  if (nrm < 1e-10) beta0 <- rep(1, p) / sqrt(p) else beta0 <- beta0 / nrm
  if (beta0[1] < 0) beta0 <- -beta0
  h0 <- if (is.null(bandwidth)) .hrz_silverman(X %*% beta0) else as.numeric(bandwidth)

  obj <- function(b) {
    nb <- sqrt(sum(b^2)); if (nb < 1e-12) return(1e12)
    bn <- b / nb; idx <- as.numeric(X %*% bn)
    u <- outer(idx, idx, `-`) / h0
    w <- exp(-0.5 * u^2); diag(w) <- 0
    wsum <- rowSums(w); safe <- ifelse(wsum > 0, wsum, 1)
    g_hat <- as.numeric((w %*% y) / safe)
    mean((y - g_hat)^2)
  }
  res <- stats::optim(beta0, obj, method = "Nelder-Mead",
                       control = list(maxit = 200, reltol = 1e-5))
  bh <- res$par; bh <- bh / max(sqrt(sum(bh^2)), 1e-12)
  if (bh[1] < 0) bh <- -bh
  # Numerical Hessian for SE
  eps <- 1e-4; H <- matrix(0, p, p)
  for (i in 1:p) for (j in 1:p) {
    bp <- bh; bp[i] <- bp[i] + eps; bp[j] <- bp[j] + eps
    bm <- bh; bm[i] <- bm[i] - eps; bm[j] <- bm[j] - eps
    bpm <- bh; bpm[i] <- bpm[i] + eps; bpm[j] <- bpm[j] - eps
    bmp <- bh; bmp[i] <- bmp[i] - eps; bmp[j] <- bmp[j] + eps
    H[i, j] <- (obj(bp) - obj(bpm) - obj(bmp) + obj(bm)) / (4 * eps^2)
  }
  H <- 0.5 * (H + t(H))
  cov_m <- tryCatch(MASS::ginv(H) / n, error = function(e) matrix(NA, p, p))
  se <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = bh, se = se, bandwidth = h0, n = n, loss = res$value,
       method = "Ichimura (1993) single-index model")
}


# ---------------------------------------------------------------------------
# hrzi2: Powell-Stock-Stoker density-weighted average derivative
# ---------------------------------------------------------------------------

#' Density-weighted average derivative
#' @keywords internal
hrzi2 <- function(x, y, bandwidth = NULL) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(20, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "avg-deriv (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(X[, 1]) else as.numeric(bandwidth)
  if (h <= 0) h <- max(.hrz_silverman(X[, 1]), 1e-6)
  # Pairwise differences
  diffs <- array(0, c(n, n, p))
  for (j in seq_len(p)) diffs[, , j] <- outer(X[, j], X[, j], `-`)
  sq <- apply(diffs^2, c(1, 2), sum) / (h^2)
  K <- exp(-0.5 * sq) / ((2 * pi)^(p/2) * h^p)
  diag(K) <- 0
  grad_f <- matrix(0, n, p)
  for (j in seq_len(p)) grad_f[, j] <- -rowSums(diffs[, , j] * K) / (n * h^2)
  delta <- -(2 / n) * colSums(y * grad_f)
  psi <- -2 * y * grad_f
  if (p == 1) {
    se <- sqrt(stats::var(psi) / n)
  } else {
    se <- sqrt(pmax(diag(stats::cov(psi)) / n, 0))
  }
  list(estimate = if (p == 1) as.numeric(delta) else as.numeric(delta),
       se = if (p == 1) as.numeric(se) else as.numeric(se),
       bandwidth = h, n = n,
       method = "Powell-Stock-Stoker density-weighted average derivative")
}


# ---------------------------------------------------------------------------
# hrzb1: Manski maximum-score
# ---------------------------------------------------------------------------

#' Manski (1975) maximum-score estimator
#' @keywords internal
hrzb1 <- function(x, y) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "maximum-score (insufficient data)"))
  ys <- 2 * y - 1
  score <- function(b) -mean(ys * (X %*% b > 0))
  beta0 <- as.numeric(stats::coef(stats::lm.fit(X, ys)))
  nrm <- sqrt(sum(beta0^2)); if (nrm > 1e-12) beta0 <- beta0 / nrm
  if (beta0[1] < 0) beta0 <- -beta0
  best <- beta0; best_l <- score(best)
  set.seed(0)
  for (k in 1:8) {
    s <- stats::rnorm(p); s <- s / sqrt(sum(s^2))
    r <- stats::optim(s, score, method = "Nelder-Mead",
                       control = list(maxit = 300, reltol = 1e-4))
    b <- r$par / max(sqrt(sum(r$par^2)), 1e-12)
    if (b[1] < 0) b <- -b
    l <- score(b)
    if (l < best_l) { best_l <- l; best <- b }
  }
  # Subsample SE (cube-root rescale)
  set.seed(42); B <- 30; m <- max(20L, n %/% 2L)
  boot <- matrix(0, B, p)
  for (b_idx in 1:B) {
    idx <- sample.int(n, m, replace = FALSE)
    Xb <- X[idx, , drop = FALSE]; yb <- ys[idx]
    sc <- function(b) -mean(yb * (Xb %*% b > 0))
    r <- stats::optim(best + 0.05 * stats::rnorm(p), sc, method = "Nelder-Mead",
                       control = list(maxit = 150))
    bb <- r$par / max(sqrt(sum(r$par^2)), 1e-12)
    if (bb[1] < 0) bb <- -bb
    boot[b_idx, ] <- bb
  }
  se <- apply(boot, 2, stats::sd) * (m / n)^(1/3)
  list(estimate = best, se = se, score = -best_l, n = n,
       method = "Manski (1975) maximum-score (binary response)",
       warnings = list("Cube-root asymptotics: subsample-rescaled SEs."))
}


# ---------------------------------------------------------------------------
# hrzb2: Horowitz smoothed maximum-score
# ---------------------------------------------------------------------------

#' Horowitz (1992) smoothed maximum-score estimator
#' @keywords internal
hrzb2 <- function(x, y, bandwidth = NULL) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "smoothed-max-score (insufficient data)"))
  ys <- 2 * y - 1
  h <- if (is.null(bandwidth)) max(.hrz_silverman(X %*% rep(1 / sqrt(p), p)), 1e-3)
       else as.numeric(bandwidth)
  loss <- function(b) {
    nb <- sqrt(sum(b^2)); if (nb < 1e-12) return(1e12)
    bn <- b / nb; z <- (X %*% bn) / h
    -mean(ys * stats::pnorm(z))
  }
  beta0 <- as.numeric(stats::coef(stats::lm.fit(X, ys)))
  nrm <- sqrt(sum(beta0^2)); if (nrm > 1e-12) beta0 <- beta0 / nrm
  if (beta0[1] < 0) beta0 <- -beta0
  res <- stats::optim(beta0, loss, method = "BFGS",
                       control = list(maxit = 200))
  bh <- res$par / max(sqrt(sum(res$par^2)), 1e-12)
  if (bh[1] < 0) bh <- -bh
  z <- (X %*% bh) / h
  phi <- stats::dnorm(z)
  score_i <- -as.numeric(ys * phi) * X / h
  info <- t(score_i) %*% score_i / n
  cov_m <- tryCatch(MASS::ginv(info) / n,
                    error = function(e) matrix(NA, p, p))
  se <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = bh, se = se, bandwidth = h, n = n,
       method = "Horowitz (1992) smoothed maximum-score")
}


# ---------------------------------------------------------------------------
# hrzc1: Powell censored LAD (CLAD)
# ---------------------------------------------------------------------------

.hrz_qreg_irls <- function(X, y, tau = 0.5, maxit = 50, tol = 1e-6) {
  beta <- as.numeric(stats::coef(stats::lm.fit(X, y)))
  for (k in 1:maxit) {
    r <- y - X %*% beta
    w <- ifelse(r > 0, tau / pmax(r, 1e-6),
                (1 - tau) / pmax(-r, 1e-6))
    w <- as.numeric(w)
    new <- tryCatch(solve(t(X) %*% (X * w), t(X) %*% (w * y)),
                    error = function(e) MASS::ginv(t(X) %*% (X * w)) %*% (t(X) %*% (w * y)))
    if (max(abs(new - beta)) < tol) { beta <- new; break }
    beta <- new
  }
  as.numeric(beta)
}

#' Powell (1984) censored LAD (CLAD)
#' @keywords internal
hrzc1 <- function(x, y, censor = 0.0) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X); c <- as.numeric(censor)
  if (n < max(10, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "CLAD (insufficient data)"))
  keep <- y > c
  if (sum(keep) < max(5, p + 1))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "CLAD (too few uncensored obs)"))
  beta <- .hrz_qreg_irls(X[keep, , drop = FALSE], y[keep])
  for (k in 1:30) {
    active <- as.numeric(X %*% beta) > c
    if (sum(active) < max(5, p + 1)) break
    new <- .hrz_qreg_irls(X[active, , drop = FALSE], y[active])
    if (max(abs(new - beta)) < 1e-5) { beta <- new; break }
    beta <- new
  }
  r <- y - as.numeric(X %*% beta); active <- as.numeric(X %*% beta) > c
  if (sum(active) < max(5, p + 1)) {
    se <- rep(NA_real_, p)
  } else {
    Xa <- X[active, , drop = FALSE]; ra <- r[active]
    h <- max(1.06 * stats::sd(ra) * length(ra)^(-1/5), 1e-4)
    f0 <- mean(exp(-0.5 * (ra / h)^2) / (h * sqrt(2 * pi)))
    A <- t(Xa) %*% Xa * f0
    cov_m <- tryCatch(0.25 * MASS::ginv(A) %*% (t(Xa) %*% Xa) %*% MASS::ginv(A),
                      error = function(e) matrix(NA, p, p))
    se <- sqrt(pmax(diag(cov_m), 0))
  }
  list(estimate = if (p == 1) as.numeric(beta) else as.numeric(beta),
       se = if (p == 1) as.numeric(se) else as.numeric(se),
       n = n, n_uncensored = as.integer(sum(active)), censor = c,
       method = "Powell (1984) censored LAD")
}


# ---------------------------------------------------------------------------
# hrzs1: semiparametric sample-selection
# ---------------------------------------------------------------------------

.hrz_probit_newton <- function(D, Z, maxit = 50, tol = 1e-8) {
  q <- ncol(Z); beta <- rep(0, q)
  for (k in 1:maxit) {
    eta <- pmin(pmax(as.numeric(Z %*% beta), -50), 50)
    p <- stats::pnorm(eta); phi <- stats::dnorm(eta)
    w <- phi * (D - p) / pmax(p * (1 - p), 1e-8)
    Hd <- phi^2 / pmax(p * (1 - p), 1e-8)
    g <- t(Z) %*% w
    H <- t(Z) %*% (Z * Hd)
    step <- tryCatch(solve(H + 1e-8 * diag(q), g),
                     error = function(e) MASS::ginv(H) %*% g)
    beta <- beta + step
    if (max(abs(step)) < tol) break
  }
  as.numeric(beta)
}

#' Heckman-Powell-Newey-Vella semiparametric sample-selection
#' @keywords internal
hrzs1 <- function(x, y, z, d) {
  y <- as.numeric(y); d <- as.numeric(d)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  Z <- if (is.null(dim(z))) matrix(z, ncol = 1) else as.matrix(z)
  n <- length(y)
  if (n < 20 || nrow(X) != n || nrow(Z) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "sample-selection (insufficient data)"))
  Zc <- if (all(Z[, 1] == 1)) Z else cbind(1, Z)
  gamma <- .hrz_probit_newton(d, Zc)
  eta <- as.numeric(Zc %*% gamma)
  mills <- stats::dnorm(eta) / pmax(stats::pnorm(eta), 1e-8)
  Xc <- if (all(X[, 1] == 1)) X else cbind(1, X)
  sel <- d > 0.5
  if (sum(sel) < max(10, ncol(Xc) + 2))
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "sample-selection (too few selected)"))
  M <- cbind(Xc[sel, , drop = FALSE], mills[sel])
  yy <- y[sel]
  coef <- as.numeric(MASS::ginv(t(M) %*% M) %*% (t(M) %*% yy))
  beta <- coef[seq_len(ncol(Xc))]
  rho_sigma <- coef[length(coef)]
  resid <- yy - as.numeric(M %*% coef)
  sigma2 <- mean(resid^2)
  cov_m <- sigma2 * MASS::ginv(t(M) %*% M)
  se_all <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = as.numeric(beta),
       se = as.numeric(se_all[seq_len(ncol(Xc))]),
       selection_correction = as.numeric(rho_sigma), n = n,
       n_selected = as.integer(sum(sel)),
       method = "Semiparametric Heckman/Powell-Newey-Vella sample selection")
}


# ---------------------------------------------------------------------------
# hrzd1: Cox proportional hazards
# ---------------------------------------------------------------------------

#' Cox partial-likelihood proportional-hazards estimator
#' @keywords internal
hrzd1 <- function(t, x, event) {
  t <- as.numeric(t); event <- as.numeric(event)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p) || length(t) != n || length(event) != n)
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "Cox PH (insufficient data)"))
  o <- order(-t)
  Xs <- X[o, , drop = FALSE]; ev <- event[o]
  beta <- rep(0, p)
  H <- diag(p)
  for (it in 1:50) {
    eta <- pmin(pmax(as.numeric(Xs %*% beta), -50), 50)
    ehb <- exp(eta)
    S0 <- cumsum(ehb)
    S1 <- apply(Xs * ehb, 2, cumsum)
    if (is.null(dim(S1))) S1 <- matrix(S1, ncol = p)
    mean_X <- S1 / pmax(S0, 1e-12)
    diff_X <- Xs - mean_X
    score <- colSums(ev * diff_X)
    # build cumulative S2 (n, p, p)
    S2 <- array(0, c(n, p, p))
    for (i in 1:p) for (j in 1:p) {
      S2[, i, j] <- cumsum(Xs[, i] * Xs[, j] * ehb)
    }
    var_X <- array(0, c(n, p, p))
    for (i in 1:p) for (j in 1:p) {
      var_X[, i, j] <- S2[, i, j] / pmax(S0, 1e-12) - mean_X[, i] * mean_X[, j]
    }
    info <- matrix(0, p, p)
    for (i in 1:p) for (j in 1:p) info[i, j] <- sum(ev * var_X[, i, j])
    step <- tryCatch(solve(info + 1e-8 * diag(p), score),
                     error = function(e) MASS::ginv(info) %*% score)
    beta <- beta + as.numeric(step)
    if (max(abs(step)) < 1e-6) break
  }
  cov_m <- tryCatch(MASS::ginv(info), error = function(e) matrix(NA, p, p))
  se <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = if (p == 1) as.numeric(beta) else as.numeric(beta),
       se = if (p == 1) as.numeric(se) else as.numeric(se),
       n = n, n_events = as.integer(sum(event)),
       method = "Cox proportional hazards (partial likelihood)")
}


# ---------------------------------------------------------------------------
# hrzt1: kernel-matching ATE
# ---------------------------------------------------------------------------

.hrz_logit_newton <- function(D, X, maxit = 50, tol = 1e-8) {
  p <- ncol(X); beta <- rep(0, p)
  for (k in 1:maxit) {
    eta <- pmin(pmax(as.numeric(X %*% beta), -50), 50)
    mu <- 1 / (1 + exp(-eta))
    W <- mu * (1 - mu)
    g <- t(X) %*% (D - mu)
    H <- t(X) %*% (X * W)
    step <- tryCatch(solve(H + 1e-8 * diag(p), g),
                     error = function(e) MASS::ginv(H) %*% g)
    beta <- beta + step
    if (max(abs(step)) < tol) break
  }
  1 / (1 + exp(-pmin(pmax(as.numeric(X %*% beta), -50), 50)))
}

#' Heckman-Ichimura-Todd kernel-matching ATE
#' @keywords internal
hrzt1 <- function(x, y, treatment, bandwidth = NULL, .bootstrap = TRUE) {
  y <- as.numeric(y); D <- as.numeric(treatment)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- length(y)
  if (n < 30 || length(D) != n || nrow(X) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "kernel-matching ATE (insufficient data)"))
  Xp <- if (!isTRUE(all(X[, 1] == 1))) cbind(1, X) else X
  e <- .hrz_logit_newton(D, Xp)
  e <- pmin(pmax(e, 1e-6), 1 - 1e-6)
  h <- if (is.null(bandwidth)) max(.hrz_silverman(e), 1e-3) else as.numeric(bandwidth)
  t_idx <- which(D > 0.5); c_idx <- which(D < 0.5)
  if (length(t_idx) < 2 || length(c_idx) < 2)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "kernel-matching ATE (one arm empty)"))
  e_t <- e[t_idx]; e_c <- e[c_idx]
  u <- outer(e_t, e_c, `-`) / h; K <- exp(-0.5 * u^2)
  w <- K / pmax(rowSums(K), 1e-12)
  cf_t <- as.numeric(w %*% y[c_idx])
  u2 <- outer(e_c, e_t, `-`) / h; K2 <- exp(-0.5 * u2^2)
  w2 <- K2 / pmax(rowSums(K2), 1e-12)
  cf_c <- as.numeric(w2 %*% y[t_idx])
  att <- mean(y[t_idx] - cf_t); atu <- mean(cf_c - y[c_idx])
  ate <- (length(t_idx) * att + length(c_idx) * atu) / n
  # Bootstrap SE (guarded against recursive blow-up)
  se <- NA_real_
  if (.bootstrap) {
    set.seed(0); B <- 50; boot <- numeric(B)
    for (b in 1:B) {
      idx <- sample.int(n, n, replace = TRUE)
      sub <- tryCatch(hrzt1(X[idx, , drop = FALSE], y[idx], D[idx],
                            bandwidth = h, .bootstrap = FALSE),
                      error = function(e) list(estimate = ate))
      boot[b] <- if (is.numeric(sub$estimate) && !is.na(sub$estimate)) sub$estimate else ate
    }
    se <- as.numeric(stats::sd(boot))
  }
  list(estimate = as.numeric(ate), se = se,
       att = att, atu = atu, bandwidth = h, n = n,
       n_treated = as.integer(length(t_idx)),
       n_control = as.integer(length(c_idx)),
       method = "Kernel-matching ATE (Heckman-Ichimura-Todd)")
}


# ---------------------------------------------------------------------------
# hrzt2: IV LATE (Wald)
# ---------------------------------------------------------------------------

#' IV Wald estimator for LATE (Imbens-Angrist)
#' @keywords internal
hrzt2 <- function(x, y, z, treatment) {
  y <- as.numeric(y); z <- as.numeric(z); D <- as.numeric(treatment)
  n <- length(y)
  if (n < 20 || length(z) != n || length(D) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "LATE (insufficient data)"))
  uniq <- unique(z)
  z_bin <- if (length(uniq) > 2) as.numeric(z > stats::median(z)) else
    as.numeric(z == max(uniq))
  n1 <- sum(z_bin > 0.5); n0 <- sum(z_bin < 0.5)
  if (n1 < 5 || n0 < 5)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "LATE (one arm of Z empty)"))
  Y1 <- mean(y[z_bin > 0.5]); Y0 <- mean(y[z_bin < 0.5])
  D1 <- mean(D[z_bin > 0.5]); D0 <- mean(D[z_bin < 0.5])
  num <- Y1 - Y0; den <- D1 - D0
  if (abs(den) < 1e-8)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "LATE (weak instrument)"))
  late <- num / den
  vY <- stats::var(y[z_bin > 0.5]) / n1 + stats::var(y[z_bin < 0.5]) / n0
  vD <- stats::var(D[z_bin > 0.5]) / n1 + stats::var(D[z_bin < 0.5]) / n0
  v_late <- (vY + late^2 * vD) / den^2
  list(estimate = as.numeric(late), se = sqrt(max(v_late, 0)),
       first_stage = as.numeric(den), reduced_form = as.numeric(num),
       n = n,
       method = "IV Wald estimator (Imbens-Angrist LATE)")
}


# ---------------------------------------------------------------------------
# hrzq1: linear quantile regression (Koenker-Bassett)
# ---------------------------------------------------------------------------

#' Koenker-Bassett linear quantile regression
#' @keywords internal
hrzq1 <- function(x, y, tau = 0.5) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p) || !(tau > 0 && tau < 1))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, tau = tau,
                method = "QReg (insufficient data or invalid tau)"))
  has_int <- isTRUE(all(X[, 1] == 1))
  Xp <- if (!has_int) cbind(1, X) else X
  beta <- .hrz_qreg_irls(Xp, y, tau)
  r <- y - as.numeric(Xp %*% beta)
  h <- (stats::qnorm(1 - 0.05)^(2/3)) *
    ((1.5 * stats::dnorm(stats::qnorm(tau))^2) /
       (2 * stats::qnorm(tau)^2 + 1))^(1/3) * n^(-1/3)
  h <- max(h, 1e-3)
  f0 <- mean(abs(r) < h) / (2 * h); if (f0 < 1e-6) f0 <- 1e-6
  cov_m <- (tau * (1 - tau) / f0^2) * MASS::ginv(t(Xp) %*% Xp)
  se_all <- sqrt(pmax(diag(cov_m), 0))
  if (!has_int) {
    beta_out <- if (ncol(Xp) > 1) beta[-1] else beta
    se_out <- if (ncol(Xp) > 1) se_all[-1] else se_all
    intercept <- as.numeric(beta[1])
  } else { beta_out <- beta; se_out <- se_all; intercept <- NULL }
  list(estimate = if (length(beta_out) == 1) as.numeric(beta_out) else beta_out,
       se = if (length(se_out) == 1) as.numeric(se_out) else se_out,
       intercept = intercept, tau = tau, n = n,
       method = "Koenker-Bassett quantile regression (IRLS)")
}


# ---------------------------------------------------------------------------
# hrzm1: Gaussian mixture model (EM)
# ---------------------------------------------------------------------------

#' k-component Gaussian mixture EM
#' @keywords internal
hrzm1 <- function(y, k = 2, maxit = 200, tol = 1e-6, seed = 0) {
  y <- as.numeric(y); n <- length(y)
  if (n < max(10, 3 * k))
    return(list(estimate = NA_real_, n = n,
                method = "mixture-EM (insufficient data)"))
  set.seed(seed)
  mu <- as.numeric(stats::quantile(y, seq(0.1, 0.9, length.out = k)))
  sigma <- rep(stats::sd(y) / k + 1e-3, k)
  pii <- rep(1 / k, k)
  ll_prev <- -Inf; it <- 0
  for (it in 1:maxit) {
    comps <- sapply(1:k, function(j) pii[j] * stats::dnorm(y, mu[j], sigma[j]))
    denom <- rowSums(comps); denom <- ifelse(denom > 0, denom, 1e-12)
    gamma_w <- comps / denom
    Nk <- colSums(gamma_w); Nk <- ifelse(Nk > 0, Nk, 1e-12)
    mu <- colSums(gamma_w * y) / Nk
    sigma <- sqrt(colSums(gamma_w * (y - matrix(mu, n, k, byrow = TRUE))^2) / Nk)
    sigma <- pmax(sigma, 1e-4)
    pii <- Nk / n
    ll <- sum(log(pmax(rowSums(comps), 1e-300)))
    if (abs(ll - ll_prev) < tol) break
    ll_prev <- ll
  }
  list(estimate = list(pi = as.numeric(pii), mu = as.numeric(mu),
                        sigma = as.numeric(sigma)),
       log_likelihood = ll_prev, n = n, k = k, iters = it,
       method = sprintf("%d-component Gaussian mixture EM", k))
}


# ---------------------------------------------------------------------------
# hrzn1: nonparametric IV (series-Tikhonov)
# ---------------------------------------------------------------------------

.hrz_hermite <- function(t, J) {
  n <- length(t); H <- matrix(0, n, J)
  H[, 1] <- 1
  if (J > 1) H[, 2] <- t
  if (J > 2) for (k in 3:J) H[, k] <- t * H[, k - 1] - (k - 2) * H[, k - 2]
  for (k in 1:J) H[, k] <- H[, k] / sqrt(max(factorial(k - 1), 1))
  H
}

#' Series-Tikhonov nonparametric instrumental variables
#' @keywords internal
hrzn1 <- function(x, y, z, J = 5, alpha = 1e-3, grid = NULL,
                  .bootstrap = TRUE) {
  x <- as.numeric(x); y <- as.numeric(y); z <- as.numeric(z)
  n <- length(y)
  if (n < 50 || length(x) != n || length(z) != n) {
    # 2SLS fallback
    Xc <- cbind(1, x); Zc <- cbind(1, z)
    Pz <- Zc %*% MASS::ginv(t(Zc) %*% Zc) %*% t(Zc)
    beta <- MASS::ginv(t(Xc) %*% Pz %*% Xc) %*% (t(Xc) %*% Pz %*% y)
    return(list(estimate = as.numeric(beta[2]), se = NA_real_, n = n,
                method = "NPIV fallback: linear 2SLS"))
  }
  x_s <- (x - mean(x)) / max(stats::sd(x), 1e-6)
  z_s <- (z - mean(z)) / max(stats::sd(z), 1e-6)
  Bx <- .hrz_hermite(x_s, J); Bz <- .hrz_hermite(z_s, J)
  M <- (t(Bz) %*% Bx) / n
  BzY <- as.numeric((t(Bz) %*% y) / n)
  BzBz <- (t(Bz) %*% Bz) / n
  inv_BzBz <- MASS::ginv(BzBz + alpha * diag(J))
  A <- t(M) %*% inv_BzBz %*% M + alpha * diag(J)
  rhs <- t(M) %*% inv_BzBz %*% BzY
  coef <- solve(A, rhs)
  if (is.null(grid)) grid <- seq(min(x), max(x), length.out = 21)
  grid <- as.numeric(grid)
  grid_s <- (grid - mean(x)) / max(stats::sd(x), 1e-6)
  Bx_g <- .hrz_hermite(grid_s, J)
  g_hat <- as.numeric(Bx_g %*% coef)
  # Bootstrap SE (guarded against recursion explosion)
  if (.bootstrap) {
    set.seed(0); B <- 30
    boot <- matrix(0, B, length(grid))
    for (b in 1:B) {
      idx <- sample.int(n, n, replace = TRUE)
      sub <- tryCatch(hrzn1(x[idx], y[idx], z[idx], J = J, alpha = alpha,
                            grid = grid, .bootstrap = FALSE),
                      error = function(e) list(estimate = g_hat))
      boot[b, ] <- as.numeric(sub$estimate)
    }
    se <- apply(boot, 2, stats::sd)
  } else {
    se <- rep(NA_real_, length(grid))
  }
  list(estimate = g_hat, se = as.numeric(se), grid = grid, J = J, alpha = alpha,
       n = n, method = "Series-Tikhonov NPIV on Hermite basis")
}


# ---------------------------------------------------------------------------
# hrzn2: Fourier deconvolution density
# ---------------------------------------------------------------------------

#' Stefanski-Carroll Fourier-deconvolution density estimator
#' @keywords internal
hrzn2 <- function(y, sigma_u = 0.5, bandwidth = NULL, grid = NULL,
                  noise = "laplace") {
  y <- as.numeric(y); n <- length(y)
  if (n < 30) return(list(estimate = NA_real_, n = n,
                          method = "deconvolution (insufficient data)"))
  h <- if (is.null(bandwidth)) max(1.5 * stats::sd(y) * n^(-1/7), 1e-3)
       else as.numeric(bandwidth)
  if (is.null(grid)) grid <- seq(min(y), max(y), length.out = 51)
  grid <- as.numeric(grid)
  T <- seq(-15, 15, length.out = 2049) / max(h, 1e-3)
  dt <- T[2] - T[1]
  phi_Y <- colMeans(exp(1i * outer(y, T)))
  phi_U <- if (noise == "normal") exp(-0.5 * (sigma_u * T)^2)
          else 1 / (1 + (sigma_u * T)^2)
  th <- T * h
  phi_K <- ifelse(abs(th) <= 1, (1 - th^2)^3, 0)
  integrand <- phi_K * phi_Y / ifelse(abs(phi_U) > 1e-10, phi_U, complex(real = Inf))
  f_hat <- numeric(length(grid))
  for (i in seq_along(grid)) {
    f_hat[i] <- Re(sum(exp(-1i * T * grid[i]) * integrand)) * dt / (2 * pi)
  }
  f_hat <- pmax(f_hat, 0)
  list(estimate = f_hat, grid = grid, bandwidth = h,
       sigma_u = as.numeric(sigma_u), noise = noise, n = n,
       method = "Fourier deconvolution density (sinc kernel)")
}


# ---------------------------------------------------------------------------
# hrzw1: Rademacher wild bootstrap
# ---------------------------------------------------------------------------

#' Rademacher wild bootstrap for OLS coefficients
#' @keywords internal
hrzw1 <- function(x, y, residuals = NULL, B = 500, seed = 0) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p))
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "wild-bootstrap (insufficient data)"))
  beta0 <- as.numeric(MASS::ginv(t(X) %*% X) %*% (t(X) %*% y))
  res <- if (is.null(residuals)) y - as.numeric(X %*% beta0)
         else as.numeric(residuals)
  set.seed(seed)
  XtX_inv <- MASS::ginv(t(X) %*% X)
  boot <- matrix(0, B, p)
  for (b in 1:B) {
    v <- sample(c(-1, 1), n, replace = TRUE)
    y_star <- as.numeric(X %*% beta0) + res * v
    boot[b, ] <- as.numeric(XtX_inv %*% (t(X) %*% y_star))
  }
  mean_b <- colMeans(boot); se <- apply(boot, 2, stats::sd)
  ci_lo <- apply(boot, 2, stats::quantile, 0.025)
  ci_hi <- apply(boot, 2, stats::quantile, 0.975)
  list(estimate = if (p == 1) as.numeric(beta0) else beta0,
       se = if (p == 1) as.numeric(se) else se,
       ci_lower = if (p == 1) as.numeric(ci_lo) else ci_lo,
       ci_upper = if (p == 1) as.numeric(ci_hi) else ci_hi,
       boot_mean = if (p == 1) as.numeric(mean_b) else mean_b,
       B = B, n = n,
       method = "Rademacher wild bootstrap (Mammen 1993)")
}


# ---------------------------------------------------------------------------
# hrzw2: bootstrap bandwidth selection
# ---------------------------------------------------------------------------

#' Wild-bootstrap MISE bandwidth selection for NW regression
#' @keywords internal
hrzw2 <- function(x, y, B = 50, n_h = 15, seed = 0) {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (n < 30 || length(y) != n)
    return(list(estimate = NA_real_, n = n,
                method = "bw-bootstrap (insufficient data)"))
  nw_fit <- function(x_train, y_train, x_eval, h) {
    u <- outer(x_eval, x_train, `-`) / h
    w <- exp(-0.5 * u^2); s <- rowSums(w); safe <- ifelse(s > 0, s, 1)
    as.numeric((w %*% y_train) / safe)
  }
  h_sil <- .hrz_silverman(x)
  h_grid <- seq(0.5 * h_sil, 2.5 * h_sil, length.out = n_h)
  m_pilot <- nw_fit(x, y, x, h_sil)
  r <- y - m_pilot
  set.seed(seed)
  mise <- numeric(n_h)
  for (j in seq_along(h_grid)) {
    ise <- 0
    for (b in 1:B) {
      v <- sample(c(-1, 1), n, replace = TRUE)
      y_star <- m_pilot + r * v
      m_star <- nw_fit(x, y_star, x, h_grid[j])
      ise <- ise + mean((m_star - m_pilot)^2)
    }
    mise[j] <- ise / B
  }
  j_star <- which.min(mise)
  list(estimate = as.numeric(h_grid[j_star]), h_silverman = as.numeric(h_sil),
       mise_curve = mise, h_grid = h_grid, n = n, B = B,
       method = "Wild-bootstrap MISE bandwidth selection (Faraway-Jhun)")
}
