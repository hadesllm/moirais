# SPDX-License-Identifier: AGPL-3.0-or-later

# --- R-side Hawkes-process fitter (v0.9.1, task #74) ----------------------
#
# Fits a self-exciting (Hawkes) point process to event times by maximum
# likelihood. The negative log-likelihood runs on the shared C++ core
# (morie_core.hpp) via the Rcpp bindings in morie_fast.cpp -- the same
# kernels the Python side uses -- and falls back to a pure-R O(n^2)
# likelihood when the compiled core is unavailable.
#
# Constant-baseline triggering kernels: exponential, Weibull, Lomax,
# gamma. Parameter vector theta is (a0, eta, <shape/scale ...>): a0 is
# the log baseline (nu = exp(a0)), eta the branching ratio in (0, 1).

.hawkes_param_names <- function(kernel) {
  switch(kernel,
    exponential = c("a0", "eta", "beta"),
    weibull     = c("a0", "eta", "alpha", "lambda"),
    lomax       = c("a0", "eta", "alpha", "c"),
    gamma       = c("a0", "eta", "alpha", "beta"),
    stop("unknown kernel: ", kernel))
}

# Optimisation runs in an unconstrained space phi to avoid the hard
# feasibility cliffs: a0 is free, eta = plogis(phi) in (0, 1), and the
# shape/scale parameters are exp(phi) > 0.
.hawkes_to_theta <- function(phi) {
  theta <- phi
  theta[2] <- stats::plogis(phi[2])
  theta[-(1:2)] <- exp(phi[-(1:2)])
  theta
}

.hawkes_to_phi <- function(theta) {
  phi <- theta
  phi[2] <- stats::qlogis(theta[2])
  phi[-(1:2)] <- log(theta[-(1:2)])
  phi
}

.hawkes_nll_cpp <- function(theta, times, end_time, kernel) {
  switch(kernel,
    exponential = morie_hawkes_ll_exp_const_cpp(
      times, end_time, theta[1], theta[2], theta[3]),
    weibull = morie_hawkes_ll_weibull_const_cpp(
      times, end_time, theta[1], theta[2], theta[3], theta[4]),
    lomax = morie_hawkes_ll_lomax_const_cpp(
      times, end_time, theta[1], theta[2], theta[3], theta[4]),
    gamma = morie_hawkes_ll_gamma_const_cpp(
      times, end_time, theta[1], theta[2], theta[3], theta[4]))
}

# Triggering kernel g(u) and its integral G(u) = integral_0^u g, for the
# pure-R fallback. Returns NULL for an infeasible parameter vector.
.hawkes_kernel_funs <- function(kernel, theta) {
  if (kernel == "exponential") {
    beta <- theta[3]
    if (beta <= 1e-6) return(NULL)
    list(g = function(u) beta * exp(-beta * u),
         G = function(u) 1 - exp(-beta * u))
  } else if (kernel == "weibull") {
    alpha <- theta[3]; lam <- theta[4]
    if (alpha <= 1e-6 || lam <= 1e-6) return(NULL)
    list(g = function(u) (alpha / lam) * (u / lam)^(alpha - 1) *
           exp(-(u / lam)^alpha),
         G = function(u) 1 - exp(-(u / lam)^alpha))
  } else if (kernel == "lomax") {
    alpha <- theta[3]; cc <- theta[4]
    if (alpha <= 1.001 || cc <= 1e-6) return(NULL)
    list(g = function(u) ((alpha - 1) / cc) * (1 + u / cc)^(-alpha),
         G = function(u) 1 - (1 + u / cc)^(-(alpha - 1)))
  } else {
    alpha <- theta[3]; beta <- theta[4]
    if (alpha <= 1e-6 || beta <= 1e-6) return(NULL)
    list(g = function(u) stats::dgamma(u, shape = alpha, rate = beta),
         G = function(u) stats::pgamma(u, shape = alpha, rate = beta))
  }
}

.hawkes_nll_pureR <- function(theta, times, end_time, kernel) {
  nu <- exp(theta[1]); eta <- theta[2]
  if (!is.finite(nu) || nu <= 0 || eta <= 1e-6 || eta >= 0.999) {
    return(1e12)
  }
  funs <- .hawkes_kernel_funs(kernel, theta)
  if (is.null(funs)) return(1e12)
  n <- length(times)
  log_sum <- 0.0
  for (i in seq_len(n)) {
    lam <- nu
    if (i > 1L) {
      lam <- nu + eta * sum(funs$g(times[i] - times[seq_len(i - 1L)]))
    }
    if (!is.finite(lam) || lam <= 0) return(1e12)
    log_sum <- log_sum + log(lam)
  }
  integral <- nu * end_time + eta * sum(funs$G(end_time - times))
  if (!is.finite(log_sum) || !is.finite(integral)) return(1e12)
  -(log_sum - integral)
}

.hawkes_start <- function(kernel, times, end_time) {
  n <- length(times)
  dt_bar <- end_time / n                       # mean inter-arrival
  a0 <- log(max(0.5 * n / end_time, 1e-3))
  eta <- 0.3
  switch(kernel,
    exponential = c(a0, eta, 1 / dt_bar),
    weibull     = c(a0, eta, 1.2, dt_bar),
    lomax       = c(a0, eta, 2.0, dt_bar),
    gamma       = c(a0, eta, 1.5, 1 / dt_bar))
}

# Closed-form log-likelihood of the homogeneous Poisson submodel
# (eta = 0): its MLE baseline is nu_hat = n / end_time, giving
# logLik = n*log(n/T) - n. The Hawkes family nests this, so the Hawkes
# MLE log-likelihood can never fall below it.
.hawkes_loglik_poisson <- function(n, end_time) {
  n * log(n / end_time) - n
}

# Deterministic multi-start set in the unconstrained space. No RNG, so
# the fit is reproducible regardless of the caller's random seed. The
# perturbations span lower / higher eta and shifted baseline / shape;
# the lower-eta start in particular lets the optimiser reach the
# Poisson submodel when the data carries no self-excitation.
.hawkes_restarts <- function(phi0) {
  offsets <- list(
    c(0, 0, 0, 0),
    c(0, 2.0, 0, 0),
    c(0, -2.5, 0, 0),
    c(0.7, 0, 0.7, 0),
    c(-0.7, 1.0, -0.7, 0.5))
  lapply(offsets, function(o) phi0 + o[seq_along(phi0)])
}

#' Fit a Hawkes (self-exciting point process) model by maximum likelihood
#'
#' Fits a one-dimensional Hawkes process with a constant baseline to a
#' vector of event times. The conditional intensity is
#' \deqn{lambda(t) = nu + eta * sum_{t_j < t} g(t - t_j),}
#' where \code{nu = exp(a0)} is the baseline rate, \code{eta} the
#' branching ratio, and \code{g} the chosen triggering kernel.
#'
#' The negative log-likelihood is evaluated by the shared morie C++
#' core (the same kernels the Python package uses); without a compiled
#' core it falls back to a pure-R \eqn{O(n^2)} likelihood.
#'
#' @param times numeric vector of sorted, non-decreasing event times.
#' @param end_time observation horizon; defaults to the last event time.
#' @param kernel triggering kernel: one of \code{"exponential"},
#'   \code{"weibull"}, \code{"lomax"}, \code{"gamma"}.
#' @return An object of class \code{morie_hawkes_fit}: a list with the
#'   parameter \code{estimate}, \code{loglik}, \code{aic},
#'   \code{branching_ratio}, \code{baseline_rate}, \code{n_events},
#'   \code{converged} and the \code{backend} used.
#' @examples
#' \dontrun{
#'   set.seed(1)
#'   ev <- cumsum(rexp(200, rate = 2))
#'   fit <- morie_hawkes_fit(ev, kernel = "exponential")
#'   print(fit)
#' }
#' @export
morie_hawkes_fit <- function(times, end_time = NULL,
                             kernel = c("exponential", "weibull",
                                        "lomax", "gamma")) {
  kernel <- match.arg(kernel)
  times <- as.numeric(times)
  if (anyNA(times) || any(diff(times) < 0)) {
    stop("`times` must be sorted, non-decreasing, non-NA event times")
  }
  n <- length(times)
  if (n < 2L) stop("need at least 2 events to fit a Hawkes process")
  if (is.null(end_time)) end_time <- times[n]
  end_time <- as.numeric(end_time)
  if (end_time < times[n]) {
    stop("`end_time` must be >= the last event time")
  }

  use_cpp <- .cpp_available()
  obj <- function(phi) {
    theta <- .hawkes_to_theta(phi)
    if (use_cpp) {
      .hawkes_nll_cpp(theta, times, end_time, kernel)
    } else {
      .hawkes_nll_pureR(theta, times, end_time, kernel)
    }
  }

  # multi-start: optimise from the canonical start plus deterministic
  # perturbations and keep the lowest negative log-likelihood. Guards
  # against Nelder-Mead settling in a local optimum on excited data.
  phi0 <- .hawkes_to_phi(.hawkes_start(kernel, times, end_time))
  best <- NULL
  for (phi_s in .hawkes_restarts(phi0)) {
    run <- tryCatch(
      stats::optim(phi_s, obj, method = "Nelder-Mead",
                   control = list(maxit = 2000, reltol = 1e-10)),
      error = function(e) NULL)
    if (is.null(run) || !is.finite(run$value)) next
    if (is.null(best) || run$value < best$value) best <- run
  }
  if (is.null(best)) {
    stop("Hawkes fit failed: no starting point produced a valid optimum")
  }

  est <- .hawkes_to_theta(best$par)
  names(est) <- .hawkes_param_names(kernel)
  loglik <- -best$value
  k <- length(est)
  loglik_pois <- .hawkes_loglik_poisson(n, end_time)
  eta <- unname(est[["eta"]])
  # eta ~ 0 -> the triggering term vanishes, so the kernel-shape
  # parameters are unidentified and the data is consistent with a
  # homogeneous Poisson process. Report that rather than a spurious
  # non-convergence: the log-likelihood IS maximised, the MLE is just
  # a flat ridge in the unidentified directions.
  degenerate <- eta < 1e-3
  structure(
    list(kernel = kernel, baseline = "constant",
         estimate = est,
         branching_ratio = eta,
         baseline_rate = exp(unname(est[["a0"]])),
         loglik = loglik, aic = 2 * k - 2 * loglik,
         loglik_poisson = loglik_pois,
         loglik_gain = loglik - loglik_pois,
         self_excitation_detected = !degenerate,
         n_events = n, end_time = end_time,
         converged = best$convergence == 0L || degenerate,
         note = if (degenerate)
           paste("eta collapsed to ~0: data consistent with a",
                 "homogeneous Poisson process; kernel-shape",
                 "parameters are NOT identified")
         else NULL,
         backend = if (use_cpp) "cpp" else "pure-R"),
    class = "morie_hawkes_fit")
}

#' @export
print.morie_hawkes_fit <- function(x, ...) {
  cat(sprintf("morie Hawkes fit  --  %s kernel, %s baseline\n",
              x$kernel, x$baseline))
  cat(sprintf("  events: %d   horizon: %.4g   backend: %s\n",
              x$n_events, x$end_time, x$backend))
  cat("  estimate:\n")
  for (nm in names(x$estimate)) {
    cat(sprintf("    %-8s % .6g\n", nm, x$estimate[[nm]]))
  }
  cat(sprintf("  baseline rate nu : %.6g\n", x$baseline_rate))
  cat(sprintf("  branching ratio  : %.6g%s\n", x$branching_ratio,
              if (x$branching_ratio >= 1) "  (NOT stationary: eta >= 1)"
              else ""))
  cat(sprintf("  logLik: %.6g   AIC: %.6g   converged: %s\n",
              x$loglik, x$aic, x$converged))
  cat(sprintf("  vs Poisson: logLik %.6g  (gain %+.4g)\n",
              x$loglik_poisson, x$loglik_gain))
  if (!isTRUE(x$self_excitation_detected)) {
    cat("  NOTE: ", x$note, "\n", sep = "")
  }
  invisible(x)
}
