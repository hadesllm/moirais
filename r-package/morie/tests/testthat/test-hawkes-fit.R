# SPDX-License-Identifier: AGPL-3.0-or-later

test_that("morie_hawkes_fit returns a fit for each kernel", {
  set.seed(42)
  ev <- cumsum(rexp(250, rate = 2))
  for (k in c("exponential", "weibull", "lomax", "gamma")) {
    fit <- morie_hawkes_fit(ev, kernel = k)
    expect_s3_class(fit, "morie_hawkes_fit")
    expect_true(is.finite(fit$loglik))
    expect_true(is.finite(fit$aic))
    expect_gt(fit$branching_ratio, 0)
    expect_lt(fit$branching_ratio, 1)
    expect_gt(fit$baseline_rate, 0)
    expect_identical(fit$n_events, 250L)
    expect_identical(length(fit$estimate),
                     length(morie:::.hawkes_param_names(k)))
  }
})

test_that("the C++ and pure-R Hawkes likelihoods agree", {
  skip_if_not(morie_fast_available(), "compiled core not available")
  set.seed(7)
  ev <- cumsum(rexp(200, rate = 2))
  end_time <- ev[200] + 1
  cases <- list(
    list(k = "exponential", th = c(-1, 0.4, 1.5)),
    list(k = "weibull",     th = c(-1, 0.4, 1.5, 2.0)),
    list(k = "lomax",       th = c(-1, 0.4, 2.5, 1.0)),
    list(k = "gamma",       th = c(-1, 0.4, 2.5, 1.0)))
  for (cs in cases) {
    a <- morie:::.hawkes_nll_cpp(cs$th, ev, end_time, cs$k)
    b <- morie:::.hawkes_nll_pureR(cs$th, ev, end_time, cs$k)
    expect_equal(a, b, tolerance = 1e-6)
  }
})

test_that("morie_hawkes_fit validates its inputs", {
  expect_error(morie_hawkes_fit(c(3, 1, 2)), "sorted")
  expect_error(morie_hawkes_fit(c(1)), "at least 2")
  expect_error(morie_hawkes_fit(c(1, 2), end_time = 1.5), "end_time")
})

test_that("morie_hawkes_fit reports the Poisson baseline and degeneracy", {
  set.seed(99)
  ev <- cumsum(rexp(300, rate = 3))
  fit <- morie_hawkes_fit(ev, kernel = "exponential")
  # the Poisson baseline is exact
  expect_equal(fit$loglik_poisson,
               morie:::.hawkes_loglik_poisson(300, fit$end_time))
  # the Hawkes family nests Poisson, so it can never do worse
  expect_gte(fit$loglik_gain, -1e-6)
  # the self-excitation flag is consistent with the fitted eta
  expect_identical(fit$self_excitation_detected,
                   fit$branching_ratio >= 1e-3)
  if (!fit$self_excitation_detected) {
    expect_true(nzchar(fit$note))
  }
})

test_that("the unconstrained reparameterisation round-trips", {
  for (theta in list(c(-1, 0.4, 1.5), c(0.5, 0.8, 2.0, 1.3))) {
    phi <- morie:::.hawkes_to_phi(theta)
    expect_equal(morie:::.hawkes_to_theta(phi), theta, tolerance = 1e-12)
  }
})
