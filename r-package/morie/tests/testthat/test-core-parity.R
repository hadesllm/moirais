# test-core-parity.R -- v0.9.1 Phase 3.
#
# The shared C++ numeric core (src/morie_core.hpp), bound here via
# Rcpp, must match base R. The Python package binds the SAME header
# via nanobind and is parity-checked against numpy on its side, so
# verifying the R side against base R anchors the cross-language
# guarantee: one compiled source of truth for both languages.

test_that("shared C++ core kernels match base R", {
  set.seed(20260516)
  x <- rnorm(500)
  y <- 0.6 * x + rnorm(500)

  expect_equal(morie:::morie_mean_cpp(x), mean(x), tolerance = 1e-12)
  expect_equal(morie:::morie_var_cpp(x, 1L), var(x), tolerance = 1e-10)
  expect_equal(morie:::morie_var_cpp(x, 0L),
               sum((x - mean(x))^2) / length(x), tolerance = 1e-10)
  expect_equal(morie:::morie_cor_pearson_cpp(x, y), cor(x, y),
               tolerance = 1e-9)
  expect_equal(morie:::morie_normal_pdf_cpp(x, 0.3, 1.7),
               dnorm(x, 0.3, 1.7), tolerance = 1e-12)
})

test_that("shared C++ core handles degenerate input", {
  expect_true(is.na(morie:::morie_mean_cpp(numeric(0))))
  expect_true(is.na(morie:::morie_var_cpp(c(5.0), 1L)))   # n - ddof <= 0
  expect_true(is.na(morie:::morie_cor_pearson_cpp(c(1.0), c(2.0))))  # n < 2
})
