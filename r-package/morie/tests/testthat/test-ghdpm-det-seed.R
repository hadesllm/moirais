# Deterministic-seed plumbing tests for ghdpm (DP mixture density).
#
# Verifies that the `deterministic_seed` argument added in morie v0.4.0:
#   * gives the same `estimate` / `k_post` on repeat calls with the same seed
#   * gives different numbers when the seed is changed
#   * leaves the default `deterministic_seed = NULL` path unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

xs_fixture <- function() {
  c(0.1, 0.4, -0.3, 0.7, 0.05, -0.9, 1.2, -0.4, 0.6, -0.1,
    0.3, -0.2, 0.5, -0.7, 0.0, 0.2, -0.1, 0.4, -0.5, 0.8)
}

test_that("ghdpm deterministic_seed is reproducible", {
  skip_if_no_hash()
  xs <- xs_fixture()
  r1 <- ghosal_dpmixture_density(xs, n_iter = 20, burn = 5, deterministic_seed = 42L)
  r2 <- ghosal_dpmixture_density(xs, n_iter = 20, burn = 5, deterministic_seed = 42L)
  r3 <- ghosal_dpmixture_density(xs, n_iter = 20, burn = 5, deterministic_seed = 999L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$k_post, r2$k_post)
  expect_false(isTRUE(all.equal(r1$estimate, r3$estimate)))
})

test_that("ghdpm default (deterministic_seed = NULL) path is unchanged", {
  xs <- xs_fixture()
  r1 <- ghosal_dpmixture_density(xs, n_iter = 20, burn = 5, seed = 42L)
  r2 <- ghosal_dpmixture_density(xs, n_iter = 20, burn = 5, seed = 42L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$k_post, r2$k_post)
})
