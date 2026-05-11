library(testthat)

# Shared test data
df100 <- data.frame(
  id    = 1:100,
  x     = rnorm(100),
  group = c(rep("A", 60), rep("B", 40)),
  size  = abs(rnorm(100, mean = 5))
)

# ── simple_random_sample ──────────────────────────────────────────────────────

test_that("simple_random_sample returns correct number of rows", {
  s <- simple_random_sample(df100, 20)
  expect_equal(nrow(s), 20)
})

test_that("simple_random_sample adds .weight column", {
  s <- simple_random_sample(df100, 20)
  expect_true(".weight" %in% names(s))
  expect_equal(unique(s$.weight), 100 / 20)
})

test_that("simple_random_sample WOR does not duplicate rows", {
  s <- simple_random_sample(df100, 50)
  expect_equal(length(unique(s$id)), 50)
})

test_that("simple_random_sample is reproducible with same seed", {
  s1 <- simple_random_sample(df100, 20, seed = 7)
  s2 <- simple_random_sample(df100, 20, seed = 7)
  expect_equal(s1$id, s2$id)
})

# ── stratified_sample ─────────────────────────────────────────────────────────

test_that("stratified_sample returns correct total rows with equal allocation", {
  s <- stratified_sample(df100, "group", n_per_stratum = 10)
  expect_equal(nrow(s), 20)
})

test_that("stratified_sample returns correct stratum sizes", {
  s <- stratified_sample(df100, "group", n_per_stratum = c(A = 12, B = 8))
  tab <- table(s$group)
  expect_equal(as.integer(tab["A"]), 12)
  expect_equal(as.integer(tab["B"]), 8)
})

test_that("stratified_sample proportional allocation returns ~n rows", {
  s <- stratified_sample(df100, "group", n_per_stratum = 30,
                          proportional = TRUE)
  expect_true(nrow(s) >= 28 && nrow(s) <= 32)
})

# ── cluster_sample ────────────────────────────────────────────────────────────

test_that("cluster_sample returns only rows from selected clusters", {
  df_c <- data.frame(cluster = rep(1:10, each = 10), x = rnorm(100))
  s <- cluster_sample(df_c, "cluster", n_clusters = 3)
  expect_equal(length(unique(s$cluster)), 3)
})

test_that("cluster_sample adds correct weight", {
  df_c <- data.frame(cluster = rep(1:10, each = 10), x = rnorm(100))
  s <- cluster_sample(df_c, "cluster", n_clusters = 4)
  expect_equal(unique(s$.weight), 10 / 4)
})

# ── pps_sample ────────────────────────────────────────────────────────────────

test_that("pps_sample returns n rows", {
  s <- pps_sample(df100, "size", n = 30)
  expect_equal(nrow(s), 30)
})

test_that("pps_sample adds .weight column", {
  s <- pps_sample(df100, "size", n = 30)
  expect_true(".weight" %in% names(s))
  expect_true(all(s$.weight > 0))
})

# ── bootstrap_sample ──────────────────────────────────────────────────────────

test_that("bootstrap_sample returns correct fields", {
  result <- bootstrap_sample(df100, statistic = function(d) mean(d$x),
                              n_bootstrap = 200L)
  expect_named(result, c("estimate", "se", "ci_lower", "ci_upper",
                          "distribution"))
  expect_length(result$distribution, 200L)
  expect_lt(result$ci_lower, result$ci_upper)
})

test_that("bootstrap SE is approximately correct for mean", {
  set.seed(1)
  df_b <- data.frame(x = rnorm(500))
  result <- bootstrap_sample(df_b, function(d) mean(d$x), n_bootstrap = 500L)
  # True SE = 1/sqrt(500) ≈ 0.0447
  expect_lt(abs(result$se - 1 / sqrt(500)), 0.02)
})

test_that("bootstrap_sample is reproducible with same seed", {
  stat_fn <- function(d) mean(d$x)
  r1 <- bootstrap_sample(df100, stat_fn, seed = 99L)
  r2 <- bootstrap_sample(df100, stat_fn, seed = 99L)
  expect_equal(r1$distribution, r2$distribution)
})

# ── jackknife_estimate ────────────────────────────────────────────────────────

test_that("jackknife_estimate returns correct fields", {
  result <- jackknife_estimate(df100, function(d) mean(d$x))
  expect_named(result, c("estimate", "se", "bias"))
  expect_type(result$se, "double")
})

test_that("jackknife SE is close to bootstrap SE for mean", {
  df_j <- data.frame(x = rnorm(100))
  jk <- jackknife_estimate(df_j, function(d) mean(d$x))
  # True SE ≈ 0.1; jackknife should be close
  expect_lt(abs(jk$se - 0.1), 0.03)
})

# ── effective_sample_size ─────────────────────────────────────────────────────

test_that("ESS equals n for equal weights", {
  w <- rep(1, 100)
  expect_equal(effective_sample_size(w), 100)
})

test_that("ESS < n for unequal weights", {
  w <- c(rep(1, 90), rep(10, 10))
  expect_lt(effective_sample_size(w), 100)
})

# ── design_effect ─────────────────────────────────────────────────────────────

test_that("DEFF = 1 for equal weights", {
  w <- rep(1, 50)
  expect_equal(design_effect(w), 1)
})

test_that("DEFF > 1 for unequal weights", {
  w <- c(rep(1, 80), rep(5, 20))
  expect_gt(design_effect(w), 1)
})

# ── calibration_weights ───────────────────────────────────────────────────────

test_that("calibration_weights match population marginals after raking", {
  df_c <- data.frame(gender = c(rep("M", 60), rep("F", 40)))
  pop_totals <- list(gender_M = 50, gender_F = 50)
  w <- calibration_weights(df_c, "gender", pop_totals)
  # After calibration, sum of weights for M and F should both equal 50
  expect_lt(abs(sum(w[df_c$gender == "M"]) - 50), 1)
  expect_lt(abs(sum(w[df_c$gender == "F"]) - 50), 1)
})
