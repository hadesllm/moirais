# Deterministic-seed plumbing tests for mhatf (multi-head attention).
#
# Verifies that the `deterministic_seed` argument added in morie v0.4.0:
#   * gives the same output on repeat calls with the same seed
#   * gives different output when the seed is changed
#   * leaves the default `deterministic_seed = NULL` path unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

test_that("mhatf deterministic_seed is reproducible", {
  skip_if_no_hash()
  x <- diag(4)
  r1 <- mhatf_multi_head_attention_full(x, num_heads = 2L, deterministic_seed = 42L)
  r2 <- mhatf_multi_head_attention_full(x, num_heads = 2L, deterministic_seed = 42L)
  r3 <- mhatf_multi_head_attention_full(x, num_heads = 2L, deterministic_seed = 999L)
  expect_equal(r1$output, r2$output)
  expect_false(isTRUE(all.equal(r1$output, r3$output)))
})

test_that("mhatf default (deterministic_seed = NULL) path is unchanged", {
  x <- diag(4)
  r1 <- mhatf_multi_head_attention_full(x, num_heads = 2L, seed = 0L)
  r2 <- mhatf_multi_head_attention_full(x, num_heads = 2L, seed = 0L)
  expect_equal(r1$output, r2$output)
})
