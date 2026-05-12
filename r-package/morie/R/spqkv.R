# SPDX-License-Identifier: GPL-2.0-only

#' Child-2019 sparse attention mask
#'
#' @param x Sequence length (integer) or tensor whose second-to-last
#'   dim is the sequence length.
#' @param window Integer local window radius (default 4).
#' @param stride Integer strided-attention period (default 8).
#' @param n_random Integer per-row random links (default 0).
#' @param seed Integer RNG seed (default 0).
#' @return Named list with tensor (additive mask), boolean, density, method.
#' @keywords internal
sparse_attention <- function(x, window = 4L, stride = 8L,
                              n_random = 0L, seed = 0L) {
  N <- if (length(x) == 1L && is.numeric(x)) as.integer(x)
       else if (!is.null(dim(x))) dim(x)[length(dim(x)) - 1L]
       else length(x)
  set.seed(seed)
  M <- matrix(FALSE, N, N)
  for (i in seq_len(N)) {
    lo <- max(1L, i - window); hi <- min(N, i + window)
    M[i, lo:hi] <- TRUE
    M[i, seq.int(1L, N, by = stride)] <- TRUE
    if (n_random > 0L) {
      picks <- sample.int(N, size = min(n_random, N))
      M[i, picks] <- TRUE
    }
  }
  additive <- ifelse(M, 0, -Inf)
  density <- sum(M) / (N * N)
  list(tensor = additive, boolean = M, density = density,
       method = "sparse-attention")
}
