# SPDX-License-Identifier: GPL-2.0-only

#' Condorcet winner detection (Armstrong Ch 2)
#'
#' A Condorcet winner beats every other alternative in pairwise
#' majority voting. Returns the index of the Condorcet winner, or -1
#' if none exists.
#'
#' @param preference_matrix n by n matrix where entry (i, j) = number of
#'   voters preferring i to j.
#' @return Named list with `winner` (1-based, or -1), `n_candidates`,
#'   `has_winner`, `method`.
#' @export
cndrc <- function(preference_matrix) {
  M <- as.matrix(preference_matrix)
  n <- nrow(M); winner <- -1L
  for (i in seq_len(n)) {
    beats_all <- TRUE
    for (j in seq_len(n)) {
      if (i != j && M[i, j] <= M[j, i]) { beats_all <- FALSE; break }
    }
    if (beats_all) { winner <- i; break }
  }
  list(winner = winner, n_candidates = n,
       has_winner = winner > 0L, method = "condorcet_winner")
}

#' @keywords internal
#' @rdname cndrc
#' @export
condorcet_winner <- cndrc
