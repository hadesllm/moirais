# SPDX-License-Identifier: AGPL-3.0-or-later

#' Internal helpers shared across the LLM-architecture suite.
#' Not exported; consumed by llm_arch callables only.
#' @keywords internal
#' @name llm_arch_helpers
NULL

.softmax_last <- function(x) {
  # softmax along the last axis of an array
  d <- dim(x); nd <- length(d)
  if (is.null(d) || nd == 1L) {
    x <- x - max(x)
    e <- exp(x)
    return(e / sum(e))
  }
  apply(x, seq_len(nd - 1L), function(v) {
    v <- v - max(v); e <- exp(v); e / sum(e)
  }) -> out
  # apply collapses last axis to first; transpose back
  aperm(out, c(seq.int(2L, nd), 1L))
}

`%||%` <- function(a, b) if (is.null(a)) b else a
