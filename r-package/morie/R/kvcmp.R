# SPDX-License-Identifier: GPL-2.0-only

#' KV-cache append (Pope 2022)
#'
#' @param K_cache Existing key cache matrix (or NULL).
#' @param V_cache Existing value cache matrix (or NULL).
#' @param k_new New key rows to append.
#' @param v_new New value rows to append.
#' @param max_len Optional integer cap on sequence length.
#' @return Named list with K, V, T, max_len, method.
#' @keywords internal
kv_cache_management <- function(K_cache, V_cache, k_new, v_new,
                                max_len = NULL) {
  if (is.null(K_cache)) {
    K_new <- as.matrix(k_new); V_new <- as.matrix(v_new)
  } else {
    K_new <- rbind(as.matrix(K_cache), as.matrix(k_new))
    V_new <- rbind(as.matrix(V_cache), as.matrix(v_new))
  }
  if (!is.null(max_len) && nrow(K_new) > max_len) {
    K_new <- K_new[(nrow(K_new) - max_len + 1L):nrow(K_new), , drop = FALSE]
    V_new <- V_new[(nrow(V_new) - max_len + 1L):nrow(V_new), , drop = FALSE]
  }
  list(K = K_new, V = V_new, T = nrow(K_new), max_len = max_len,
       method = "kv-cache-append")
}
