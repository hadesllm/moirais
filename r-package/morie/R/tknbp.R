# SPDX-License-Identifier: GPL-2.0-only

#' Byte-pair encoding tokenizer (Sennrich 2016)
#'
#' @param x Character vector or whitespace-delimited corpus.
#' @param num_merges Integer number of BPE merges (default 10).
#' @return Named list with merges, vocab, corpus, n_merges, n_vocab, method.
#' @keywords internal
bpe_tokenizer <- function(x, num_merges = 10L) {
  if (length(x) == 1L && is.character(x) && grepl("\\s", x))
    words <- strsplit(x, "\\s+")[[1L]]
  else
    words <- as.character(x)
  if (!length(words))
    return(list(merges = list(), vocab = character(0),
                n_merges = 0L, n_vocab = 0L, method = "BPE"))
  tab <- table(words)
  corpus <- lapply(names(tab), function(w)
    c(strsplit(w, "")[[1L]], "</w>"))
  freq <- as.integer(tab)
  merges <- list()
  for (m in seq_len(num_merges)) {
    pair_counts <- list()
    for (k in seq_along(corpus)) {
      sym <- corpus[[k]]; f <- freq[[k]]
      if (length(sym) < 2L) next
      for (i in seq_len(length(sym) - 1L)) {
        key <- paste(sym[i], sym[i + 1L], sep = "\x1f")
        pair_counts[[key]] <- (pair_counts[[key]] %||% 0L) + f
      }
    }
    if (!length(pair_counts)) break
    best_key <- names(which.max(unlist(pair_counts)))
    best <- strsplit(best_key, "\x1f", fixed = TRUE)[[1L]]
    merges[[length(merges) + 1L]] <- best
    # merge in corpus
    corpus <- lapply(corpus, function(sym) {
      if (length(sym) < 2L) return(sym)
      out <- character(0); i <- 1L
      while (i <= length(sym)) {
        if (i < length(sym) &&
            sym[i] == best[1L] && sym[i + 1L] == best[2L]) {
          out <- c(out, paste0(best[1L], best[2L])); i <- i + 2L
        } else { out <- c(out, sym[i]); i <- i + 1L }
      }
      out
    })
  }
  vocab <- unique(unlist(corpus))
  list(merges = merges, vocab = vocab, corpus = corpus,
       n_merges = length(merges), n_vocab = length(vocab),
       method = "BPE")
}
