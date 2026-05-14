# SPDX-License-Identifier: MIT OR Apache-2.0

#' Design-of-Experiments toolkit (R parity)
#'
#' R parity of \code{morie.mrm_doe}.  Closes the Chapter-3/4/5 coverage
#' gap from designexptr.org.
#'
#' @references
#' Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). Statistics for
#'   Experimenters (2nd ed.). Wiley.
#' Cochran, W. G., & Cox, G. M. (1957). Experimental Designs (2nd ed.). Wiley.
#' Montgomery, D. C. (2017). Design and Analysis of Experiments (9th ed.).
#' Box, G. E. P., & Wilson, K. B. (1951). On the experimental attainment of
#'   optimum conditions. JRSS-B, 13(1), 1-45.
#' Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
#'
#' @name mrm_doe
NULL


#' One-way ANOVA with pairwise Bonferroni-adjusted t-tests
#'
#' @param data data.frame.
#' @param response_col Response column name.
#' @param group_col Group column name.
#' @param alpha Family-wise error rate (default 0.05).
#' @return Named list with f_statistic, p_value, n_groups, n_pairs,
#'   alpha, alpha_per_pair, pairs (data.frame), interpretation.
#' @export
mrm_anova_bonferroni <- function(data, response_col, group_col, alpha = 0.05) {
  d <- data[, c(response_col, group_col)]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  d[[group_col]] <- factor(d[[group_col]])
  fit <- stats::aov(stats::as.formula(paste(response_col, "~", group_col)),
                     data = d)
  summ <- summary(fit)[[1]]
  f <- summ[["F value"]][1]; p_anova <- summ[["Pr(>F)"]][1]
  groups <- levels(d[[group_col]])
  pairs_list <- list()
  k <- 0L
  for (i in seq_along(groups)) {
    for (j in seq_along(groups)) {
      if (j <= i) next
      a <- d[d[[group_col]] == groups[i], response_col]
      b <- d[d[[group_col]] == groups[j], response_col]
      tt <- stats::t.test(a, b, var.equal = FALSE)
      k <- k + 1L
      pairs_list[[k]] <- data.frame(
        group_a = groups[i], group_b = groups[j],
        diff = round(mean(a) - mean(b), 4),
        t = round(as.numeric(tt$statistic), 4),
        p_raw = tt$p.value,
        stringsAsFactors = FALSE
      )
    }
  }
  pairs <- do.call(rbind, pairs_list)
  n_pairs <- nrow(pairs)
  pairs$p_bonferroni <- pmin(1.0, pairs$p_raw * n_pairs)
  pairs$significant <- pairs$p_bonferroni < alpha
  list(
    f_statistic = round(as.numeric(f), 4),
    p_value = as.numeric(p_anova),
    n_groups = length(groups),
    n_pairs = n_pairs,
    alpha = alpha,
    alpha_per_pair = alpha / max(n_pairs, 1L),
    pairs = pairs,
    interpretation = sprintf(
      "F = %.3f, p = %.3g; %d/%d pairs significant after Bonferroni at alpha = %g.",
      f, p_anova, sum(pairs$significant), n_pairs, alpha
    )
  )
}


#' Randomised complete block design (RCBD) two-way ANOVA
#'
#' Model: y_ij = mu + tau_i (treatment) + beta_j (block) + eps_ij
#' Returns Type-I ANOVA: block enters first, then treatment.
#'
#' @param data data.frame.
#' @param response_col,treatment_col,block_col Column names.
#' @return Named list with anova (data.frame), n, n_treatments,
#'   n_blocks, interpretation.
#' @export
mrm_rcbd <- function(data, response_col, treatment_col, block_col) {
  d <- data[, c(response_col, treatment_col, block_col)]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  d[[treatment_col]] <- factor(d[[treatment_col]])
  d[[block_col]]    <- factor(d[[block_col]])
  fit <- stats::aov(stats::as.formula(paste(
    response_col, "~", block_col, "+", treatment_col)),
    data = d)
  summ <- as.data.frame(summary(fit)[[1]])
  summ$source <- trimws(rownames(summ)); rownames(summ) <- NULL
  list(
    anova = summ,
    n = nrow(d),
    n_treatments = nlevels(d[[treatment_col]]),
    n_blocks = nlevels(d[[block_col]]),
    interpretation = sprintf(
      "RCBD on n=%d, %d treatments, %d blocks; Treatment p = %.3g.",
      nrow(d), nlevels(d[[treatment_col]]), nlevels(d[[block_col]]),
      summ[summ$source == treatment_col, "Pr(>F)"]
    )
  )
}


#' Latin-square three-way ANOVA (row, col, treatment)
#'
#' @param data data.frame.
#' @param response_col,row_col,col_col,treatment_col Column names.
#' @return Named list with anova, n, k, interpretation.
#' @export
mrm_latin_square <- function(data, response_col, row_col, col_col,
                              treatment_col) {
  d <- data[, c(response_col, row_col, col_col, treatment_col)]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  for (c in c(row_col, col_col, treatment_col)) d[[c]] <- factor(d[[c]])
  fit <- stats::aov(stats::as.formula(paste(
    response_col, "~", row_col, "+", col_col, "+", treatment_col)),
    data = d)
  summ <- as.data.frame(summary(fit)[[1]])
  summ$source <- trimws(rownames(summ)); rownames(summ) <- NULL
  list(
    anova = summ,
    n = nrow(d),
    k = nlevels(d[[treatment_col]]),
    interpretation = sprintf(
      "%dx%d Latin square on n=%d; Treatment p = %.3g.",
      nlevels(d[[treatment_col]]), nlevels(d[[treatment_col]]), nrow(d),
      summ[summ$source == treatment_col, "Pr(>F)"]
    )
  )
}


#' Graeco-Latin square four-way ANOVA (row, col, Latin, Greek)
#'
#' @param data data.frame.
#' @param response_col,row_col,col_col,latin_col,greek_col Column names.
#' @return Named list with anova, n, interpretation.
#' @export
mrm_graeco_latin <- function(data, response_col, row_col, col_col,
                              latin_col, greek_col) {
  d <- data[, c(response_col, row_col, col_col, latin_col, greek_col)]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  for (c in c(row_col, col_col, latin_col, greek_col)) d[[c]] <- factor(d[[c]])
  fit <- stats::aov(stats::as.formula(paste(
    response_col, "~", row_col, "+", col_col, "+",
    latin_col, "+", greek_col)),
    data = d)
  summ <- as.data.frame(summary(fit)[[1]])
  summ$source <- trimws(rownames(summ)); rownames(summ) <- NULL
  list(
    anova = summ,
    n = nrow(d),
    interpretation = sprintf(
      "Graeco-Latin square on n=%d; Latin p = %.3g, Greek p = %.3g.",
      nrow(d),
      summ[summ$source == latin_col, "Pr(>F)"],
      summ[summ$source == greek_col, "Pr(>F)"]
    )
  )
}


#' Fractional 2^(k-p) factorial: main effects + alias structure
#'
#' Factor columns assumed +/-1.
#'
#' @param data data.frame.
#' @param response_col Response column.
#' @param factor_cols Character vector of factor columns (each coded -1 or +1).
#' @param generator Optional generator string "X=YZ,..." for aliasing.
#' @return Named list with main_effects, alias_structure, n, k,
#'   interpretation.
#' @export
mrm_fractional_factorial <- function(data, response_col, factor_cols,
                                      generator = NULL) {
  d <- data[, c(response_col, factor_cols), drop = FALSE]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  y <- d[[response_col]]
  main <- vapply(factor_cols, function(f) {
    mean(y[d[[f]] == 1]) - mean(y[d[[f]] == -1])
  }, numeric(1))
  names(main) <- factor_cols
  aliases <- list()
  if (!is.null(generator) && nzchar(generator)) {
    for (clause in strsplit(generator, ",")[[1]]) {
      parts <- strsplit(trimws(clause), "=")[[1]]
      aliases[[trimws(parts[1])]] <- trimws(parts[2])
    }
  }
  big <- names(main)[which.max(abs(main))]
  list(
    main_effects = as.list(round(main, 6)),
    alias_structure = aliases,
    n = nrow(d), k = length(factor_cols),
    interpretation = sprintf(
      "2^%d fractional on n=%d. Largest main effect: %s = %.3f",
      length(factor_cols), nrow(d), big, main[[big]]
    )
  )
}


#' Second-order response-surface fit (Box-Wilson 1951)
#'
#' Fits y = b0 + sum b_i x_i + sum b_ii x_i^2 + sum b_ij x_i x_j and
#' returns the stationary point if the quadratic matrix B is invertible.
#'
#' @param data data.frame.
#' @param response_col Response column.
#' @param factor_cols Character vector of factor columns.
#' @return Named list with coefficients, stationary_point,
#'   stationary_y, stationary_nature, eigenvalues, n, interpretation.
#' @export
mrm_response_surface <- function(data, response_col, factor_cols) {
  d <- data[, c(response_col, factor_cols), drop = FALSE]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  y <- d[[response_col]]
  X <- as.matrix(d[, factor_cols, drop = FALSE])
  k <- length(factor_cols)

  cols <- list(intercept = rep(1, nrow(X)))
  for (i in seq_len(k)) cols[[factor_cols[i]]] <- X[, i]
  for (i in seq_len(k)) cols[[paste0(factor_cols[i], "^2")]] <- X[, i] ^ 2
  for (i in seq_len(k - 1L)) {
    for (j in seq.int(i + 1L, k)) {
      cols[[paste0(factor_cols[i], ":", factor_cols[j])]] <- X[, i] * X[, j]
    }
  }
  D <- do.call(cbind, cols)
  beta <- as.numeric(stats::lm.fit(D, y)$coefficients)
  names(beta) <- colnames(D)
  b <- beta[2:(1 + k)]
  B <- matrix(0, k, k)
  for (i in seq_len(k)) B[i, i] <- beta[1 + k + i]
  idx <- 1L + 2L * k
  for (i in seq_len(k - 1L)) {
    for (j in seq.int(i + 1L, k)) {
      idx <- idx + 1L
      B[i, j] <- beta[idx] / 2
      B[j, i] <- B[i, j]
    }
  }
  x_star <- tryCatch(-0.5 * solve(B, b), error = function(e) NULL)
  if (!is.null(x_star)) {
    y_star <- as.numeric(beta[1] + b %*% x_star + t(x_star) %*% B %*% x_star)
  } else {
    y_star <- NA_real_
  }
  eigvals <- eigen(B, symmetric = TRUE, only.values = TRUE)$values
  nature <- if (all(eigvals < 0)) "maximum"
            else if (all(eigvals > 0)) "minimum"
            else "saddle"
  list(
    coefficients = as.list(round(beta, 6)),
    stationary_point = if (is.null(x_star)) NULL else
      setNames(as.list(round(as.numeric(x_star), 4)), factor_cols),
    stationary_y = if (is.null(x_star)) NULL else round(y_star, 4),
    stationary_nature = nature,
    eigenvalues = round(eigvals, 4),
    n = nrow(d),
    interpretation = sprintf(
      "Second-order RSM on n=%d, k=%d; stationary point is a %s.",
      nrow(d), k, nature
    )
  )
}


#' Power of one-way ANOVA given Cohen's f
#'
#' @param k_groups Number of groups.
#' @param n_per_group Per-group sample size.
#' @param effect_size_f Cohen's f.
#' @param alpha Type-I error (default 0.05).
#' @return Named list with k_groups, n_per_group, N_total, effect_size_f,
#'   alpha, df1, df2, noncentrality, F_critical, power, interpretation.
#' @export
mrm_anova_power <- function(k_groups, n_per_group, effect_size_f,
                             alpha = 0.05) {
  df1 <- k_groups - 1L
  N <- k_groups * n_per_group
  df2 <- N - k_groups
  ncp <- N * effect_size_f ^ 2
  F_crit <- stats::qf(1 - alpha, df1, df2)
  power <- 1 - stats::pf(F_crit, df1, df2, ncp = ncp)
  list(
    k_groups = as.integer(k_groups),
    n_per_group = as.integer(n_per_group),
    N_total = as.integer(N),
    effect_size_f = effect_size_f,
    alpha = alpha,
    df1 = as.integer(df1), df2 = as.integer(df2),
    noncentrality = round(ncp, 4),
    F_critical = round(F_crit, 4),
    power = round(power, 4),
    interpretation = sprintf(
      "Power = %.3f for k=%d, n_per_group=%d, Cohen's f = %g, alpha = %g.",
      power, k_groups, n_per_group, effect_size_f, alpha
    )
  )
}


#' Empirical Monte-Carlo power
#'
#' @param simulator A function(seed) returning a p-value.
#' @param n_sims Number of simulated datasets.
#' @param alpha Type-I error level.
#' @param seed Seed for outer RNG.
#' @return Named list with empirical_power, se, ci95 bounds.
#' @export
mrm_mc_power <- function(simulator, n_sims = 1000L, alpha = 0.05, seed = 42L) {
  set.seed(seed)
  p_values <- vapply(seq_len(n_sims),
                      function(i) simulator(sample.int(.Machine$integer.max, 1)),
                      numeric(1))
  pwr <- mean(p_values < alpha)
  se <- sqrt(pwr * (1 - pwr) / n_sims)
  list(
    n_sims = as.integer(n_sims),
    alpha = alpha,
    empirical_power = round(pwr, 4),
    se = round(se, 4),
    ci95_lower = round(max(0, pwr - 1.96 * se), 4),
    ci95_upper = round(min(1, pwr + 1.96 * se), 4),
    interpretation = sprintf(
      "Empirical power = %.3f (SE %.3f) over %d sims at alpha=%g.",
      pwr, se, n_sims, alpha
    )
  )
}


#' Block-permutation test for treatment effect
#'
#' Permutes treatment labels within each block.
#'
#' @param data data.frame.
#' @param response_col,treatment_col,block_col Column names.
#' @param n_perm Number of permutations.
#' @param seed RNG seed.
#' @return Named list with observed_statistic, n_perm, p_value,
#'   interpretation.
#' @export
mrm_perm_block <- function(data, response_col, treatment_col, block_col,
                            n_perm = 1000L, seed = 42L) {
  d <- data[, c(response_col, treatment_col, block_col)]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  trt <- d[[treatment_col]]
  blk <- d[[block_col]]
  y <- d[[response_col]]
  diff_obs <- function(tcol) {
    m <- tapply(y, tcol, mean)
    as.numeric(diff(m))[length(m) - 1]
  }
  obs <- diff_obs(trt)
  set.seed(seed)
  perm_stats <- numeric(n_perm)
  blocks <- unique(blk)
  for (k in seq_len(n_perm)) {
    permuted <- trt
    for (b in blocks) {
      idx <- which(blk == b)
      permuted[idx] <- sample(trt[idx])
    }
    perm_stats[k] <- diff_obs(permuted)
  }
  p <- mean(abs(perm_stats) >= abs(obs))
  list(
    observed_statistic = round(obs, 6),
    n_perm = as.integer(n_perm),
    p_value = p,
    interpretation = sprintf(
      "Block-permutation test: observed diff = %.4f, two-sided p = %.3g over %d block-shuffles.",
      obs, p, n_perm
    )
  )
}


#' Generate a random k x k Latin square
#'
#' Builds the cyclic Latin square then permutes rows, columns,
#' and symbols.  Uniform over a subset of Latin squares (not all).
#'
#' @param k Side length.
#' @param seed RNG seed.
#' @return A k x k matrix with row names R1..Rk and column names C1..Ck.
#' @export
mrm_random_latin <- function(k, seed = 42L) {
  set.seed(seed)
  base <- matrix(0L, k, k)
  for (i in seq_len(k)) for (j in seq_len(k)) {
    base[i, j] <- ((i + j - 2L) %% k)
  }
  row_perm <- sample.int(k)
  col_perm <- sample.int(k)
  sym_perm <- sample.int(k) - 1L
  sq <- base[row_perm, col_perm]
  sq[] <- sym_perm[sq + 1L]
  dimnames(sq) <- list(paste0("R", seq_len(k)), paste0("C", seq_len(k)))
  sq
}
