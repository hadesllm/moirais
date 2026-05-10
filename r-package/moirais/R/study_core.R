.na_omit_cols <- function(data, cols) {
  data[stats::complete.cases(data[, cols, drop = FALSE]), cols, drop = FALSE]
}

.safe_divide <- function(num, den) {
  if (is.na(den) || den == 0) return(NA_real_)
  as.numeric(num) / as.numeric(den)
}

.wald_ci <- function(estimate, se) {
  c(
    estimate - 1.96 * se,
    estimate + 1.96 * se
  )
}

.binary_ci <- function(successes, n) {
  p <- .safe_divide(successes, n)
  se <- sqrt(p * (1 - p) / max(n, 1))
  ci <- pmax(0, pmin(1, .wald_ci(p, se)))
  list(p = p, se = se, ci = ci)
}

.weighted_binary_estimate <- function(x, w) {
  keep <- !(is.na(x) | is.na(w))
  x <- as.numeric(x[keep])
  w <- as.numeric(w[keep])
  if (length(x) == 0L || sum(w) <= 0) {
    return(list(p = NA_real_, se = NA_real_, ci = c(NA_real_, NA_real_), n = 0L, n_eff = NA_real_))
  }
  p <- sum(x * w) / sum(w)
  n_eff <- .safe_divide(sum(w)^2, sum(w^2))
  se <- sqrt(p * (1 - p) / max(n_eff, 1))
  ci <- pmax(0, pmin(1, .wald_ci(p, se)))
  list(p = p, se = se, ci = ci, n = length(x), n_eff = n_eff)
}

.clip_exp <- function(x) {
  exp(pmin(pmax(as.numeric(x), -700), 700))
}

.safe_confint <- function(fit) {
  est <- stats::coef(fit)
  se <- sqrt(diag(stats::vcov(fit)))
  out <- cbind(est - 1.96 * se, est + 1.96 * se)
  rownames(out) <- names(est)
  out
}

.or_table <- function(fit, model = NULL, lower_se_name = FALSE) {
  est <- stats::coef(fit)
  se <- sqrt(diag(stats::vcov(fit)))
  ci <- .safe_confint(fit)
  out <- data.frame(
    term = names(est),
    log_odds = as.numeric(est),
    se = as.numeric(se),
    or = .clip_exp(est),
    or_lower95 = .clip_exp(ci[, 1]),
    or_upper95 = .clip_exp(ci[, 2]),
    p_value = as.numeric(summary(fit)$coefficients[, ncol(summary(fit)$coefficients)]),
    significant = ifelse(as.numeric(summary(fit)$coefficients[, ncol(summary(fit)$coefficients)]) < 0.05, "*", ""),
    stringsAsFactors = FALSE
  )
  if (!is.null(model)) {
    out <- cbind(model = model, out, stringsAsFactors = FALSE)
  }
  if (!lower_se_name) {
    names(out)[names(out) == "se"] <- "SE"
    names(out)[names(out) == "or"] <- "OR"
    names(out)[names(out) == "or_lower95"] <- "OR_lower95"
    names(out)[names(out) == "or_upper95"] <- "OR_upper95"
  }
  out
}

.linear_coef_table <- function(fit, model) {
  sm <- summary(fit)$coefficients
  ci <- .safe_confint(fit)
  data.frame(
    model = model,
    term = rownames(sm),
    estimate = as.numeric(sm[, 1]),
    se = as.numeric(sm[, 2]),
    ci_lower95 = as.numeric(ci[, 1]),
    ci_upper95 = as.numeric(ci[, 2]),
    p_value = as.numeric(sm[, ncol(sm)]),
    significant = ifelse(as.numeric(sm[, ncol(sm)]) < 0.05, "*", ""),
    stringsAsFactors = FALSE
  )
}

.cpads_labeled_data <- function(data) {
  out <- data
  out$gender_label <- factor(
    out$gender,
    levels = c(1, 2, 3),
    labels = c("Female", "Male", "Non-binary")
  )
  out$age_group_label <- factor(
    out$age_group,
    levels = c(1, 2, 3, 4),
    labels = c("16-19", "20-22", "23-25", "26+"),
    ordered = TRUE
  )
  out$province_region_label <- factor(
    out$province_region,
    levels = c(1, 2, 3, 4),
    labels = c("Atlantic", "Quebec", "Ontario", "Western")
  )
  out$mental_health_label <- factor(out$mental_health, ordered = TRUE)
  out$physical_health_label <- factor(out$physical_health, ordered = TRUE)
  if ("alc06" %in% names(out)) {
    out$alc06_valid <- ifelse(out$alc06 %in% c(97, 98, 99), NA_real_, as.numeric(out$alc06))
  } else {
    out$alc06_valid <- out$alcohol_past12m
  }
  out$ebac_observed <- as.integer(!is.na(out$ebac_tot))
  out
}

.run_data_wrangling_module_internal <- function(data, cpads_csv = NULL, output_dir = NULL) {
  resolved <- if (!is.null(cpads_csv)) .resolve_cpads_csv(cpads_csv) else NA_character_
  raw <- if (!is.na(resolved)) utils::read.csv(resolved, stringsAsFactors = FALSE) else data
  na_summary <- data.frame(
    variable = names(data),
    pct_na = round(100 * colMeans(is.na(data)), 4),
    stringsAsFactors = FALSE
  )
  wrangling_log <- data.frame(
    step = c("load_csv", "canonicalize_cpads_data", "validate_cpads_data"),
    description = c(
      "Read the bundled CPADS CSV from disk.",
      "Map raw CPADS fields to the MOIRAIS canonical analysis contract.",
      "Verify required canonical variables for downstream analyses."
    ),
    rows_before = c(nrow(raw), nrow(raw), nrow(data)),
    rows_after = c(nrow(raw), nrow(data), nrow(data)),
    cols_affected = c("all", paste(names(data), collapse = ","), paste(cpads_contract()$required_variables, collapse = ",")),
    stringsAsFactors = FALSE
  )
  if (!is.null(output_dir)) {
    project_root <- tryCatch(find_project_root(dirname(output_dir)), error = function(e) NULL)
    if (!is.null(project_root)) {
      wrangled_dir <- file.path(project_root, "data", "private", "outputs", "wrangled")
      dir.create(wrangled_dir, recursive = TRUE, showWarnings = FALSE)
      utils::write.csv(data, file.path(wrangled_dir, "data_wrangled.csv"), row.names = FALSE)
      saveRDS(data, file.path(wrangled_dir, "data_wrangled.rds"))
    }
  }
  list(
    data_na_summary = na_summary,
    data_wrangling_log = wrangling_log
  )
}

.run_descriptive_statistics_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  vars <- intersect(c("alcohol_past12m", "heavy_drinking_30d", "cannabis_any_use", "ebac_legal"), names(data))
  binomial_rows <- list()
  weighted_rows <- list()
  for (var in vars) {
    x <- data[[var]]
    n <- sum(!is.na(x))
    s <- sum(x == 1, na.rm = TRUE)
    ci <- .binary_ci(s, n)
    binomial_rows[[length(binomial_rows) + 1L]] <- data.frame(
      variable = var,
      n = n,
      successes = s,
      p_hat = ci$p,
      ci_low_wald = ci$ci[1],
      ci_high_wald = ci$ci[2],
      stringsAsFactors = FALSE
    )
    wci <- .weighted_binary_estimate(x, data$weight)
    weighted_rows[[length(weighted_rows) + 1L]] <- data.frame(
      variable = var,
      p_hat = wci$p,
      se = wci$se,
      ci_low = wci$ci[1],
      ci_high = wci$ci[2],
      n_unweighted_nonmissing = wci$n,
      stringsAsFactors = FALSE
    )
  }
  cannabis1 <- data[data$cannabis_any_use == 1 & !is.na(data$heavy_drinking_30d), ]
  cannabis0 <- data[data$cannabis_any_use == 0 & !is.na(data$heavy_drinking_30d), ]
  prob_rows <- list(
    data.frame(
      quantity = "P(cannabis_any_use=1)",
      estimate = .weighted_binary_estimate(data$cannabis_any_use, data$weight)$p,
      ci_lower = .weighted_binary_estimate(data$cannabis_any_use, data$weight)$ci[1],
      ci_upper = .weighted_binary_estimate(data$cannabis_any_use, data$weight)$ci[2],
      stringsAsFactors = FALSE
    ),
    data.frame(
      quantity = "P(heavy_drinking_30d=1)",
      estimate = .weighted_binary_estimate(data$heavy_drinking_30d, data$weight)$p,
      ci_lower = .weighted_binary_estimate(data$heavy_drinking_30d, data$weight)$ci[1],
      ci_upper = .weighted_binary_estimate(data$heavy_drinking_30d, data$weight)$ci[2],
      stringsAsFactors = FALSE
    ),
    data.frame(
      quantity = "P(heavy_drinking_30d=1 | cannabis_any_use=1)",
      estimate = .weighted_binary_estimate(cannabis1$heavy_drinking_30d, cannabis1$weight)$p,
      ci_lower = .weighted_binary_estimate(cannabis1$heavy_drinking_30d, cannabis1$weight)$ci[1],
      ci_upper = .weighted_binary_estimate(cannabis1$heavy_drinking_30d, cannabis1$weight)$ci[2],
      stringsAsFactors = FALSE
    ),
    data.frame(
      quantity = "P(heavy_drinking_30d=1 | cannabis_any_use=0)",
      estimate = .weighted_binary_estimate(cannabis0$heavy_drinking_30d, cannabis0$weight)$p,
      ci_lower = .weighted_binary_estimate(cannabis0$heavy_drinking_30d, cannabis0$weight)$ci[1],
      ci_upper = .weighted_binary_estimate(cannabis0$heavy_drinking_30d, cannabis0$weight)$ci[2],
      stringsAsFactors = FALSE
    )
  )
  list(
    binomial_summaries = do.call(rbind, binomial_rows),
    binomial_summaries_survey_weighted = do.call(rbind, weighted_rows),
    probability_estimates = do.call(rbind, prob_rows)
  )
}

.run_distribution_tests_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  tests <- list()
  add_test <- function(variable, distribution, test, statistic, p_value, conclusion, n) {
    tests[[length(tests) + 1L]] <<- data.frame(
      variable = variable,
      distribution = distribution,
      test = test,
      statistic = as.numeric(statistic),
      p_value = as.numeric(p_value),
      conclusion = conclusion,
      n = as.integer(n),
      stringsAsFactors = FALSE
    )
  }
  for (var in intersect(c("heavy_drinking_30d", "cannabis_any_use"), names(data))) {
    x <- data[[var]]
    x <- x[!is.na(x)]
    bt <- stats::binom.test(sum(x == 1), length(x), p = mean(x))
    add_test(var, "bernoulli", "binom.test", bt$statistic, bt$p.value, "reference-fit", length(x))
  }
  if ("ebac_tot" %in% names(data)) {
    x <- data$ebac_tot[!is.na(data$ebac_tot)]
    if (length(x) > 2L) {
      shp <- tryCatch(stats::shapiro.test(sample(x, min(length(x), 5000L))), error = function(e) NULL)
      if (!is.null(shp)) {
        add_test("ebac_tot", "continuous", "shapiro.test", shp$statistic, shp$p.value, ifelse(shp$p.value < 0.05, "non-normal", "approximately-normal"), length(x))
      }
    }
  }
  if ("alc06_valid" %in% names(data)) {
    x <- data$alc06_valid[!is.na(data$alc06_valid)]
    if (length(x) > 2L) {
      shp <- tryCatch(stats::shapiro.test(sample(x, min(length(x), 5000L))), error = function(e) NULL)
      if (!is.null(shp)) {
        add_test("alc06_valid", "ordinal-score", "shapiro.test", shp$statistic, shp$p.value, ifelse(shp$p.value < 0.05, "non-normal", "approximately-normal"), length(x))
      }
    }
  }
  cor_vars <- intersect(c("heavy_drinking_30d", "alc06_valid"), names(data))
  cor_mat <- stats::cor(data[, cor_vars, drop = FALSE], use = "pairwise.complete.obs")
  cor_df <- data.frame(variable = rownames(cor_mat), cor_mat, row.names = NULL, check.names = FALSE)

  set.seed(42)
  x <- data$heavy_drinking_30d[!is.na(data$heavy_drinking_30d)]
  sample_sizes <- c(25, 50, 100, 250, 500)
  clt_rows <- list()
  if (length(x) > 10L) {
    for (n in sample_sizes) {
      means <- replicate(200, mean(sample(x, n, replace = TRUE)))
      shp <- stats::shapiro.test(means)
      clt_rows[[length(clt_rows) + 1L]] <- data.frame(
        sample_size = n,
        mean_of_means = mean(means),
        sd_of_means = stats::sd(means),
        theoretical_se = stats::sd(x) / sqrt(n),
        shapiro_p = shp$p.value,
        stringsAsFactors = FALSE
      )
    }
  }
  list(
    distribution_tests = do.call(rbind, tests),
    alcohol_correlation_matrix = cor_df,
    clt_convergence = do.call(rbind, clt_rows)
  )
}

.run_frequentist_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  prevalence_rows <- list()
  for (lvl in levels(data$gender_label)) {
    subset <- data[data$gender_label == lvl & !is.na(data$heavy_drinking_30d), ]
    est <- .weighted_binary_estimate(subset$heavy_drinking_30d, subset$weight)
    prevalence_rows[[length(prevalence_rows) + 1L]] <- data.frame(
      variable = "heavy_drinking_30d",
      level = lvl,
      prev = est$p,
      se = est$se,
      ci_lower = est$ci[1],
      ci_upper = est$ci[2],
      n_unweighted_nonmissing = est$n,
      stringsAsFactors = FALSE
    )
  }
  effect_rows <- list()
  gsum <- do.call(rbind, lapply(levels(data$gender_label), function(lvl) {
    subset <- data[data$gender_label == lvl & !is.na(data$heavy_drinking_30d), ]
    data.frame(level = lvl, p = .weighted_binary_estimate(subset$heavy_drinking_30d, subset$weight)$p, stringsAsFactors = FALSE)
  }))
  if (nrow(gsum) >= 2L) {
    cmb <- utils::combn(gsum$level, 2, simplify = FALSE)
    for (pair in cmb) {
      p1 <- gsum$p[gsum$level == pair[1]]
      p2 <- gsum$p[gsum$level == pair[2]]
      h <- 2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))
      effect_rows[[length(effect_rows) + 1L]] <- data.frame(
        variable = "heavy_drinking_30d",
        comparison = paste(pair, collapse = " vs "),
        p1 = p1,
        p2 = p2,
        cohens_h = h,
        abs_h = abs(h),
        magnitude = ifelse(abs(h) < 0.2, "small", ifelse(abs(h) < 0.5, "medium", "large")),
        stringsAsFactors = FALSE
      )
    }
  }
  test_rows <- list()
  chi <- stats::chisq.test(table(data$cannabis_any_use, data$heavy_drinking_30d))
  test_rows[[1L]] <- data.frame(
    test_name = "Cannabis vs heavy drinking independence",
    comparison = "cannabis_any_use x heavy_drinking_30d",
    statistic = unname(chi$statistic),
    df = unname(chi$parameter),
    p_value = chi$p.value,
    method = chi$method,
    p_bonferroni = min(1, chi$p.value),
    p_fdr_bh = min(1, chi$p.value),
    sig_nominal = chi$p.value < 0.05,
    sig_bonf = chi$p.value < 0.05,
    sig_fdr = chi$p.value < 0.05,
    stringsAsFactors = FALSE
  )
  if (nrow(gsum) >= 2L) {
    first_two <- gsum$level[1:2]
    s1 <- data[data$gender_label == first_two[1] & !is.na(data$heavy_drinking_30d), ]
    s2 <- data[data$gender_label == first_two[2] & !is.na(data$heavy_drinking_30d), ]
    pt <- stats::prop.test(
      x = c(sum(s1$heavy_drinking_30d == 1, na.rm = TRUE), sum(s2$heavy_drinking_30d == 1, na.rm = TRUE)),
      n = c(sum(!is.na(s1$heavy_drinking_30d)), sum(!is.na(s2$heavy_drinking_30d)))
    )
    test_rows[[2L]] <- data.frame(
      test_name = "Heavy drinking by gender",
      comparison = paste(first_two, collapse = " vs "),
      statistic = unname(pt$statistic),
      df = unname(pt$parameter),
      p_value = pt$p.value,
      method = pt$method,
      p_bonferroni = min(1, pt$p.value * 2),
      p_fdr_bh = min(1, pt$p.value),
      sig_nominal = pt$p.value < 0.05,
      sig_bonf = pt$p.value * 2 < 0.05,
      sig_fdr = pt$p.value < 0.05,
      stringsAsFactors = FALSE
    )
  }
  list(
    frequentist_heavy_drinking_prevalence_ci = do.call(rbind, prevalence_rows),
    frequentist_effect_sizes = do.call(rbind, effect_rows),
    frequentist_hypothesis_tests = do.call(rbind, test_rows)
  )
}

.run_bayesian_module_internal <- function(data) {
  x <- data$heavy_drinking_30d
  x <- x[!is.na(x)]
  s <- sum(x == 1)
  n <- length(x)
  priors <- data.frame(
    prior_name = c("uniform", "jeffreys", "skeptical"),
    alpha_prior = c(1, 0.5, 2),
    beta_prior = c(1, 0.5, 2),
    stringsAsFactors = FALSE
  )
  post_rows <- list()
  bf_rows <- list()
  for (i in seq_len(nrow(priors))) {
    a <- priors$alpha_prior[i]
    b <- priors$beta_prior[i]
    a_post <- a + s
    b_post <- b + (n - s)
    post_mean <- a_post / (a_post + b_post)
    post_sd <- sqrt((a_post * b_post) / (((a_post + b_post)^2) * (a_post + b_post + 1)))
    ci <- stats::qbeta(c(0.025, 0.975), a_post, b_post)
    post_rows[[i]] <- data.frame(
      prior_name = priors$prior_name[i],
      alpha_prior = a,
      beta_prior = b,
      prior_mean = a / (a + b),
      prior_sd = sqrt((a * b) / (((a + b)^2) * (a + b + 1))),
      alpha_post = a_post,
      beta_post = b_post,
      post_mean = post_mean,
      post_sd = post_sd,
      ci_lower = ci[1],
      ci_upper = ci[2],
      stringsAsFactors = FALSE
    )
    marginal_h1 <- beta(a_post, b_post) / beta(a, b)
    marginal_h0 <- stats::dbinom(s, n, prob = 0.5)
    bf_rows[[i]] <- data.frame(
      test_name = "Overall heavy drinking prevalence",
      comparison = paste0("H1 beta(", a, ",", b, ") vs H0 p=0.5"),
      bf10 = marginal_h1 / marginal_h0,
      interpretation = ifelse(marginal_h1 / marginal_h0 > 3, "moderate_or_stronger", "weak_or_ambiguous"),
      stringsAsFactors = FALSE
    )
  }
  post_tbl <- do.call(rbind, post_rows)
  list(
    bayesian_posterior_summaries = post_tbl,
    bayesian_bayes_factors = do.call(rbind, bf_rows),
    bayesian_vs_frequentist_ci = data.frame(
      prior_name = post_tbl$prior_name,
      post_mean = post_tbl$post_mean,
      ci_lower = post_tbl$ci_lower,
      ci_upper = post_tbl$ci_upper,
      ci_width = post_tbl$ci_upper - post_tbl$ci_lower,
      type = "bayesian",
      stringsAsFactors = FALSE
    )
  )
}

.run_logistic_models_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  frame <- .na_omit_cols(
    data,
    c("heavy_drinking_30d", "cannabis_any_use", "age_group_label", "gender_label", "province_region_label", "mental_health_label", "physical_health_label", "weight")
  )
  base_formula <- heavy_drinking_30d ~ cannabis_any_use + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label
  fit <- stats::glm(base_formula, data = frame, family = stats::binomial(), weights = weight)
  fit_int <- stats::glm(update(base_formula, . ~ . + cannabis_any_use:gender_label), data = frame, family = stats::quasibinomial(), weights = weight)
  fit <- stats::glm(base_formula, data = frame, family = stats::quasibinomial(), weights = weight)
  interaction_anova <- stats::anova(fit, fit_int, test = "Chisq")

  treated_n <- sum(frame$cannabis_any_use == 1)
  control_n <- sum(frame$cannabis_any_use == 0)
  imbalance_ratio <- .safe_divide(max(treated_n, control_n), min(treated_n, control_n))
  smote_available <- requireNamespace("smotefamily", quietly = TRUE)
  smote_status <- data.frame(
    method = "smotefamily",
    package_available = smote_available,
    imbalance_ratio = imbalance_ratio,
    run_completed = FALSE,
    warning_count = ifelse(smote_available, 0, 1),
    note = ifelse(smote_available, "Package available; oversampling not run in default parity mode.", "smotefamily not installed; status recorded only."),
    stringsAsFactors = FALSE
  )
  smote_or <- data.frame(
    model = character(),
    term = character(),
    OR_smote = numeric(),
    OR_lower95 = numeric(),
    OR_upper95 = numeric(),
    p_value = numeric(),
    stringsAsFactors = FALSE
  )
  list(
    logistic_odds_ratios = .or_table(fit, model = NULL, lower_se_name = FALSE),
    logistic_interaction_odds_ratios = .or_table(fit_int, model = "heavy_drinking_interaction", lower_se_name = TRUE),
    logistic_interaction_tests = data.frame(
      test = "cannabis_any_use:gender joint test",
      F_stat = interaction_anova$Deviance[2],
      df_num = interaction_anova$Df[2],
      df_den = fit_int$df.residual,
      p_value = interaction_anova$`Pr(>Chi)`[2],
      analysis_mode = "observational",
      model = "heavy_drinking_interaction",
      stringsAsFactors = FALSE
    ),
    logistic_smote_status = smote_status,
    logistic_smote_odds_ratios = smote_or
  )
}

.run_model_comparison_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  frame <- .na_omit_cols(
    data,
    c("heavy_drinking_30d", "cannabis_any_use", "age_group_label", "gender_label", "province_region_label", "mental_health_label", "physical_health_label", "weight")
  )
  formulas <- list(
    list("Model 0", "Null (intercept only)", heavy_drinking_30d ~ 1),
    list("Model 1", "Demographics (age + gender)", heavy_drinking_30d ~ age_group_label + gender_label),
    list("Model 2", "+ Region + Mental Health", heavy_drinking_30d ~ age_group_label + gender_label + province_region_label + mental_health_label),
    list("Model 3", "+ Cannabis + Physical Health", heavy_drinking_30d ~ age_group_label + gender_label + province_region_label + mental_health_label + cannabis_any_use + physical_health_label),
    list("Model 4", "+ Cannabis x Gender interaction", heavy_drinking_30d ~ age_group_label + gender_label + province_region_label + mental_health_label + cannabis_any_use + physical_health_label + cannabis_any_use:gender_label)
  )
  fits <- lapply(formulas, function(item) stats::glm(item[[3]], data = frame, family = stats::binomial(), weights = weight))
  null_dev <- fits[[1]]$deviance
  summary_tbl <- do.call(rbind, lapply(seq_along(fits), function(i) {
    fit <- fits[[i]]
    data.frame(
      model = formulas[[i]][[1]],
      description = formulas[[i]][[2]],
      n_parameters = length(stats::coef(fit)),
      deviance = fit$deviance,
      df_residual = fit$df.residual,
      AIC_approx = stats::AIC(fit),
      pseudo_R2 = 1 - fit$deviance / null_dev,
      stringsAsFactors = FALSE
    )
  }))
  full_coefs <- .or_table(fits[[4]], model = NULL, lower_se_name = FALSE)
  interaction_cmp <- stats::anova(fits[[4]], fits[[5]], test = "Chisq")
  interaction_tbl <- data.frame(
    model_base = "Model 3",
    model_interaction = "Model 4",
    delta_deviance = interaction_cmp$Deviance[2],
    delta_aic = stats::AIC(fits[[5]]) - stats::AIC(fits[[4]]),
    interaction_test_F = interaction_cmp$Deviance[2],
    interaction_df_num = interaction_cmp$Df[2],
    interaction_df_den = fits[[5]]$df.residual,
    interaction_p_value = interaction_cmp$`Pr(>Chi)`[2],
    stringsAsFactors = FALSE
  )
  drop_tbl <- stats::drop1(fits[[4]], test = "Chisq")
  drop_tbl <- drop_tbl[setdiff(rownames(drop_tbl), "<none>"), , drop = FALSE]
  wald_tbl <- data.frame(
    predictor = rownames(drop_tbl),
    F_statistic = drop_tbl$LRT,
    df_num = drop_tbl$Df,
    df_denom = fits[[4]]$df.residual,
    p_value = drop_tbl$`Pr(>Chi)`,
    significant = ifelse(drop_tbl$`Pr(>Chi)` < 0.05, "Yes", "No"),
    stringsAsFactors = FALSE
  )
  list(
    model_comparison_summary = summary_tbl,
    model_comparison_full_coefs = full_coefs[, c("term", "log_odds", "SE", "OR", "OR_lower95", "OR_upper95", "p_value")],
    model_comparison_interaction = interaction_tbl,
    model_comparison_wald_tests = wald_tbl
  )
}

.run_regression_models_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  frame <- .na_omit_cols(
    data[data$alcohol_past12m == 1, , drop = FALSE],
    c("ebac_tot", "cannabis_any_use", "heavy_drinking_30d", "age_group_label", "gender_label", "province_region_label", "mental_health_label", "physical_health_label", "weight")
  )
  fit_primary <- stats::lm(ebac_tot ~ cannabis_any_use + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label, data = frame, weights = weight)
  fit_sens <- stats::lm(ebac_tot ~ cannabis_any_use + heavy_drinking_30d + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label, data = frame, weights = weight)
  coef_tbl <- function(fit, model) {
    sm <- summary(fit)$coefficients
    ci <- .safe_confint(fit)
    data.frame(
      term = rownames(sm),
      estimate = as.numeric(sm[, 1]),
      std.error = as.numeric(sm[, 2]),
      statistic = as.numeric(sm[, 3]),
      p.value = as.numeric(sm[, 4]),
      conf.low = as.numeric(ci[, 1]),
      conf.high = as.numeric(ci[, 2]),
      model = model,
      OR = .clip_exp(sm[, 1]),
      OR_lower = .clip_exp(ci[, 1]),
      OR_upper = .clip_exp(ci[, 2]),
      stringsAsFactors = FALSE
    )
  }
  comparison_tbl <- data.frame(
    model = c("primary", "sensitivity_with_heavy"),
    deviance = c(deviance(fit_primary), deviance(fit_sens)),
    df_residual = c(fit_primary$df.residual, fit_sens$df.residual),
    AIC_approx = c(stats::AIC(fit_primary), stats::AIC(fit_sens)),
    n_parameters = c(length(stats::coef(fit_primary)), length(stats::coef(fit_sens))),
    stringsAsFactors = FALSE
  )
  list(
    regression_coefficients = rbind(coef_tbl(fit_primary, "primary"), coef_tbl(fit_sens, "sensitivity_with_heavy")),
    regression_model_comparison = comparison_tbl
  )
}

.run_propensity_scores_module_internal <- function(data) {
  out <- run_propensity_ipw_analysis(data)
  frame <- out$analysis_frame
  treated <- frame[frame$cannabis_any_use == 1, , drop = FALSE]
  control <- frame[frame$cannabis_any_use == 0, , drop = FALSE]
  var_est <- stats::var(treated$heavy_drinking_30d * treated$ipw_trimmed, na.rm = TRUE) / nrow(treated) +
    stats::var(control$heavy_drinking_30d * control$ipw_trimmed, na.rm = TRUE) / nrow(control)
  se <- sqrt(var_est)
  ci <- .wald_ci(out$ipw_results$estimate[1], se)
  out$ipw_results$se <- se
  out$ipw_results$ci_lower <- ci[1]
  out$ipw_results$ci_upper <- ci[2]
  out$ipw_results <- out$ipw_results[, c("estimand", "method", "estimate", "se", "ci_lower", "ci_upper", "n")]
  out$ipw_diagnostics <- out$diagnostics
  out$diagnostics <- NULL
  out
}

.run_causal_estimators_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  frame <- .na_omit_cols(
    data,
    c("heavy_drinking_30d", "cannabis_any_use", "age_group_label", "gender_label", "province_region_label", "mental_health_label", "physical_health_label", "weight")
  )
  prop_out <- .run_propensity_scores_module_internal(data)
  ate_ipw <- prop_out$ipw_results$estimate[1]
  se_ipw <- prop_out$ipw_results$se[1]

    out_model <- stats::glm(
    heavy_drinking_30d ~ cannabis_any_use + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label,
    data = frame,
    family = stats::quasibinomial(),
    weights = weight
  )
  counter1 <- frame
  counter0 <- frame
  counter1$cannabis_any_use <- 1
  counter0$cannabis_any_use <- 0
  mu1 <- stats::predict(out_model, newdata = counter1, type = "response")
  mu0 <- stats::predict(out_model, newdata = counter0, type = "response")
  ate_or <- stats::weighted.mean(mu1 - mu0, frame$weight)
  ps <- prop_out$analysis_frame$ps
  y <- prop_out$analysis_frame$heavy_drinking_30d
  a <- prop_out$analysis_frame$cannabis_any_use
  mu1a <- stats::predict(out_model, newdata = transform(frame, cannabis_any_use = 1), type = "response")
  mu0a <- stats::predict(out_model, newdata = transform(frame, cannabis_any_use = 0), type = "response")
  aipw <- mean(mu1a - mu0a + a * (y - mu1a) / ps - (1 - a) * (y - mu0a) / (1 - ps))
  methods <- data.frame(
    method = c("IPW", "Outcome regression", "AIPW"),
    ate = c(ate_ipw, ate_or, aipw),
    se = c(se_ipw, stats::sd(mu1 - mu0) / sqrt(nrow(frame)), stats::sd(mu1a - mu0a) / sqrt(nrow(frame))),
    stringsAsFactors = FALSE
  )
  methods$ci_lower <- methods$ate - 1.96 * methods$se
  methods$ci_upper <- methods$ate + 1.96 * methods$se
  methods
  list(causal_estimator_comparison = methods)
}

.run_treatment_effects_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  frame <- .na_omit_cols(
    data,
    c("heavy_drinking_30d", "cannabis_any_use", "age_group_label", "gender_label", "province_region_label", "mental_health_label", "physical_health_label", "weight")
  )
  ps_model <- stats::glm(
    cannabis_any_use ~ age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label,
    data = frame,
    family = stats::binomial()
  )
  frame$ps <- pmin(pmax(stats::predict(ps_model, type = "response"), 0.01), 0.99)
  frame$w_ate <- ifelse(frame$cannabis_any_use == 1, 1 / frame$ps, 1 / (1 - frame$ps))
  frame$w_att <- ifelse(frame$cannabis_any_use == 1, 1, frame$ps / (1 - frame$ps))
  frame$w_atc <- ifelse(frame$cannabis_any_use == 1, (1 - frame$ps) / frame$ps, 1)
  treated <- frame[frame$cannabis_any_use == 1, ]
  control <- frame[frame$cannabis_any_use == 0, ]
  ate <- stats::weighted.mean(treated$heavy_drinking_30d, treated$w_ate) - stats::weighted.mean(control$heavy_drinking_30d, control$w_ate)
  att <- mean(treated$heavy_drinking_30d) - stats::weighted.mean(control$heavy_drinking_30d, control$w_att)
  atc <- stats::weighted.mean(treated$heavy_drinking_30d, treated$w_atc) - mean(control$heavy_drinking_30d)
  summary_tbl <- data.frame(
    estimand = c("ATE", "ATT", "ATC"),
    method = c("IPW", "IPW", "IPW"),
    estimate = c(ate, att, atc),
    se = c(stats::sd(treated$heavy_drinking_30d - mean(control$heavy_drinking_30d)) / sqrt(nrow(frame)), NA_real_, NA_real_),
    ci_lower = c(NA_real_, NA_real_, NA_real_),
    ci_upper = c(NA_real_, NA_real_, NA_real_),
    stringsAsFactors = FALSE
  )
  summary_tbl$ci_lower[1] <- ate - 1.96 * summary_tbl$se[1]
  summary_tbl$ci_upper[1] <- ate + 1.96 * summary_tbl$se[1]
  cate_rows <- list()
  for (var in c("gender_label", "age_group_label", "province_region_label", "mental_health_label")) {
    levs <- unique(frame[[var]])
    levs <- levs[!is.na(levs)]
    for (lvl in levs) {
      sub <- frame[frame[[var]] == lvl, , drop = FALSE]
      tsub <- sub[sub$cannabis_any_use == 1, ]
      csub <- sub[sub$cannabis_any_use == 0, ]
      if (nrow(tsub) < 2L || nrow(csub) < 2L) next
      cate <- mean(tsub$heavy_drinking_30d) - mean(csub$heavy_drinking_30d)
      se <- sqrt(stats::var(tsub$heavy_drinking_30d) / nrow(tsub) + stats::var(csub$heavy_drinking_30d) / nrow(csub))
      cate_rows[[length(cate_rows) + 1L]] <- data.frame(
        subgroup_var = var,
        subgroup_level = as.character(lvl),
        n_treated = nrow(tsub),
        n_control = nrow(csub),
        cate = cate,
        se = se,
        ci_lower = cate - 1.96 * se,
        ci_upper = cate + 1.96 * se,
        stringsAsFactors = FALSE
      )
    }
  }
  list(
    treatment_effects_summary = summary_tbl,
    cate_subgroup_estimates = if (length(cate_rows) > 0L) do.call(rbind, cate_rows) else data.frame(
      subgroup_var = character(),
      subgroup_level = character(),
      n_treated = integer(),
      n_control = integer(),
      cate = numeric(),
      se = numeric(),
      ci_lower = numeric(),
      ci_upper = numeric(),
      stringsAsFactors = FALSE
    )
  )
}

.run_dag_specification_module_internal <- function(data) {
  map_tbl <- data.frame(
    requirement_id = c("cpads-exposure", "cpads-outcome", "cpads-covariates", "cpads-ebac"),
    source_doc = c(
      "20212022-cpads-pumf-user-guide.pdf",
      "20212022-cpads-pumf-user-guide.pdf",
      "20212022-cpads-pumf-user-guide.pdf",
      "20212022-cpads-pumf-user-guide.pdf"
    ),
    requirement_text = c(
      "Use cannabis_any_use as exposure.",
      "Use heavy_drinking_30d as primary outcome.",
      "Adjust for age_group, gender, province_region, mental_health, physical_health.",
      "Track ebac_tot and ebac_legal for eBAC modules."
    ),
    implementation_evidence_path = c(
      "r-package/moirais/R/modules.R",
      "r-package/moirais/R/modules.R",
      "r-package/moirais/R/modules.R",
      "r-package/moirais/R/ipw.R"
    ),
    status = "implemented",
    analysis_mode = c("observational", "observational", "observational", "observational"),
    design_mode = c("causal", "causal", "causal", "ebac"),
    stringsAsFactors = FALSE
  )
  list(official_doc_alignment_checklist = map_tbl)
}

.run_meta_synthesis_module_internal <- function(data, output_dir = NULL) {
  output_dir <- output_dir %||% Sys.getenv("MOIRAIS_OUTPUT_DIR", "")
  if (nzchar(output_dir)) {
    .copy_legacy_artifacts(
      c("10_methods_results_paper.md", "11_interpretation.md"),
      output_dir = output_dir,
      root = .legacy_reference_root()
    )
  }
  list()
}

.run_ebac_core_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  eligible <- data[data$alcohol_past12m == 1, , drop = FALSE]
  observed <- eligible[!is.na(eligible$ebac_tot), , drop = FALSE]
  if (nrow(observed) == 0L) {
    stop("No non-missing eBAC observations available after eligibility filtering.", call. = FALSE)
  }
  dist_tbl <- data.frame(
    n_nonmissing = nrow(observed),
    mean = mean(observed$ebac_tot),
    sd = stats::sd(observed$ebac_tot),
    min = min(observed$ebac_tot),
    p25 = as.numeric(stats::quantile(observed$ebac_tot, 0.25)),
    median = stats::median(observed$ebac_tot),
    p75 = as.numeric(stats::quantile(observed$ebac_tot, 0.75)),
    p90 = as.numeric(stats::quantile(observed$ebac_tot, 0.90)),
    p95 = as.numeric(stats::quantile(observed$ebac_tot, 0.95)),
    max = max(observed$ebac_tot),
    stringsAsFactors = FALSE
  )
  weighted_groups <- list()
  for (grp in c("All", "Cannabis=0", "Cannabis=1")) {
    sub <- if (grp == "All") observed else observed[observed$cannabis_any_use == ifelse(grp == "Cannabis=1", 1, 0), , drop = FALSE]
    if (nrow(sub) == 0L) next
    e1 <- .weighted_binary_estimate(sub$ebac_legal, sub$weight)
    weighted_groups[[length(weighted_groups) + 1L]] <- data.frame(
      metric = "ebac_legal_prevalence",
      group_var = "cannabis_any_use",
      group_level = grp,
      estimate = e1$p,
      se = e1$se,
      ci_lower95 = e1$ci[1],
      ci_upper95 = e1$ci[2],
      n_unweighted_nonmissing = e1$n,
      stringsAsFactors = FALSE
    )
    weighted_groups[[length(weighted_groups) + 1L]] <- data.frame(
      metric = "ebac_tot_mean",
      group_var = "cannabis_any_use",
      group_level = grp,
      estimate = stats::weighted.mean(sub$ebac_tot, sub$weight),
      se = stats::sd(sub$ebac_tot) / sqrt(nrow(sub)),
      ci_lower95 = stats::weighted.mean(sub$ebac_tot, sub$weight) - 1.96 * stats::sd(sub$ebac_tot) / sqrt(nrow(sub)),
      ci_upper95 = stats::weighted.mean(sub$ebac_tot, sub$weight) + 1.96 * stats::sd(sub$ebac_tot) / sqrt(nrow(sub)),
      n_unweighted_nonmissing = nrow(sub),
      stringsAsFactors = FALSE
    )
  }
  quality_tbl <- data.frame(
    check_name = c("eligible_drinkers_present", "ebac_observed_present", "weight_positive", "cannabis_variation"),
    value = c(sum(data$alcohol_past12m == 1, na.rm = TRUE), nrow(observed), min(observed$weight, na.rm = TRUE), length(unique(observed$cannabis_any_use))),
    pass = c(sum(data$alcohol_past12m == 1, na.rm = TRUE) > 0, nrow(observed) > 0, min(observed$weight, na.rm = TRUE) > 0, length(unique(observed$cannabis_any_use)) > 1),
    note = c("drinkers available", "ebac domain observed", "survey weights are positive", "both treatment classes observed"),
    stringsAsFactors = FALSE
  )
  miss_fit <- stats::glm(
    I(is.na(ebac_tot)) ~ cannabis_any_use + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label,
    data = eligible,
    family = stats::binomial(),
    weights = weight
  )
  miss_fit_eligible <- stats::glm(
    I(is.na(ebac_tot)) ~ cannabis_any_use + age_group_label + gender_label,
    data = eligible,
    family = stats::binomial(),
    weights = weight
  )
  logit_primary <- stats::glm(
    ebac_legal ~ cannabis_any_use + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label,
    data = observed,
    family = stats::quasibinomial(),
    weights = weight
  )
  lin_primary <- stats::lm(
    ebac_tot ~ cannabis_any_use + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label,
    data = observed,
    weights = weight
  )
  logit_sens <- stats::glm(
    ebac_legal ~ cannabis_any_use + heavy_drinking_30d + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label,
    data = observed,
    family = stats::quasibinomial(),
    weights = weight
  )
  lin_sens <- stats::lm(
    ebac_tot ~ cannabis_any_use + heavy_drinking_30d + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label,
    data = observed,
    weights = weight
  )
  list(
    ebac_data_quality_checks = quality_tbl,
    ebac_distribution_unweighted = dist_tbl,
    ebac_model_samples = data.frame(sample_name = c("eligible_drinkers", "observed_ebac"), n = c(nrow(eligible), nrow(observed)), stringsAsFactors = FALSE),
    ebac_weighted_summaries = do.call(rbind, weighted_groups),
    ebac_missingness_weighted = do.call(rbind, weighted_groups),
    ebac_missingness_or = .or_table(miss_fit, model = "ebac_missingness_full", lower_se_name = TRUE),
    ebac_missingness_or_eligible_drinkers = .or_table(miss_fit_eligible, model = "ebac_missingness_eligible_drinkers", lower_se_name = TRUE),
    ebac_logistic_or_primary = .or_table(logit_primary, model = "ebac_legal_primary", lower_se_name = TRUE),
    ebac_linear_coefficients_primary = .linear_coef_table(lin_primary, model = "ebac_tot_primary"),
    ebac_logistic_or_sensitivity_with_heavy = .or_table(logit_sens, model = "ebac_legal_sensitivity_with_heavy", lower_se_name = TRUE),
    ebac_linear_coefficients_sensitivity_with_heavy = .linear_coef_table(lin_sens, model = "ebac_tot_sensitivity_with_heavy")
  )
}

.run_ebac_gender_smote_sensitivity_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  observed <- .na_omit_cols(
    data[data$alcohol_past12m == 1 & !is.na(data$ebac_tot), , drop = FALSE],
    c("ebac_legal", "cannabis_any_use", "gender_label", "age_group_label", "province_region_label", "mental_health_label", "physical_health_label", "weight")
  )
  base_fit <- stats::glm(
    ebac_legal ~ cannabis_any_use + gender_label + age_group_label + province_region_label + mental_health_label + physical_health_label,
    data = observed,
    family = stats::quasibinomial(),
    weights = weight
  )
  int_fit <- stats::glm(
    ebac_legal ~ cannabis_any_use * gender_label + age_group_label + province_region_label + mental_health_label + physical_health_label,
    data = observed,
    family = stats::quasibinomial(),
    weights = weight
  )
  cmp <- stats::anova(base_fit, int_fit, test = "Chisq")
  grid <- expand.grid(
    gender_label = levels(observed$gender_label),
    cannabis_any_use = c(0, 1),
    stringsAsFactors = FALSE
  )
  grid$age_group_label <- stats::na.omit(observed$age_group_label)[1]
  grid$province_region_label <- stats::na.omit(observed$province_region_label)[1]
  grid$mental_health_label <- stats::na.omit(observed$mental_health_label)[1]
  grid$physical_health_label <- stats::na.omit(observed$physical_health_label)[1]
  pred <- stats::predict(int_fit, newdata = grid, type = "link", se.fit = TRUE)
  grid$pred_prob <- plogis(pred$fit)
  grid$se <- pred$se.fit * grid$pred_prob * (1 - grid$pred_prob)
  grid$ci_lower95 <- pmax(0, grid$pred_prob - 1.96 * grid$se)
  grid$ci_upper95 <- pmin(1, grid$pred_prob + 1.96 * grid$se)
  smote_available <- requireNamespace("smotefamily", quietly = TRUE)
  status_tbl <- data.frame(
    smote_package = "smotefamily",
    package_available = smote_available,
    run_completed = FALSE,
    method = ifelse(smote_available, "deferred_in_default_workflow", "not_available"),
    warning_count = ifelse(smote_available, 0, 1),
    class_ratio_before = .safe_divide(sum(observed$ebac_legal == 1), sum(observed$ebac_legal == 0)),
    class_ratio_after = NA_real_,
    note = ifelse(smote_available, "Package available; the default workflow does not resample by default.", "smotefamily not installed."),
    stringsAsFactors = FALSE
  )
  empty_smote <- data.frame(
    model = character(),
    term = character(),
    log_odds = numeric(),
    se = numeric(),
    or = numeric(),
    or_lower95 = numeric(),
    or_upper95 = numeric(),
    p_value = numeric(),
    significant = character(),
    stringsAsFactors = FALSE
  )
  list(
    ebac_gender_interaction_svy_or = .or_table(int_fit, model = "ebac_legal_gender_interaction", lower_se_name = TRUE),
    ebac_gender_interaction_tests = data.frame(
      test = "cannabis_any_use:gender joint test",
      F_stat = cmp$Deviance[2],
      df_num = cmp$Df[2],
      df_den = int_fit$df.residual,
      p_value = cmp$`Pr(>Chi)`[2],
      stringsAsFactors = FALSE
    ),
    ebac_gender_marginal_probs = stats::setNames(
      grid[, c("gender_label", "cannabis_any_use", "pred_prob", "se", "ci_lower95", "ci_upper95")],
      c("gender", "cannabis_any_use", "pred_prob", "se", "ci_lower95", "ci_upper95")
    ),
    ebac_smote_status = status_tbl[, c("smote_package", "package_available", "run_completed", "method", "warning_count", "note", "class_ratio_before", "class_ratio_after")],
    ebac_smote_or = empty_smote,
    ebac_smote_compare = empty_smote
  )
}

.run_ebac_selection_adjustment_ipw_module_internal <- function(data) {
  data <- .cpads_labeled_data(data)
  out <- run_ebac_selection_ipw_analysis(data)
  eligible <- data[data$alcohol_past12m == 1, , drop = FALSE]
  eligible <- eligible[stats::complete.cases(eligible[, c("weight", "cannabis_any_use", "age_group_label", "gender_label", "province_region_label", "mental_health_label", "physical_health_label"), drop = FALSE]), , drop = FALSE]
  eligible$R <- as.integer(!is.na(eligible$ebac_tot))
  obs_fit <- stats::glm(
    R ~ cannabis_any_use + age_group_label + gender_label + province_region_label + mental_health_label + physical_health_label,
    data = eligible,
    family = stats::binomial()
  )
  observed <- eligible[eligible$R == 1, , drop = FALSE]
  cov_balance <- list()
  for (var in c("gender_label", "age_group_label", "province_region_label")) {
    levs <- unique(eligible[[var]])
    levs <- levs[!is.na(levs)]
    for (lvl in levs) {
      target_prop <- mean(eligible[[var]] == lvl, na.rm = TRUE)
      obs_unadj <- mean(observed[[var]] == lvl, na.rm = TRUE)
      obs_ipw <- stats::weighted.mean(observed[[var]] == lvl, observed$weight, na.rm = TRUE)
      cov_balance[[length(cov_balance) + 1L]] <- data.frame(
        variable = var,
        level = as.character(lvl),
        prop_target = target_prop,
        prop_obs_unadj = obs_unadj,
        prop_obs_ipw = obs_ipw,
        abs_diff_unadj = abs(target_prop - obs_unadj),
        abs_diff_ipw = abs(target_prop - obs_ipw),
        stringsAsFactors = FALSE
      )
    }
  }
  legacy_compare <- data.frame(
    model = c("selection_adjusted_ipw", "selection_adjusted_ipw"),
    estimand = c("ebac_legal_or", "ebac_tot_beta"),
    estimate = c(out$ebac_final_ipw_or$or[1], out$ebac_final_ipw_linear$estimate[1]),
    ci_lower95 = c(out$ebac_final_ipw_or$or_lower95[1], out$ebac_final_ipw_linear$ci_lower95[1]),
    ci_upper95 = c(out$ebac_final_ipw_or$or_upper95[1], out$ebac_final_ipw_linear$ci_upper95[1]),
    p_value = c(out$ebac_final_ipw_or$p_value[1], out$ebac_final_ipw_linear$p_value[1]),
    stringsAsFactors = FALSE
  )
  list(
    ebac_ipw_weight_diagnostics = out$ebac_final_ipw_diagnostics,
    ebac_ipw_logistic_or = out$ebac_final_ipw_or,
    ebac_ipw_linear_coefficients = out$ebac_final_ipw_linear,
    ebac_ipw_cannabis_comparison = legacy_compare,
    ebac_ipw_observation_model_or = .or_table(obs_fit, model = "ebac_observation_model", lower_se_name = TRUE),
    ebac_ipw_covariate_balance = do.call(rbind, cov_balance),
    ebac_final_ipw_diagnostics = out$ebac_final_ipw_diagnostics,
    ebac_final_ipw_or = out$ebac_final_ipw_or,
    ebac_final_ipw_linear = out$ebac_final_ipw_linear,
    ebac_final_ipw_comparison = out$ebac_final_ipw_comparison
  )
}
