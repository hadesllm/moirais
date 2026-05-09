#' Return the canonical CPADS local-data contract
#'
#' @return Named list describing the expected local CPADS contract.
#' @export
cpads_contract <- function() {
  list(
    source_kind = "local_private_file",
    expected_wrangled_path = "data/cache/cpads_pumf_wrangled.rds",
    required_variables = c(
      "weight",
      "alcohol_past12m",
      "heavy_drinking_30d",
      "ebac_tot",
      "ebac_legal",
      "cannabis_any_use",
      "age_group",
      "gender",
      "province_region",
      "mental_health",
      "physical_health"
    ),
    note = "CPADS row-level data must be supplied locally and must not be committed to git."
  )
}

#' Validate a CPADS analysis data frame
#'
#' @param data Data frame to validate.
#' @param strict If `TRUE`, stop when required variables are missing.
#' @return Character vector of missing variable names.
#' @export
validate_cpads_data <- function(data, strict = TRUE) {
  required <- cpads_contract()$required_variables
  missing <- setdiff(required, names(data))
  if (isTRUE(strict) && length(missing) > 0) {
    stop("CPADS data is missing required variables: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  missing
}

.weighted_prop <- function(x, w) {
  sum(x * w, na.rm = TRUE) / sum(w, na.rm = TRUE)
}

.ess <- function(w) {
  (sum(w)^2) / sum(w^2)
}

#' Run the CPADS propensity/IPW workflow
#'
#' Mirrors the core outputs of the old `07_propensity.R` workflow.
#'
#' @param data Analysis data frame.
#' @param output_dir Optional directory for CSV outputs.
#' @param trim Quantile pair used to trim extreme IPW values.
#' @param treatment Binary treatment column.
#' @param outcome Binary outcome column.
#' @param covariates Covariate names for the propensity model.
#' @return Named list of output tables and the analysis data.
#' @export
run_propensity_ipw_analysis <- function(
  data,
  output_dir = NULL,
  trim = c(0.01, 0.99),
  treatment = "cannabis_any_use",
  outcome = "heavy_drinking_30d",
  covariates = c("age_group", "gender", "province_region", "mental_health", "physical_health")
) {
  validate_cpads_data(data, strict = TRUE)
  needed <- unique(c(treatment, outcome, covariates, "weight"))
  frame <- stats::na.omit(data[, needed, drop = FALSE])

  ps_formula <- stats::as.formula(
    paste(treatment, "~", paste(covariates, collapse = " + "))
  )
  ps_model <- stats::glm(ps_formula, data = frame, family = stats::binomial())
  frame$ps <- pmin(pmax(stats::predict(ps_model, type = "response"), 0.01), 0.99)
  frame$ipw <- ifelse(frame[[treatment]] == 1, 1 / frame$ps, 1 / (1 - frame$ps))
  q01 <- as.numeric(stats::quantile(frame$ipw, trim[[1]], na.rm = TRUE))
  q99 <- as.numeric(stats::quantile(frame$ipw, trim[[2]], na.rm = TRUE))
  frame$ipw_trimmed <- pmin(pmax(frame$ipw, q01), q99)

  treated <- frame[frame[[treatment]] == 1, , drop = FALSE]
  control <- frame[frame[[treatment]] == 0, , drop = FALSE]
  y1_ipw <- .weighted_prop(treated[[outcome]], treated$ipw_trimmed)
  y0_ipw <- .weighted_prop(control[[outcome]], control$ipw_trimmed)
  ate_ipw <- y1_ipw - y0_ipw

  ipw_results <- data.frame(
    estimand = "ATE",
    method = "IPW (trimmed)",
    estimate = ate_ipw,
    n = nrow(frame),
    y1_ipw = y1_ipw,
    y0_ipw = y0_ipw,
    stringsAsFactors = FALSE
  )

  diagnostics <- data.frame(
    metric = c("ps_mean", "ps_min", "ps_max", "ipw_mean", "ipw_trimmed_mean", "ess_ipw_trimmed"),
    value = c(
      mean(frame$ps),
      min(frame$ps),
      max(frame$ps),
      mean(frame$ipw),
      mean(frame$ipw_trimmed),
      .ess(frame$ipw_trimmed)
    ),
    stringsAsFactors = FALSE
  )

  if (!is.null(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(ipw_results, file.path(output_dir, "ipw_results.csv"), row.names = FALSE)
    utils::write.csv(diagnostics, file.path(output_dir, "ipw_diagnostics.csv"), row.names = FALSE)
  }

  list(
    analysis_frame = frame,
    ipw_results = ipw_results,
    diagnostics = diagnostics
  )
}

#' Run the eBAC selection-adjusted IPW workflow
#'
#' Mirrors the core outputs of the old `07_ebac_ipw.R` workflow.
#'
#' @param data Analysis data frame.
#' @param output_dir Optional directory for CSV outputs.
#' @param treatment Treatment column name.
#' @param covariates Covariate names used in the observation model.
#' @return Named list of output tables and the observed-domain analysis frame.
#' @export
run_ebac_selection_ipw_analysis <- function(
  data,
  output_dir = NULL,
  treatment = "cannabis_any_use",
  covariates = c("age_group", "gender", "province_region", "mental_health", "physical_health")
) {
  validate_cpads_data(data, strict = TRUE)
  if (!requireNamespace("survey", quietly = TRUE)) {
    stop("The `survey` package is required for run_ebac_selection_ipw_analysis().", call. = FALSE)
  }

  needed <- unique(c(
    "weight", "alcohol_past12m", "ebac_tot", "ebac_legal", treatment, covariates
  ))
  target <- data[, needed, drop = FALSE]
  target <- target[target$alcohol_past12m == 1, , drop = FALSE]
  target <- target[stats::complete.cases(target[, c("weight", treatment, covariates), drop = FALSE]), , drop = FALSE]
  target$R <- as.integer(!is.na(target$ebac_tot))

  obs_formula <- stats::as.formula(
    paste("R ~", paste(c(treatment, covariates), collapse = " + "))
  )
  obs_model <- stats::glm(obs_formula, data = target, family = stats::binomial())
  target$p_hat <- pmin(pmax(stats::predict(obs_model, type = "response"), 0.01), 0.99)
  p_obs <- mean(target$R)

  observed <- target[target$R == 1, , drop = FALSE]
  observed$sw <- p_obs / observed$p_hat
  q01 <- as.numeric(stats::quantile(observed$sw, 0.01, na.rm = TRUE))
  q99 <- as.numeric(stats::quantile(observed$sw, 0.99, na.rm = TRUE))
  observed$sw_trim <- pmin(pmax(observed$sw, q01), q99)
  observed$w_combined_trim <- observed$weight * observed$sw_trim

  diag_tbl <- data.frame(
    metric = c("eligible_n", "observed_n", "observed_rate", "sw_min", "sw_q01", "sw_q99", "sw_max", "ess_survey_x_ipw_trim"),
    value = c(
      nrow(target),
      nrow(observed),
      p_obs,
      min(observed$sw),
      q01,
      q99,
      max(observed$sw),
      .ess(observed$w_combined_trim)
    ),
    stringsAsFactors = FALSE
  )

  des_ipw <- survey::svydesign(ids = ~1, weights = ~w_combined_trim, data = observed)
  fit_bin <- survey::svyglm(
    stats::as.formula(paste("ebac_legal ~", paste(c(treatment, covariates), collapse = " + "))),
    design = des_ipw,
    family = stats::quasibinomial()
  )
  fit_lin <- survey::svyglm(
    stats::as.formula(paste("ebac_tot ~", paste(c(treatment, covariates), collapse = " + "))),
    design = des_ipw
  )

  bin_coef <- summary(fit_bin)$coefficients[treatment, , drop = FALSE]
  bin_ci <- suppressWarnings(stats::confint(fit_bin))[treatment, ]
  lin_coef <- summary(fit_lin)$coefficients[treatment, , drop = FALSE]
  lin_ci <- suppressWarnings(stats::confint(fit_lin))[treatment, ]

  ebac_final_ipw_or <- data.frame(
    model = "selection_adjusted_ipw",
    term = treatment,
    log_odds = as.numeric(bin_coef[1, 1]),
    se = as.numeric(bin_coef[1, 2]),
    or = exp(as.numeric(bin_coef[1, 1])),
    or_lower95 = exp(as.numeric(bin_ci[1])),
    or_upper95 = exp(as.numeric(bin_ci[2])),
    p_value = as.numeric(bin_coef[1, ncol(bin_coef)]),
    significant = ifelse(as.numeric(bin_coef[1, ncol(bin_coef)]) < 0.05, "*", ""),
    stringsAsFactors = FALSE
  )

  ebac_final_ipw_linear <- data.frame(
    model = "selection_adjusted_ipw",
    term = treatment,
    estimate = as.numeric(lin_coef[1, 1]),
    se = as.numeric(lin_coef[1, 2]),
    ci_lower95 = as.numeric(lin_ci[1]),
    ci_upper95 = as.numeric(lin_ci[2]),
    p_value = as.numeric(lin_coef[1, ncol(lin_coef)]),
    significant = ifelse(as.numeric(lin_coef[1, ncol(lin_coef)]) < 0.05, "*", ""),
    stringsAsFactors = FALSE
  )

  ebac_final_ipw_comparison <- data.frame(
    metric = c("ebac_legal_or_cannabis", "ebac_tot_beta_cannabis"),
    model = c("selection_adjusted_ipw", "selection_adjusted_ipw"),
    estimate = c(ebac_final_ipw_or$or, ebac_final_ipw_linear$estimate),
    ci_lower95 = c(ebac_final_ipw_or$or_lower95, ebac_final_ipw_linear$ci_lower95),
    ci_upper95 = c(ebac_final_ipw_or$or_upper95, ebac_final_ipw_linear$ci_upper95),
    p_value = c(ebac_final_ipw_or$p_value, ebac_final_ipw_linear$p_value),
    stringsAsFactors = FALSE
  )

  if (!is.null(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(diag_tbl, file.path(output_dir, "ebac_final_ipw_diagnostics.csv"), row.names = FALSE)
    utils::write.csv(ebac_final_ipw_or, file.path(output_dir, "ebac_final_ipw_or.csv"), row.names = FALSE)
    utils::write.csv(ebac_final_ipw_linear, file.path(output_dir, "ebac_final_ipw_linear.csv"), row.names = FALSE)
    utils::write.csv(ebac_final_ipw_comparison, file.path(output_dir, "ebac_final_ipw_comparison.csv"), row.names = FALSE)
  }

  list(
    analysis_frame = observed,
    ebac_final_ipw_diagnostics = diag_tbl,
    ebac_final_ipw_or = ebac_final_ipw_or,
    ebac_final_ipw_linear = ebac_final_ipw_linear,
    ebac_final_ipw_comparison = ebac_final_ipw_comparison
  )
}
