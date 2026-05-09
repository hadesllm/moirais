#' List implemented MOIRAIS CPADS modules
#'
#' @return Data frame describing the implemented module surface.
#' @export
list_moirais_modules <- function() {
  data.frame(
    name = c(
      "data-wrangling",
      "descriptive-statistics",
      "distribution-tests",
      "frequentist-inference",
      "bayesian-inference",
      "power-design",
      "logistic-models",
      "model-comparison",
      "regression-models",
      "propensity-scores",
      "causal-estimators",
      "treatment-effects",
      "dag-specification",
      "meta-synthesis",
      "ebac-core",
      "ebac-selection-adjustment-ipw",
      "ebac-integrations",
      "ebac-gender-smote-sensitivity",
      "figures",
      "tables",
      "final-report"
    ),
    description = c(
      "Canonicalize and validate the real CPADS PUMF input.",
      "Survey-weighted prevalence and probability summaries.",
      "Distributional diagnostics, correlations, and CLT checks.",
      "Frequentist prevalence, effect-size, and hypothesis-test outputs.",
      "Beta-binomial Bayesian summaries for key CPADS endpoints.",
      "Survey-weighted power planning summaries from real CPADS data.",
      "Weighted logistic models for heavy-drinking outcomes.",
      "Nested heavy-drinking model comparison outputs.",
      "Weighted regression models for eBAC outcomes.",
      "Propensity/IPW workflow for cannabis and heavy drinking.",
      "Causal-estimator comparison across IPW, outcome-regression, and AIPW.",
      "ATE, ATT, ATC, and subgroup treatment-effect summaries.",
      "DAG and official-document alignment checklist outputs.",
      "Narrative synthesis outputs for study integration and interpretation.",
      "Core eBAC weighted, missingness, and model outputs.",
      "Selection-adjusted eBAC IPW workflow.",
      "Integrated eBAC final-summary outputs.",
      "eBAC interaction and SMOTE-sensitivity status outputs.",
      "Figure exports for the documented analysis workflow.",
      "HTML table exports for the documented analysis workflow.",
      "Final report and output-audit summaries."
    ),
    stringsAsFactors = FALSE
  )
}

.cpads_default_csv <- function() {
  # Primary: built-in SQLite DB. Fallback: raw CSV in datasets/.
  candidates <- c(
    "data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv"
  )
  for (p in candidates) {
    if (file.exists(p)) return(p)
  }
  # Fallback: first candidate (will be resolved by .resolve_cpads_csv).
  candidates[1L]
}

.resolve_cpads_csv <- function(cpads_csv) {
  if (file.exists(cpads_csv)) {
    return(normalizePath(cpads_csv, mustWork = TRUE))
  }
  current <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  for (i in seq_len(10L)) {
    candidate <- file.path(current, cpads_csv)
    if (file.exists(candidate)) {
      return(normalizePath(candidate, mustWork = TRUE))
    }
    parent <- dirname(current)
    if (identical(parent, current)) {
      break
    }
    current <- parent
  }
  stop("CPADS CSV not found: ", cpads_csv, call. = FALSE)
}

#' Canonicalize raw CPADS PUMF columns
#'
#' @param data Raw CPADS data frame.
#' @return Data frame with canonical MOIRAIS analysis columns.
#' @export
canonicalize_cpads_data <- function(data) {
  required_raw <- c(
    "wtpumf", "alc05", "alc12_30d_prev_total", "alc12_30d_prev", "can05", "age_groups",
    "dvdemq01", "region", "hwbq01", "hwbq02", "ebac_tot", "ebac_legal"
  )
  missing_raw <- setdiff(required_raw, names(data))
  if (length(missing_raw) > 0) {
    validate_cpads_data(data, strict = TRUE)
    return(data)
  }

  canonical <- data.frame(
    weight = as.numeric(data$wtpumf),
    alcohol_past12m = ifelse(data$alc05 == 1, 1, ifelse(data$alc05 == 2, 0, NA)),
    heavy_drinking_30d = ifelse(
      data$alc12_30d_prev_total == 1,
      1,
      ifelse(
        data$alc12_30d_prev_total == 0,
        0,
        ifelse(data$alc12_30d_prev == 1, 1, ifelse(data$alc12_30d_prev == 0, 0, NA))
      )
    ),
    ebac_tot = as.numeric(data$ebac_tot),
    ebac_legal = as.numeric(data$ebac_legal),
    cannabis_any_use = ifelse(data$can05 == 1, 1, ifelse(data$can05 == 2, 0, NA)),
    age_group = ifelse(data$age_groups %in% c(98, 99), NA, data$age_groups),
    gender = ifelse(data$dvdemq01 %in% c(98, 99), NA, data$dvdemq01),
    province_region = ifelse(data$region %in% c(98, 99), NA, data$region),
    mental_health = ifelse(data$hwbq02 %in% c(98, 99), NA, data$hwbq02),
    physical_health = ifelse(data$hwbq01 %in% c(98, 99), NA, data$hwbq01)
  )
  out <- data
  for (nm in names(canonical)) {
    out[[nm]] <- canonical[[nm]]
  }
  validate_cpads_data(out, strict = TRUE)
  out
}

#' Load the real CPADS CSV from this repository
#'
#' @param cpads_csv Path to the CPADS CSV.
#' @return Canonicalized CPADS data frame.
#' @export
load_cpads_data <- function(cpads_csv = .cpads_default_csv()) {
  cpads_csv <- .resolve_cpads_csv(cpads_csv)
  canonicalize_cpads_data(utils::read.csv(cpads_csv, stringsAsFactors = FALSE))
}

.write_module_outputs <- function(outputs, output_dir = NULL) {
  if (is.null(output_dir)) {
    return(outputs)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  for (nm in names(outputs)) {
    if (is.data.frame(outputs[[nm]])) {
      file_name <- if (grepl("\\.[A-Za-z0-9]+$", nm)) nm else paste0(nm, ".csv")
      utils::write.csv(outputs[[nm]], file.path(output_dir, file_name), row.names = FALSE)
    }
    if (is.character(outputs[[nm]]) && length(outputs[[nm]]) == 1L) {
      file_name <- if (grepl("\\.[A-Za-z0-9]+$", nm)) nm else paste0(nm, ".txt")
      writeLines(outputs[[nm]], file.path(output_dir, file_name), useBytes = TRUE)
    }
  }
  outputs
}

#' Run one implemented MOIRAIS module against CPADS data
#'
#' @param module_name Module name.
#' @param cpads_csv Path to the CPADS CSV.
#' @param output_dir Optional directory for CSV outputs.
#' @return Named list of data-frame outputs.
#' @export
run_moirais_module <- function(module_name, cpads_csv = .cpads_default_csv(), output_dir = NULL) {
  data <- load_cpads_data(cpads_csv)

  outputs <- switch(
    module_name,
    "data-wrangling" = .run_data_wrangling_module_internal(data, cpads_csv = cpads_csv, output_dir = output_dir),
    "descriptive-statistics" = .run_descriptive_statistics_module_internal(data),
    "distribution-tests" = .run_distribution_tests_module_internal(data),
    "frequentist-inference" = .run_frequentist_module_internal(data),
    "bayesian-inference" = .run_bayesian_module_internal(data),
    "power-design" = .run_power_design_module_extended(data),
    "logistic-models" = .run_logistic_models_module_internal(data),
    "model-comparison" = .run_model_comparison_module_internal(data),
    "regression-models" = .run_regression_models_module_internal(data),
    "propensity-scores" = .run_propensity_scores_module_internal(data),
    "causal-estimators" = .run_causal_estimators_module_internal(data),
    "treatment-effects" = .run_treatment_effects_module_internal(data),
    "dag-specification" = .run_dag_specification_module_internal(data),
    "meta-synthesis" = .run_meta_synthesis_module_internal(data, output_dir = output_dir),
    "ebac-core" = .run_ebac_core_module_internal(data),
    "ebac-selection-adjustment-ipw" = .run_ebac_selection_adjustment_ipw_module_internal(data),
    "ebac-integrations" = .run_ebac_integrations_module_internal(data, output_dir = output_dir),
    "ebac-gender-smote-sensitivity" = .run_ebac_gender_smote_sensitivity_module_internal(data),
    "figures" = .run_figures_module_internal(data, output_dir = output_dir),
    "tables" = .run_tables_module_internal(data, output_dir = output_dir),
    "final-report" = .run_final_report_module_internal(data, output_dir = output_dir),
    stop("Unknown module: ", module_name, call. = FALSE)
  )

  outputs <- outputs[!names(outputs) %in% c("analysis_frame")]
  .write_module_outputs(outputs, output_dir)
}

#' Run multiple implemented MOIRAIS modules
#'
#' @param modules Character vector of module names.
#' @param cpads_csv Path to the CPADS CSV.
#' @param output_dir Optional directory for CSV outputs.
#' @return Named list of module outputs.
#' @export
run_moirais_modules <- function(
  modules = list_moirais_modules()$name,
  cpads_csv = .cpads_default_csv(),
  output_dir = NULL
) {
  stats::setNames(
    lapply(modules, function(m) run_moirais_module(m, cpads_csv = cpads_csv, output_dir = output_dir)),
    modules
  )
}
