inv_logit <- function(x) {
  1 / (1 + exp(-x))
}

inject_special_codes <- function(x, rate = 0.02, codes = c(97L, 98L, 99L, 997L, 998L, 999L)) {
  if (rate <= 0) return(x)
  n <- length(x)
  idx <- stats::runif(n) < rate
  if (any(idx)) {
    x[idx] <- sample(codes, sum(idx), replace = TRUE)
  }
  x
}

synthetic_required_keys <- function() {
  c(
    "id", "weight", "sex", "age_group", "region",
    "alcohol_ever", "alcohol_12m", "alcohol_freq_30d",
    "heavy_drinking_30d", "bac", "bac_legal", "risk_indicator",
    "cannabis_use", "physical_health", "mental_health"
  )
}

resolve_synthetic_name_map <- function(name_map, profile) {
  required <- synthetic_required_keys()

  if (is.null(name_map)) {
    return(default_synthetic_name_map(profile = profile))
  }

  if (is.list(name_map)) {
    name_map <- unlist(name_map, use.names = TRUE)
  }
  if (!is.character(name_map) || is.null(names(name_map))) {
    stop("`name_map` must be a named character vector.", call. = FALSE)
  }

  missing_keys <- setdiff(required, names(name_map))
  if (length(missing_keys) > 0) {
    stop(
      "`name_map` is missing required keys: ",
      paste(missing_keys, collapse = ", "),
      call. = FALSE
    )
  }

  resolved <- name_map[required]
  if (anyNA(resolved) || any(!nzchar(resolved))) {
    stop("`name_map` has empty values for one or more required keys.", call. = FALSE)
  }
  if (anyDuplicated(unname(resolved)) > 0) {
    stop("`name_map` values must be unique.", call. = FALSE)
  }

  resolved
}

#' Default synthetic-data variable name map
#'
#' Returns a named character vector mapping canonical variable keys used by
#' [generate_synthetic_data()] to output column names.
#'
#' @param profile Name profile. `"generic"` is recommended for new projects.
#'   `"morie_legacy"` reproduces previous EML legacy column names.
#' @return Named character vector.
#' @export
default_synthetic_name_map <- function(profile = c("generic", "morie_legacy")) {
  profile <- match.arg(profile)

  if (identical(profile, "generic")) {
    return(c(
      id = "id",
      weight = "weight",
      sex = "sex",
      age_group = "age_group",
      region = "region",
      alcohol_ever = "alcohol_ever",
      alcohol_12m = "alcohol_12m",
      alcohol_freq_30d = "alcohol_freq_30d",
      heavy_drinking_30d = "heavy_drinking_30d",
      bac = "bac",
      bac_legal = "bac_legal",
      risk_indicator = "risk_indicator",
      cannabis_use = "cannabis_use",
      physical_health = "physical_health",
      mental_health = "mental_health"
    ))
  }

  c(
    id = "seqid",
    weight = "weight",
    sex = "dvdemq01",
    age_group = "age_groups",
    region = "region",
    alcohol_ever = "alc03",
    alcohol_12m = "alc05",
    alcohol_freq_30d = "alc06",
    heavy_drinking_30d = "alc12_30d_prev_total",
    bac = "ebac_tot",
    bac_legal = "ebac_legal",
    risk_indicator = "alc13_lrdg",
    cannabis_use = "can05",
    physical_health = "hwbq01",
    mental_health = "hwbq02"
  )
}

#' Generate synthetic epidemiology-style tabular data
#'
#' Generates non-identifying synthetic data suitable for development, testing,
#' and demos. The generator uses a canonical variable set and allows output
#' column renaming through `name_map` so it can be adapted to multiple studies.
#' Synthetic data should not be used for final inferential reporting.
#'
#' @param n Number of rows.
#' @param seed Random seed for reproducibility.
#' @param special_code_rate Proportion of values replaced with survey-style
#'   special missing codes (`97/98/99/997/998/999`) in discrete fields.
#' @param profile Convenience profile for output naming; ignored when
#'   `name_map` is supplied.
#' @param name_map Optional named character vector mapping canonical keys to
#'   output column names. Use [default_synthetic_name_map()] as a template.
#' @return A data.frame with synthetic records.
#' @export
generate_synthetic_data <- function(
  n = 5000L,
  seed = 42L,
  special_code_rate = 0.02,
  profile = c("generic", "morie_legacy"),
  name_map = NULL
) {
  n <- as.integer(n)
  if (is.na(n) || n < 100L) {
    stop("`n` must be an integer >= 100.", call. = FALSE)
  }
  if (!is.numeric(special_code_rate) || length(special_code_rate) != 1 ||
      is.na(special_code_rate) || special_code_rate < 0 || special_code_rate > 0.2) {
    stop("`special_code_rate` must be in [0, 0.2].", call. = FALSE)
  }

  profile <- match.arg(profile)
  map <- resolve_synthetic_name_map(name_map = name_map, profile = profile)

  set.seed(as.integer(seed))

  id <- seq_len(n)
  age_group <- sample(1:4, n, replace = TRUE, prob = c(0.27, 0.34, 0.23, 0.16))
  sex <- sample(1:3, n, replace = TRUE, prob = c(0.50, 0.47, 0.03))
  region <- sample(1:4, n, replace = TRUE, prob = c(0.11, 0.23, 0.39, 0.27))

  physical_health <- sample(1:5, n, replace = TRUE, prob = c(0.14, 0.25, 0.35, 0.18, 0.08))
  mental_health <- sample(1:5, n, replace = TRUE, prob = c(0.10, 0.22, 0.36, 0.21, 0.11))

  age_effect <- c(0.30, 0.20, 0.00, -0.20)
  mh_effect <- c(-0.10, 0.00, 0.10, 0.20, 0.35)
  sex_effect <- ifelse(sex == 2L, 0.12, 0.00)

  cannabis_prob <- inv_logit(-1.25 + age_effect[age_group] + mh_effect[mental_health] + sex_effect)
  cannabis_use <- ifelse(stats::runif(n) < cannabis_prob, 1L, 2L)

  alcohol_ever_prob <- inv_logit(1.90 + 0.10 * (age_group <= 2))
  alcohol_ever <- ifelse(stats::runif(n) < alcohol_ever_prob, 1L, 2L)

  alcohol_12m_prob <- inv_logit(0.70 + 0.25 * (age_group <= 2) + 0.20 * (cannabis_use == 1L))
  alcohol_12m <- ifelse(stats::runif(n) < alcohol_12m_prob, 1L, 2L)

  alcohol_freq_30d <- integer(n)
  for (i in seq_len(n)) {
    p <- if (alcohol_12m[i] == 1L) {
      if (cannabis_use[i] == 1L) c(0.08, 0.20, 0.19, 0.23, 0.17, 0.13) else c(0.05, 0.15, 0.17, 0.24, 0.20, 0.19)
    } else {
      c(0.01, 0.02, 0.03, 0.09, 0.15, 0.70)
    }
    alcohol_freq_30d[i] <- sample(1:6, size = 1, prob = p)
  }

  heavy_prob <- inv_logit(
    -1.00 +
      0.85 * (cannabis_use == 1L) +
      0.55 * (alcohol_freq_30d <= 3) +
      0.15 * (mental_health >= 4) +
      0.08 * (sex == 2L)
  )

  heavy_drinking_30d <- stats::rbinom(n, 1L, heavy_prob)

  bac_linear <- 0.015 +
    0.018 * (alcohol_freq_30d <= 4) +
    0.030 * (alcohol_freq_30d <= 3) +
    0.042 * (heavy_drinking_30d == 1L) +
    0.010 * (cannabis_use == 1L) +
    stats::rnorm(n, mean = 0, sd = 0.015)
  bac <- round(pmax(0, pmin(0.35, bac_linear)), 3)
  bac_legal <- as.integer(bac > 0.08)

  # Closely related legal-risk field with small disagreement for realism.
  risk_indicator <- ifelse(stats::runif(n) < 0.97, bac_legal, 1L - bac_legal)

  weight <- round(stats::rgamma(n, shape = 2.4, scale = 45), 1)

  core <- data.frame(
    id = id,
    weight = weight,
    sex = sex,
    age_group = age_group,
    region = region,
    alcohol_ever = alcohol_ever,
    alcohol_12m = alcohol_12m,
    alcohol_freq_30d = alcohol_freq_30d,
    heavy_drinking_30d = heavy_drinking_30d,
    bac = bac,
    bac_legal = bac_legal,
    risk_indicator = risk_indicator,
    cannabis_use = cannabis_use,
    physical_health = physical_health,
    mental_health = mental_health,
    stringsAsFactors = FALSE
  )

  names(core) <- unname(map[names(core)])

  discrete_keys <- c(
    "sex", "age_group", "region", "alcohol_ever", "alcohol_12m",
    "alcohol_freq_30d", "heavy_drinking_30d", "bac_legal",
    "risk_indicator", "cannabis_use", "physical_health", "mental_health"
  )
  for (k in discrete_keys) {
    nm <- map[[k]]
    core[[nm]] <- inject_special_codes(core[[nm]], rate = special_code_rate)
  }

  attr(core, "synthetic") <- TRUE
  attr(core, "synthetic_note") <- "Synthetic data for development/testing only"
  attr(core, "synthetic_profile") <- profile
  attr(core, "synthetic_name_map") <- map
  core
}

#' Write synthetic epidemiology-style data to CSV
#'
#' @param path Output CSV path.
#' @param n Number of rows.
#' @param seed Random seed.
#' @param special_code_rate Proportion of survey-style missing codes.
#' @param profile Naming profile for output columns.
#' @param name_map Optional custom variable name map.
#' @param overwrite If `TRUE`, overwrite existing file.
#' @return Normalized output path.
#' @export
write_synthetic_data <- function(
  path,
  n = 5000L,
  seed = 42L,
  special_code_rate = 0.02,
  profile = c("generic", "morie_legacy"),
  name_map = NULL,
  overwrite = FALSE
) {
  if (missing(path) || !nzchar(path)) {
    stop("Provide a non-empty `path`.", call. = FALSE)
  }
  if (file.exists(path) && !isTRUE(overwrite)) {
    stop("File already exists. Set `overwrite = TRUE` to replace it.", call. = FALSE)
  }

  dat <- generate_synthetic_data(
    n = n,
    seed = seed,
    special_code_rate = special_code_rate,
    profile = profile,
    name_map = name_map
  )
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(dat, path, row.names = FALSE)
  normalizePath(path, winslash = "/", mustWork = FALSE)
}
