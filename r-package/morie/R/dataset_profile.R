#' Infer the measurement level of a vector
#'
#' Mirrors the Python `morie.infer_measurement_level()`. Heuristically
#' classifies a vector as one of `"binary"`, `"nominal"`, `"ordinal"`,
#' `"interval"`, or `"ratio"` based on Stevens' (1946) typology.
#'
#' Rules: logical or 2-level factor/character -> `"binary"`; ordered factor ->
#' `"ordinal"`; unordered factor or character -> `"nominal"`; integer/numeric
#' with non-negative range -> `"ratio"`; otherwise -> `"interval"`.
#'
#' @param x A vector (any atomic type or factor).
#'
#' @return Character scalar in
#'   `c("binary", "nominal", "ordinal", "interval", "ratio")`.
#' @export
#' @examples
#' infer_measurement_level(c(0, 1, 1, 0))               # "binary"
#' infer_measurement_level(factor(c("a", "b", "c")))    # "nominal"
#' infer_measurement_level(ordered(c("low", "med", "high"))) # "ordinal"
#' infer_measurement_level(c(1.2, 3.4, 5.6))            # "ratio"
#' infer_measurement_level(c(-1.5, 0.0, 2.3))           # "interval"
infer_measurement_level <- function(x) {
  if (is.logical(x)) return("binary")
  uniq <- unique(stats::na.omit(x))
  if (length(uniq) == 2L && all(uniq %in% c(0, 1, "0", "1", TRUE, FALSE)))
    return("binary")
  if (is.ordered(x)) return("ordinal")
  if (is.factor(x) || is.character(x))
    return(if (length(uniq) == 2L) "binary" else "nominal")
  if (is.numeric(x)) {
    if (length(uniq) == 2L) return("binary")
    return(if (all(x >= 0, na.rm = TRUE)) "ratio" else "interval")
  }
  "nominal"
}

#' Profile a data.frame: per-column types, missingness, summary statistics
#'
#' Mirrors the Python `morie.profile_dataset()`. Returns a list of
#' per-column profiles plus dataset-level metadata.
#'
#' @param df A `data.frame`.
#'
#' @return A list with components:
#' \describe{
#'   \item{`n_rows`, `n_cols`}{Dataset dimensions.}
#'   \item{`columns`}{A named list, one entry per column, each containing
#'     `name`, `dtype`, `measurement_level`, `n_missing`, `n_unique`, and
#'     (for numeric columns) `mean`, `sd`, `min`, `max`, `q25`, `q50`, `q75`.}
#' }
#' @export
#' @examples
#' p <- profile_dataset(iris)
#' p$columns$Species
#' p$columns$Sepal.Length
profile_dataset <- function(df) {
  if (!is.data.frame(df)) stop("df must be a data.frame.", call. = FALSE)
  cols <- lapply(names(df), function(nm) {
    x <- df[[nm]]
    base <- list(
      name              = nm,
      dtype             = paste(class(x), collapse = "/"),
      measurement_level = infer_measurement_level(x),
      n_missing         = sum(is.na(x)),
      n_unique          = length(unique(stats::na.omit(x)))
    )
    if (is.numeric(x)) {
      qs <- stats::quantile(x, probs = c(0.25, 0.5, 0.75), na.rm = TRUE,
                            names = FALSE)
      base$mean <- mean(x, na.rm = TRUE)
      base$sd   <- stats::sd(x, na.rm = TRUE)
      base$min  <- min(x, na.rm = TRUE)
      base$max  <- max(x, na.rm = TRUE)
      base$q25  <- qs[[1L]]
      base$q50  <- qs[[2L]]
      base$q75  <- qs[[3L]]
    }
    base
  })
  names(cols) <- names(df)
  list(n_rows = nrow(df), n_cols = ncol(df), columns = cols)
}

#' Suggest an analysis plan from a dataset profile
#'
#' Mirrors the Python `morie.suggest_analysis_plan()`. Inspects the output
#' of [profile_dataset()] and returns plain-English recommendations for
#' candidate analyses.
#'
#' @param profile A list returned by [profile_dataset()].
#'
#' @return Character vector of suggestion strings, one per recommendation.
#' @export
#' @examples
#' suggest_analysis_plan(profile_dataset(iris))
suggest_analysis_plan <- function(profile) {
  if (!is.list(profile) || is.null(profile$columns))
    stop("profile must be a list returned by profile_dataset().", call. = FALSE)

  suggestions <- character(0)
  cols <- profile$columns
  levels <- vapply(cols, `[[`, character(1L), "measurement_level")

  n_binary  <- sum(levels == "binary")
  n_numeric <- sum(levels %in% c("ratio", "interval"))
  n_nominal <- sum(levels == "nominal")
  n_ordinal <- sum(levels == "ordinal")

  if (n_binary >= 1L && n_numeric >= 1L)
    suggestions <- c(suggestions,
      "Binary outcome + numeric predictors detected. Logistic regression (glm with family=binomial) is appropriate.")
  if (n_binary >= 2L)
    suggestions <- c(suggestions,
      "Two or more binary variables. Consider chi-square test, Fisher's exact test, or risk-difference / odds-ratio CIs.")
  if (n_numeric >= 2L)
    suggestions <- c(suggestions,
      "Multiple numeric variables. Consider linear regression (lm), Pearson correlation, or principal-components analysis.")
  if (n_nominal >= 1L && n_numeric >= 1L)
    suggestions <- c(suggestions,
      "Nominal grouping + numeric outcome. Consider one-way ANOVA, Kruskal-Wallis, or per-group descriptives.")
  if (n_ordinal >= 1L)
    suggestions <- c(suggestions,
      "Ordinal variable detected. Consider Spearman or Kendall's tau correlation, or proportional-odds models.")

  any_missing <- any(vapply(cols, function(c) c$n_missing > 0L, logical(1L)))
  if (any_missing)
    suggestions <- c(suggestions,
      "Missing values present. Consider multiple imputation (e.g. mice) or complete-case sensitivity analysis.")

  if (length(suggestions) == 0L)
    suggestions <- "No standard analysis pattern triggered. Inspect the profile manually."

  suggestions
}
