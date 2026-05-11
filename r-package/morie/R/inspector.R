#' Inspect a serialised analysis output (JSON, CSV, or RDS)
#'
#' Mirrors the Python `morie.inspect_output()`. Reads a structured output
#' file and returns a brief summary of its contents.
#'
#' Supported formats: `.json` (via `jsonlite`), `.csv` (via base
#' `utils::read.csv`), `.rds` (via `base::readRDS`).
#'
#' @param path Path to a JSON, CSV, or RDS file.
#'
#' @return A list with components `path`, `format`, `exists`, `size_bytes`,
#'   and (on success) `contents_preview` plus type-appropriate metadata.
#' @export
#' @examples
#' tmp <- tempfile(fileext = ".json")
#' if (requireNamespace("jsonlite", quietly = TRUE)) {
#'   jsonlite::write_json(list(estimate = 0.123, se = 0.045), tmp)
#'   inspect_output(tmp)
#'   unlink(tmp)
#' }
inspect_output <- function(path) {
  result <- list(
    path        = path,
    format      = tools::file_ext(path),
    exists      = file.exists(path),
    size_bytes  = if (file.exists(path)) file.info(path)$size else 0L
  )
  if (!result$exists) {
    result$status <- "missing"
    return(result)
  }
  ext <- tolower(tools::file_ext(path))
  result$contents_preview <- tryCatch({
    if (ext == "json") {
      if (!requireNamespace("jsonlite", quietly = TRUE)) {
        result$status <- "jsonlite-unavailable"
        return(result)
      }
      obj <- jsonlite::fromJSON(path)
      if (is.list(obj)) names(obj) else utils::head(obj)
    } else if (ext == "csv") {
      df <- utils::read.csv(path, nrows = 5L)
      result$n_columns <- ncol(df)
      utils::head(df)
    } else if (ext == "rds") {
      obj <- readRDS(path)
      result$class <- class(obj)
      if (is.data.frame(obj)) utils::head(obj) else utils::head(names(obj))
    } else {
      result$status <- paste0("unsupported-extension: ", ext)
      return(result)
    }
  }, error = function(e) {
    result$status <- paste0("read-error: ", conditionMessage(e))
    NULL
  })
  result$status <- "ok"
  result
}

#' Verify that a serialised statistical output meets minimum quality gates
#'
#' Mirrors the Python `morie.verify_statistical_output()`. Runs a small
#' suite of sanity checks on a JSON output containing fields commonly used
#' across MORIE estimators: `ate`, `se`, `ci_lower`, `ci_upper`, `n`,
#' `p_value`. Each check is a named boolean; the verification passes if all
#' checks are `TRUE`.
#'
#' Checks: SE non-negative; CI lower < CI upper; estimate inside the CI;
#' n positive; p-value (if present) in \[0, 1\]; estimate finite.
#'
#' @param path Path to a JSON output file.
#'
#' @return A list with `path`, `passed` (logical), and `checks` (named list
#'   of boolean check results).
#' @export
#' @examples
#' tmp <- tempfile(fileext = ".json")
#' if (requireNamespace("jsonlite", quietly = TRUE)) {
#'   jsonlite::write_json(
#'     list(ate = 0.5, se = 0.1, ci_lower = 0.3, ci_upper = 0.7, n = 200),
#'     tmp, auto_unbox = TRUE
#'   )
#'   verify_statistical_output(tmp)
#'   unlink(tmp)
#' }
verify_statistical_output <- function(path) {
  out <- list(path = path, passed = FALSE, checks = list())
  if (!file.exists(path)) {
    out$checks$file_exists <- FALSE
    return(out)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE))
    stop("jsonlite is required for verify_statistical_output().",
         call. = FALSE)

  obj <- tryCatch(jsonlite::fromJSON(path), error = function(e) NULL)
  if (is.null(obj)) {
    out$checks$json_parses <- FALSE
    return(out)
  }
  out$checks$json_parses <- TRUE

  has <- function(k) !is.null(obj[[k]]) && length(obj[[k]]) >= 1L

  if (has("se"))
    out$checks$se_nonneg <- isTRUE(as.numeric(obj$se) >= 0)
  if (has("ci_lower") && has("ci_upper"))
    out$checks$ci_ordered <- isTRUE(as.numeric(obj$ci_lower) <
                                    as.numeric(obj$ci_upper))
  if (has("ate") && has("ci_lower") && has("ci_upper"))
    out$checks$estimate_in_ci <-
      isTRUE(as.numeric(obj$ate) >= as.numeric(obj$ci_lower) &&
             as.numeric(obj$ate) <= as.numeric(obj$ci_upper))
  if (has("n"))
    out$checks$n_positive <- isTRUE(as.numeric(obj$n) > 0)
  if (has("p_value")) {
    p <- as.numeric(obj$p_value)
    out$checks$p_in_unit <- isTRUE(p >= 0 && p <= 1)
  }
  if (has("ate"))
    out$checks$estimate_finite <- isTRUE(is.finite(as.numeric(obj$ate)))

  out$passed <- length(out$checks) > 0L &&
                all(unlist(out$checks))
  out
}
