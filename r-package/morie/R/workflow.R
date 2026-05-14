#' Default workflow step map
#'
#' Returns the default named map of workflow steps to project script paths.
#'
#' @return Named character vector.
#' @export
default_workflow_map <- function() {
  c(
    modules = "libexec/config/tests/rtests/run_modules.R",
    publish = "libexec/config/tests/rtests/publish_public_artifacts.R",
    render = "libexec/config/tests/rtests/render_site.R",
    readiness = "libexec/config/tests/rtests/check_connect_publish_readiness.R"
  )
}

validate_workflow_map <- function(script_map) {
  if (!is.character(script_map) || is.null(names(script_map))) {
    stop("`script_map` must be a named character vector.", call. = FALSE)
  }
  if (anyNA(names(script_map)) || any(!nzchar(names(script_map)))) {
    stop("`script_map` has empty step names.", call. = FALSE)
  }
  if (anyNA(script_map) || any(!nzchar(script_map))) {
    stop("`script_map` has empty script paths.", call. = FALSE)
  }
  if (anyDuplicated(names(script_map)) > 0) {
    stop("`script_map` step names must be unique.", call. = FALSE)
  }
  script_map
}

#' Run one project workflow step
#'
#' @param step Step name present in `script_map`.
#' @param project_root Project root directory.
#' @param script_map Named character vector mapping steps to script paths.
#' @param rscript_bin Optional path to `Rscript` binary.
#' @param verbose If `TRUE`, streams command output.
#' @return Named list with step metadata and exit status.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
run_workflow_step <- function(
  step,
  project_root = NULL,
  script_map = default_workflow_map(),
  rscript_bin = file.path(R.home("bin"), "Rscript"),
  verbose = TRUE
) {
  script_map <- validate_workflow_map(script_map)

  if (missing(step) || length(step) != 1 || !nzchar(step)) {
    stop("Provide exactly one workflow `step`.", call. = FALSE)
  }
  if (!(step %in% names(script_map))) {
    stop(
      "Unknown step: ", step,
      ". Valid steps are: ", paste(names(script_map), collapse = ", "),
      call. = FALSE
    )
  }

  paths <- morie_paths(project_root)
  script_rel <- script_map[[step]]
  script_abs <- file.path(paths$project_root, script_rel)

  if (!file.exists(script_abs)) {
    stop("Workflow script not found: ", script_abs, call. = FALSE)
  }
  if (!nzchar(Sys.which(rscript_bin)) && !file.exists(rscript_bin)) {
    stop("Rscript binary not found: ", rscript_bin, call. = FALSE)
  }

  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(paths$project_root)

  status <- system2(
    rscript_bin,
    script_rel,
    stdout = if (isTRUE(verbose)) "" else FALSE,
    stderr = if (isTRUE(verbose)) "" else FALSE
  )

  list(
    step = step,
    script = script_abs,
    status = as.integer(status)
  )
}

#' Run multiple workflow steps
#'
#' @param steps Ordered vector of workflow step names.
#' @param project_root Project root directory.
#' @param script_map Named character vector mapping steps to script paths.
#' @param stop_on_error If `TRUE`, stop at first failure.
#' @param verbose If `TRUE`, streams command output.
#' @return Data frame of step statuses.
#' @export
run_pipeline <- function(
  steps = NULL,
  project_root = NULL,
  script_map = default_workflow_map(),
  stop_on_error = TRUE,
  verbose = TRUE
) {
  script_map <- validate_workflow_map(script_map)

  if (is.null(steps)) {
    steps <- names(script_map)
  }
  if (!is.character(steps) || length(steps) == 0) {
    stop("`steps` must be a non-empty character vector.", call. = FALSE)
  }

  bad <- setdiff(steps, names(script_map))
  if (length(bad) > 0) {
    stop("Unknown steps: ", paste(bad, collapse = ", "), call. = FALSE)
  }

  out <- vector("list", length(steps))
  for (i in seq_along(steps)) {
    step <- steps[[i]]
    result <- tryCatch(
      run_workflow_step(
        step = step,
        project_root = project_root,
        script_map = script_map,
        verbose = verbose
      ),
      error = function(e) list(step = step, script = NA_character_, status = 1L, error = conditionMessage(e))
    )

    out[[i]] <- data.frame(
      step = result$step,
      script = result$script,
      status = result$status,
      error = result$error %||% NA_character_,
      stringsAsFactors = FALSE
    )

    if (isTRUE(stop_on_error) && !identical(result$status, 0L)) {
      break
    }
  }

  do.call(rbind, out)
}
