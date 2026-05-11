# Internal infix helper for defaults.
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) return(y)
  if (length(x) == 1 && (is.na(x) || identical(x, ""))) return(y)
  x
}

is_absolute_path <- function(path) {
  grepl("^(/|[A-Za-z]:[/\\\\])", path)
}

#' Find a project root directory
#'
#' Searches upward from `start` for a directory containing the current
#' Sphinx/package-root markers, while still tolerating legacy Quarto-era
#' markers in older checkouts.
#'
#' @param start Starting directory.
#' @param max_up Maximum number of parent traversals.
#' @return Absolute path to the detected project root.
#' @export
find_project_root <- function(start = getwd(), max_up = 10L) {
  current <- normalizePath(start, winslash = "/", mustWork = FALSE)

  for (i in seq_len(max_up)) {
    has_pyproject <- file.exists(file.path(current, "pyproject.toml"))
    has_libexec <- dir.exists(file.path(current, "libexec", "config"))
    has_sphinx <- dir.exists(file.path(current, "docs", "source"))
    if (has_pyproject && (has_libexec || has_sphinx)) {
      return(current)
    }

    parent <- dirname(current)
    if (identical(parent, current)) break
    current <- parent
  }

  stop(
    "Unable to detect project root. Provide `project_root` explicitly.",
    call. = FALSE
  )
}

#' Resolve standard project paths
#'
#' @param project_root Project root directory. If `NULL`, inferred from the
#'   current working directory.
#' @return Named list of key paths.
#' @export
morie_paths <- function(project_root = NULL) {
  root <- project_root %||% find_project_root()
  root <- normalizePath(root, winslash = "/", mustWork = FALSE)

  list(
    project_root = root,
    data_dir = file.path(root, "data"),
    cache_dir = file.path(root, "data", "cache"),
    datasets_dir = file.path(root, "data", "datasets"),
    outputs_dir = file.path(root, "data", "manifest", "outputs"),
    outputs_manifest = file.path(root, "data", "manifest", "outputs_manifest.csv"),
    rtests_dir = file.path(root, "libexec", "config", "tests", "rtests"),
    pytests_dir = file.path(root, "libexec", "config", "tests", "pytests"),
    tools_dir = file.path(root, "libexec", "config", "tools"),
    docs_dir = file.path(root, "docs")
  )
}
