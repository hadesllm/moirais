escape_regex <- function(x) {
  gsub("([][{}()+*^$.|\\\\?])", "\\\\\\1", x)
}

#' Validate outputs manifest structure
#'
#' @param manifest Data frame to validate.
#' @param strict If `TRUE`, stop on validation failures.
#' @return `TRUE` when validation passes.
#' @export
validate_outputs_manifest <- function(manifest, strict = TRUE) {
  required <- c("output", "public_path", "size_kb", "modified")

  fail <- function(msg) {
    if (isTRUE(strict)) stop(msg, call. = FALSE)
    warning(msg, call. = FALSE)
    FALSE
  }

  if (!is.data.frame(manifest)) {
    return(fail("Manifest must be a data.frame."))
  }

  missing_cols <- setdiff(required, names(manifest))
  if (length(missing_cols) > 0) {
    return(fail(paste0("Manifest missing required columns: ", paste(missing_cols, collapse = ", "))))
  }

  if (anyDuplicated(manifest$output) > 0) {
    return(fail("Manifest column `output` contains duplicates."))
  }

  TRUE
}

#' Read outputs manifest from a project
#'
#' @param project_root Project root path.
#' @param manifest_path Optional explicit manifest path.
#' @param validate If `TRUE`, validate schema.
#' @return Manifest data frame.
#' @export
read_outputs_manifest <- function(project_root = NULL, manifest_path = NULL, validate = TRUE) {
  paths <- moirais_paths(project_root)
  path <- manifest_path %||% paths$outputs_manifest

  if (!file.exists(path)) {
    stop("Outputs manifest not found: ", path, call. = FALSE)
  }

  manifest <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (isTRUE(validate)) {
    validate_outputs_manifest(manifest, strict = TRUE)
  }
  manifest
}

#' Build an outputs manifest from a directory of artifacts
#'
#' @param output_dir Directory containing output files.
#' @param manifest_path CSV path to write.
#' @param public_prefix Prefix used in `public_path` values.
#' @param extensions File extensions to include (without dots).
#' @return Manifest data frame.
#' @export
build_outputs_manifest <- function(
  output_dir,
  manifest_path,
  public_prefix = "data/manifest/outputs",
  extensions = c("csv", "pdf", "png", "html", "txt", "md")
) {
  if (!dir.exists(output_dir)) {
    stop("Output directory does not exist: ", output_dir, call. = FALSE)
  }

  output_dir <- normalizePath(output_dir, winslash = "/", mustWork = TRUE)
  all_files <- list.files(output_dir, recursive = TRUE, full.names = TRUE, include.dirs = FALSE)

  ext_pattern <- paste0("\\.(", paste(extensions, collapse = "|"), ")$")
  keep <- grepl(ext_pattern, tolower(all_files))
  files <- all_files[keep]

  manifest <- data.frame(
    output = character(),
    public_path = character(),
    size_kb = character(),
    modified = character(),
    stringsAsFactors = FALSE
  )

  if (length(files) > 0) {
    root_rx <- paste0("^", escape_regex(output_dir), "/?")
    rel <- sub(root_rx, "", normalizePath(files, winslash = "/", mustWork = FALSE))

    info <- file.info(files)
    manifest <- data.frame(
      output = rel,
      public_path = file.path(public_prefix, rel),
      size_kb = sprintf("%.1f", as.numeric(info$size) / 1024),
      modified = format(info$mtime, "%Y-%m-%d %H:%M:%S"),
      stringsAsFactors = FALSE
    )
    manifest <- manifest[order(manifest$output), , drop = FALSE]
  }

  dir.create(dirname(manifest_path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(manifest, manifest_path, row.names = FALSE)
  manifest
}

#' Audit declared outputs against files on disk
#'
#' @param project_root Project root directory.
#' @param manifest Manifest data frame. If `NULL`, loaded from disk.
#' @return Data frame containing declared and observed output status.
#' @export
audit_public_outputs <- function(project_root = NULL, manifest = NULL) {
  paths <- moirais_paths(project_root)
  manifest <- manifest %||% read_outputs_manifest(paths$project_root)
  validate_outputs_manifest(manifest, strict = TRUE)

  to_abs <- function(p) {
    ifelse(is_absolute_path(p), p, file.path(paths$project_root, p))
  }

  declared_abs <- normalizePath(to_abs(manifest$public_path), winslash = "/", mustWork = FALSE)
  declared_info <- file.info(declared_abs)
  declared_exists <- file.exists(declared_abs)

  declared_tbl <- data.frame(
    output = manifest$output,
    public_path = manifest$public_path,
    declared = TRUE,
    exists = declared_exists,
    size_kb_manifest = suppressWarnings(as.numeric(manifest$size_kb)),
    size_kb_actual = ifelse(declared_exists, round(as.numeric(declared_info$size) / 1024, 1), NA_real_),
    size_diff_kb = ifelse(
      declared_exists,
      round((as.numeric(declared_info$size) / 1024) - suppressWarnings(as.numeric(manifest$size_kb)), 1),
      NA_real_
    ),
    modified_manifest = manifest$modified,
    modified_actual = ifelse(
      declared_exists,
      format(declared_info$mtime, "%Y-%m-%d %H:%M:%S"),
      NA_character_
    ),
    stringsAsFactors = FALSE
  )

  unexpected_tbl <- data.frame(
    output = character(),
    public_path = character(),
    declared = logical(),
    exists = logical(),
    size_kb_manifest = numeric(),
    size_kb_actual = numeric(),
    size_diff_kb = numeric(),
    modified_manifest = character(),
    modified_actual = character(),
    stringsAsFactors = FALSE
  )

  if (dir.exists(paths$outputs_dir)) {
    actual_rel <- list.files(paths$outputs_dir, recursive = TRUE, full.names = FALSE, include.dirs = FALSE)
    unexpected <- setdiff(actual_rel, unique(manifest$output))

    if (length(unexpected) > 0) {
      unexpected_abs <- file.path(paths$outputs_dir, unexpected)
      unexpected_info <- file.info(unexpected_abs)
      unexpected_tbl <- data.frame(
        output = unexpected,
        public_path = file.path("data/manifest/outputs", unexpected),
        declared = FALSE,
        exists = TRUE,
        size_kb_manifest = NA_real_,
        size_kb_actual = round(as.numeric(unexpected_info$size) / 1024, 1),
        size_diff_kb = NA_real_,
        modified_manifest = NA_character_,
        modified_actual = format(unexpected_info$mtime, "%Y-%m-%d %H:%M:%S"),
        stringsAsFactors = FALSE
      )
    }
  }

  out <- rbind(declared_tbl, unexpected_tbl)
  out[order(-as.integer(out$declared), out$output), , drop = FALSE]
}

#' Summarize an output audit
#'
#' @param audit_tbl Result from [audit_public_outputs()].
#' @return Named list with high-level diagnostics.
#' @export
summarize_output_audit <- function(audit_tbl) {
  if (!is.data.frame(audit_tbl)) {
    stop("`audit_tbl` must be a data.frame.", call. = FALSE)
  }
  if (!all(c("declared", "exists") %in% names(audit_tbl))) {
    stop("`audit_tbl` must include `declared` and `exists` columns.", call. = FALSE)
  }

  total_declared <- sum(audit_tbl$declared)
  declared_present <- sum(audit_tbl$declared & audit_tbl$exists)
  declared_missing <- sum(audit_tbl$declared & !audit_tbl$exists)
  unexpected_files <- sum(!audit_tbl$declared & audit_tbl$exists)

  list(
    total_declared = total_declared,
    declared_present = declared_present,
    declared_missing = declared_missing,
    unexpected_files = unexpected_files,
    declared_present_pct = if (total_declared > 0) round(100 * declared_present / total_declared, 1) else NA_real_
  )
}
