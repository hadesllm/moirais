# SPDX-License-Identifier: MIT OR Apache-2.0
#' Load DMT-imaging dataset (Timmermann et al. 2023)
#'
#' R parity of \code{morie.entheo.data.load_dmt_imaging}.  Reads the
#' Carhart-Harris / Timmermann DMT-imaging dataset (20 subjects EEG +
#' parcellated fMRI; 15 motion-survived) from a local mirror.  Falls
#' back to a reproducible synthetic record when the dataset is absent
#' or when \pkg{R.matlab} is not installed, so downstream code keeps
#' running in CI.
#'
#' @param subject_id integer or character (zero-padded 2-digit).
#'   \code{NULL} means "all available subjects on disk".
#' @param root character.  Override path to the DMT_Imaging mirror.
#'   Defaults to \code{$MORIE_DMT_IMAGING_ROOT} or
#'   \code{/Volumes/VSR/rootcoderfiles/DMT_Imaging}.
#' @return Named list with components:
#' \itemize{
#'   \item \code{records}   list of subject records (eeg, fmri, behavioural).
#'   \item \code{root}      resolved dataset root path (or NA).
#'   \item \code{synthetic} TRUE if any record is the fallback.
#'   \item \code{warnings}  character vector of advisories.
#' }
#' @references
#' Timmermann, C. et al. (2023). Neural correlates of the DMT
#'   experience assessed with multivariate EEG. Scientific Reports.
#' @keywords internal
load_dmt_imaging <- function(subject_id = NULL, root = NULL) {
  default_root <- Sys.getenv("MORIE_DMT_IMAGING_ROOT",
                             "/Volumes/VSR/rootcoderfiles/DMT_Imaging")
  if (is.null(root)) root <- default_root
  root_exists <- dir.exists(root)
  warnings_vec <- character(0)

  default_subjects <- sprintf("%02d", c(1:3, 6:17))

  if (is.null(subject_id)) {
    subs <- .entheo_list_subjects(root)
    if (length(subs) == 0L) subs <- default_subjects
  } else {
    subs <- sprintf("%02d", as.integer(subject_id))
  }

  if (!root_exists) {
    warnings_vec <- c(
      sprintf("DMT_Imaging root not found at %s; using synthetic fixture.", root),
      warnings_vec)
  }

  records <- vector("list", length(subs))
  any_synth <- FALSE
  for (i in seq_along(subs)) {
    sid <- subs[i]
    rec <- NULL
    if (root_exists) {
      rec <- .entheo_load_real(sid, root)
      if (is.null(rec)) {
        warnings_vec <- c(warnings_vec, sprintf(
          "subject %s: .mat files present but unloadable (install R.matlab)", sid))
      }
    }
    if (is.null(rec)) {
      rec <- .entheo_synthetic_record(sid)
      any_synth <- TRUE
    }
    records[[i]] <- rec
  }

  list(
    records   = records,
    root      = if (root_exists) root else NA_character_,
    synthetic = any_synth,
    subject_ids = subs,
    warnings  = warnings_vec
  )
}


#' @keywords internal
.entheo_list_subjects <- function(root) {
  fmri_dir <- file.path(root, "fMRI")
  if (!dir.exists(fmri_dir)) return(character(0))
  files <- list.files(fmri_dir, pattern = "^LongS\\d{2}(DMT|PCB)\\.mat$")
  ids <- unique(substr(files, 6L, 7L))
  sort(ids)
}


#' @keywords internal
.entheo_synthetic_record <- function(subject_id,
                                     n_tp = 480L, n_chan = 32L,
                                     n_parcels = 100L) {
  set.seed(7L + as.integer(subject_id))
  list(
    subject_id = subject_id,
    condition_order = c("DMT", "PCB"),
    eeg = list(
      sfreq    = 250,
      channels = sprintf("E%02d", seq_len(n_chan)),
      data_dmt = matrix(stats::rnorm(n_chan * n_tp), n_chan, n_tp),
      data_pcb = matrix(stats::rnorm(n_chan * n_tp), n_chan, n_tp)
    ),
    fmri = list(
      tr        = 2.0,
      n_parcels = n_parcels,
      data_dmt  = matrix(stats::rnorm(n_parcels * (n_tp %/% 4L)),
                         n_parcels, n_tp %/% 4L),
      data_pcb  = matrix(stats::rnorm(n_parcels * (n_tp %/% 4L)),
                         n_parcels, n_tp %/% 4L),
      motion_fd_mm = stats::runif(n_tp %/% 4L, 0, 0.6)
    ),
    behavioural = list(
      intensity_dmt = pmin(pmax(stats::rnorm(12, 7.0, 1.5), 0), 10),
      intensity_pcb = pmin(pmax(stats::rnorm(12, 0.8, 0.6), 0), 10)
    ),
    .synthetic = TRUE
  )
}


#' @keywords internal
.entheo_load_real <- function(subject_id, root) {
  if (!requireNamespace("R.matlab", quietly = TRUE)) return(NULL)
  f_dmt <- file.path(root, "fMRI", sprintf("LongS%sDMT.mat", subject_id))
  f_pcb <- file.path(root, "fMRI", sprintf("LongS%sPCB.mat", subject_id))
  if (!(file.exists(f_dmt) && file.exists(f_pcb))) return(NULL)
  blob_dmt <- tryCatch(R.matlab::readMat(f_dmt), error = function(e) NULL)
  blob_pcb <- tryCatch(R.matlab::readMat(f_pcb), error = function(e) NULL)
  if (is.null(blob_dmt) || is.null(blob_pcb)) return(NULL)

  .pick_largest <- function(blob) {
    best <- NULL
    for (v in blob) {
      if (is.matrix(v) && is.numeric(v)) {
        if (is.null(best) || length(v) > length(best)) best <- v
      }
    }
    if (is.null(best)) matrix(0, 1, 1) else best
  }

  arr_dmt <- .pick_largest(blob_dmt)
  arr_pcb <- .pick_largest(blob_pcb)
  list(
    subject_id = subject_id,
    condition_order = c("DMT", "PCB"),
    eeg = list(sfreq = NA_real_, channels = character(0),
               data_dmt = NULL, data_pcb = NULL),
    fmri = list(tr = 2.0, n_parcels = nrow(arr_dmt),
                data_dmt = arr_dmt, data_pcb = arr_pcb,
                motion_fd_mm = NULL),
    behavioural = list(),
    .synthetic = FALSE,
    .paths = list(fmri_dmt = f_dmt, fmri_pcb = f_pcb)
  )
}
