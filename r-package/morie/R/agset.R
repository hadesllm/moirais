# SPDX-License-Identifier: MIT OR Apache-2.0

#' Romer-Rosenthal agenda-setter outcome and power (Armstrong Ch 10)
#'
#' Agenda setter offers a take-it-or-leave-it proposal from `options`;
#' legislature accepts iff the proposal beats the reversion at the
#' median voter. Power = distance moved from the status quo.
#'
#' @param options Discrete set of feasible policy proposals.
#' @param setter_ideal Agenda setter's ideal point.
#' @param reversion Status-quo / reversion point.
#' @return Named list with `chosen`, `power`, `setter_ideal`,
#'   `reversion`, `win_set_size`, `win_set_bounds`, `method`.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
agset <- function(options, setter_ideal, reversion) {
  options <- as.numeric(options)
  setter_ideal <- as.numeric(setter_ideal)[1]
  reversion <- as.numeric(reversion)[1]
  if (length(options) == 0L)
    return(list(chosen = NA_real_, power = 0,
                setter_ideal = setter_ideal, reversion = reversion,
                win_set_size = 0L,
                win_set_bounds = c(NA_real_, NA_real_),
                method = "agenda_setter_power"))
  median_voter_pt <- (setter_ideal + reversion) / 2
  win_lo <- min(reversion, 2 * median_voter_pt - reversion)
  win_hi <- max(reversion, 2 * median_voter_pt - reversion)
  in_win <- options >= win_lo & options <= win_hi
  chosen <- if (!any(in_win)) reversion
            else {
              feas <- options[in_win]
              feas[which.min(abs(feas - setter_ideal))]
            }
  list(chosen = chosen, power = abs(chosen - reversion),
       setter_ideal = setter_ideal, reversion = reversion,
       win_set_size = as.integer(sum(in_win)),
       win_set_bounds = c(win_lo, win_hi),
       method = "agenda_setter_power")
}

#' @keywords internal
#' @rdname agset
#' @export
agenda_setter_power <- agset
