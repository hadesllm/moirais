# SPDX-License-Identifier: MIT OR Apache-2.0

#' Linear LR warmup (Vaswani 2017)
#'
#' @param x Numeric vector of training steps.
#' @param lr_target Numeric target learning rate (default 1e-3).
#' @param warmup_steps Integer warmup steps (default 1000).
#' @return Named list with tensor, value, lr_target, warmup_steps, step, method.
#' @keywords internal
lr_warmup <- function(x, lr_target = 1e-3, warmup_steps = 1000L) {
  if (warmup_steps <= 0) stop("warmup_steps must be > 0")
  t <- as.numeric(x)
  lr <- lr_target * pmin(1, t / warmup_steps)
  list(tensor = lr, value = lr[1L],
       lr_target = lr_target, warmup_steps = warmup_steps,
       step = t, method = "linear-warmup")
}
