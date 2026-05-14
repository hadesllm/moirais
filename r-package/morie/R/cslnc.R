# SPDX-License-Identifier: MIT OR Apache-2.0

#' Cosine LR schedule with warmup (Loshchilov 2017)
#'
#' @param x Numeric vector of training steps.
#' @param lr_max Numeric maximum learning rate (default 1e-3).
#' @param lr_min Numeric minimum learning rate (default 0).
#' @param total_steps Integer total training steps (default 1000).
#' @param warmup_steps Integer warmup steps (default 0).
#' @return Named list with value, tensor, step, lr_max, lr_min,
#'   total_steps, warmup_steps, method.
#' @keywords internal
cosine_lr_schedule <- function(x, lr_max = 1e-3, lr_min = 0,
                                total_steps = 1000L, warmup_steps = 0L) {
  if (total_steps <= warmup_steps)
    stop("total_steps must exceed warmup_steps")
  t <- as.numeric(x)
  warm <- t < warmup_steps
  lr <- numeric(length(t))
  lr[warm] <- lr_max * t[warm] / max(1, warmup_steps)
  dec <- pmin(pmax((t - warmup_steps) /
                     (total_steps - warmup_steps), 0), 1)
  lr[!warm] <- lr_min + 0.5 * (lr_max - lr_min) *
    (1 + cos(pi * dec[!warm]))
  list(value = lr[1L], tensor = lr, step = t,
       lr_max = lr_max, lr_min = lr_min,
       total_steps = total_steps, warmup_steps = warmup_steps,
       method = "cosine-LR")
}
