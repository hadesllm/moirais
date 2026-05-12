# SPDX-License-Identifier: GPL-2.0-only

#' IV Wald estimator for LATE (Imbens-Angrist)
#'
#' @param x Numeric covariates (unused at present; kept for API parity).
#' @param y Numeric outcome vector.
#' @param z Numeric instrument vector.
#' @param treatment Integer/logical treatment indicator.
#' @return Named list with estimate, se, first_stage, reduced_form, n, method.
#' @keywords internal
hrzt2 <- function(x, y, z, treatment) {
  y <- as.numeric(y); z <- as.numeric(z); D <- as.numeric(treatment)
  n <- length(y)
  if (n < 20 || length(z) != n || length(D) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "LATE (insufficient data)"))
  uniq <- unique(z)
  z_bin <- if (length(uniq) > 2) as.numeric(z > stats::median(z)) else
    as.numeric(z == max(uniq))
  n1 <- sum(z_bin > 0.5); n0 <- sum(z_bin < 0.5)
  if (n1 < 5 || n0 < 5)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "LATE (one arm of Z empty)"))
  Y1 <- mean(y[z_bin > 0.5]); Y0 <- mean(y[z_bin < 0.5])
  D1 <- mean(D[z_bin > 0.5]); D0 <- mean(D[z_bin < 0.5])
  num <- Y1 - Y0; den <- D1 - D0
  if (abs(den) < 1e-8)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "LATE (weak instrument)"))
  late <- num / den
  vY <- stats::var(y[z_bin > 0.5]) / n1 + stats::var(y[z_bin < 0.5]) / n0
  vD <- stats::var(D[z_bin > 0.5]) / n1 + stats::var(D[z_bin < 0.5]) / n0
  v_late <- (vY + late^2 * vD) / den^2
  list(estimate = as.numeric(late), se = sqrt(max(v_late, 0)),
       first_stage = as.numeric(den), reduced_form = as.numeric(num),
       n = n,
       method = "IV Wald estimator (Imbens-Angrist LATE)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzt2
#' @keywords internal
#' @export
horowitz_local_ate <- hrzt2
