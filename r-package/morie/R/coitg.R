# SPDX-License-Identifier: GPL-2.0-only

#' Engle-Granger two-step cointegration test
#'
#' @param y1 Numeric, first series.
#' @param y2 Numeric, second series.
#' @param max_lag Max ADF augmentation lags. Default \code{floor(12*(n/100)^0.25)}.
#' @return Named list with \code{adf_statistic, p_value, beta, n, method}.
#' @export
eg_coint <- function(y1, y2, max_lag = NULL) {
  y1 <- as.numeric(y1); y2 <- as.numeric(y2)
  if (length(y1) != length(y2)) stop("Length mismatch.")
  n <- length(y1); if (n < 20) stop("Need >=20 obs.")
  if (is.null(max_lag)) max_lag <- floor(12 * (n / 100)^0.25)
  fit_ls <- lm(y1 ~ y2)
  beta <- coef(fit_ls)
  resid <- residuals(fit_ls)
  if (requireNamespace("urca", quietly = TRUE)) {
    adf <- urca::ur.df(resid, type = "none", lags = max_lag,
                       selectlags = "AIC")
    stat <- as.numeric(adf@teststat[1])
  } else {
    # Plain ADF-style t-stat on residuals.
    dr <- diff(resid); T <- length(dr) - max_lag
    dep <- dr[(max_lag + 1):length(dr)]
    Xr <- resid[(max_lag + 1):length(resid) - 1]
    Xr <- cbind(Xr)
    if (max_lag >= 1) {
      for (i in seq_len(max_lag))
        Xr <- cbind(Xr, dr[(max_lag + 1 - i):(length(dr) - i)])
    }
    b <- lsfit(Xr, dep, intercept = FALSE)
    e <- dep - Xr %*% b$coef
    sig2 <- sum(e^2) / (T - ncol(Xr))
    se <- sqrt(sig2 * solve(crossprod(Xr))[1, 1])
    stat <- b$coef[1] / se
  }
  crit <- c(`1%` = -3.90, `5%` = -3.34, `10%` = -3.04)
  approx_p <- if (stat < crit["1%"]) 0.005 else
              if (stat < crit["5%"]) 0.03 else
              if (stat < crit["10%"]) 0.07 else
              min(1, 2 * pnorm(stat))
  list(adf_statistic = as.numeric(stat),
       p_value = as.numeric(approx_p),
       beta = unname(beta),
       critical_values = crit,
       n = n,
       method = "Engle-Granger 2-step cointegration (Engle & Granger 1987)")
}
