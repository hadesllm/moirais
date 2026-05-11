#' Estimate the ATE via the Interactive Regression Model (IRM)
#'
#' Thin R wrapper that dispatches to the CRAN `DoubleML` package's
#' \code{DoubleML::DoubleMLIRM} \link[R6]{R6Class}, mirroring the Python sibling
#' `morie.estimate_irm()` (which dispatches to the Python `DoubleML` package).
#'
#' Following the DoubleML R package's own conventions, this uses
#' the `mlr3` ecosystem for the nuisance learners (\code{ml_g} for
#' \eqn{E[Y|T,X]} and \code{ml_m} for \eqn{P(T=1|X)}). Defaults are
#' `lrn("regr.lm")` and `lrn("classif.log_reg")`, which require nothing
#' beyond `stats`. For higher-capacity defaults, install `ranger` and pass
#' `lrn("regr.ranger")` / `lrn("classif.ranger")` via the underlying
#' `DoubleML::DoubleMLIRM$new()` directly.
#'
#' Following Chernozhukov et al. (2018), the IRM extends the partially linear
#' model by allowing fully heterogeneous treatment effects:
#' \deqn{Y = g_0(T, X) + U,\quad E[U|T,X] = 0}
#' \deqn{T = m_0(X) + V,\quad E[V|X] = 0}
#'
#' @param data A `data.frame` containing outcome, treatment, and covariates.
#' @param treatment Column name of the binary treatment.
#' @param outcome Column name of the outcome.
#' @param covariates Character vector of covariate column names.
#' @param n_folds Number of cross-fitting folds (default 5).
#' @param random_state Random seed (default 42).
#'
#' @return A list with components: `ate`, `se`, `ci_lower`, `ci_upper`,
#'   `n`, `method` (`"IRM (DoubleML)"`).
#'
#' @section CRAN \code{Suggests}:
#' Requires the suggested packages `DoubleML`, `mlr3`, and `mlr3learners`.
#' Install with `install.packages(c("DoubleML", "mlr3", "mlr3learners"))`.
#' If any are unavailable, the function raises an informative error.
#'
#' @references
#' Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
#' Newey, W., & Robins, J. (2018). Double/debiased machine learning for
#' treatment and structural parameters. \emph{The Econometrics Journal}, 21(1),
#' C1--C68. \doi{10.1111/ectj.12097}
#'
#' Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2024). DoubleML --
#' An object-oriented implementation of double machine learning in R.
#' \emph{Journal of Statistical Software}, 108(3). \doi{10.18637/jss.v108.i03}
#'
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("DoubleML", quietly = TRUE) &&
#'     requireNamespace("mlr3", quietly = TRUE) &&
#'     requireNamespace("mlr3learners", quietly = TRUE)) {
#'   set.seed(1)
#'   n <- 200
#'   X <- matrix(rnorm(n * 5), n, 5)
#'   ps <- plogis(X[, 1] - X[, 2])
#'   T <- rbinom(n, 1, ps)
#'   Y <- 0.5 * T + X[, 1] + rnorm(n)
#'   df <- data.frame(Y = Y, T = T, X)
#'   estimate_irm(df, treatment = "T", outcome = "Y",
#'                covariates = paste0("X", 1:5))
#' }
#' }
estimate_irm <- function(data, treatment, outcome, covariates,
                         n_folds = 5, random_state = 42) {
  for (pkg in c("DoubleML", "mlr3", "mlr3learners")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf(
        "Package %s is required for estimate_irm(). Install with: install.packages(%s)",
        sQuote(pkg), sQuote(pkg)
      ), call. = FALSE)
    }
  }

  cols <- c(treatment, outcome, covariates)
  frame <- stats::na.omit(data[, cols, drop = FALSE])

  for (col in covariates) {
    if (!is.numeric(frame[[col]]))
      frame[[col]] <- as.numeric(as.factor(frame[[col]]))
  }

  set.seed(random_state)

  dml_data <- DoubleML::DoubleMLData$new(
    data = data.table::as.data.table(frame),
    y_col = outcome,
    d_cols = treatment,
    x_cols = covariates
  )

  ml_g <- mlr3::lrn("regr.lm")
  ml_m <- mlr3::lrn("classif.log_reg")

  dml_irm <- DoubleML::DoubleMLIRM$new(dml_data, ml_g, ml_m, n_folds = n_folds)
  dml_irm$fit()

  ate <- as.numeric(dml_irm$coef)[[1L]]
  se  <- as.numeric(dml_irm$se)[[1L]]
  z   <- 1.959964

  list(
    ate      = ate,
    se       = se,
    ci_lower = ate - z * se,
    ci_upper = ate + z * se,
    n        = nrow(frame),
    method   = "IRM (DoubleML)"
  )
}
