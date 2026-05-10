#' moirais: Methods for Observational Inference and Robust Analysis
#'
#' Multi-domain scientific computing toolkit for observational
#' inference. Hosts the MRM (McNamara-Ruhela-Medina) framework as a
#' primary application for Canadian carceral, police, and oversight
#' data, with general-purpose causal inference, signal-processing,
#' cryptographic, spatial-statistics, statistical-physics, and
#' psychometrics modules.
#'
#' @keywords internal
#' @aliases moirais-package
#' @importFrom stats aggregate ave deviance median na.omit plogis
#'   setNames update weighted.mean
"_PACKAGE"

# `weight` is referenced as a data-frame column name inside
# non-standard-evaluation contexts (e.g. `subset()`, `transform()`,
# `with()`); silence the R CMD check NOTE without weakening the rest
# of the codetools analysis.
utils::globalVariables(c("weight"))
