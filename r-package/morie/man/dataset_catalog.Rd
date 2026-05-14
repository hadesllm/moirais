\name{dataset_catalog}
\alias{dataset_catalog}
\docType{data}
\title{Built-in MORIE dataset catalog}
\description{
  Table of the datasets that ship with \pkg{morie} (Canadian
  carceral, police, and oversight data plus general epidemiological
  reference data). Mirrors the Python-side \code{DATASET_CATALOG}.
}
\usage{
  data(dataset_catalog)
}
\format{
  A data frame keyed by dataset short-name with columns describing
  source, survey, year, file path, and CKAN resource identifier.
}
\keyword{datasets}
\examples{
\dontrun{
  # See the package vignettes for usage examples:
  #   vignette(package = "morie")
}
}
