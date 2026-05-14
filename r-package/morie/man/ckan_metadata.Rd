\name{ckan_metadata}
\alias{ckan_metadata}
\docType{data}
\title{CKAN dataset-source metadata}
\description{
  Metadata table for the CKAN-published datasets that
  \pkg{morie} can fetch and analyse, keyed by dataset short-name.
}
\usage{
  data(ckan_metadata)
}
\format{
  A data frame. One row per registered CKAN dataset with columns
  describing the publishing agency, resource id, and last refresh date.
}
\keyword{datasets}
\examples{
\dontrun{
  # See the package vignettes for usage examples:
  #   vignette(package = "morie")
}
}
