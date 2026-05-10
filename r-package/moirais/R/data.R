# data.R -- roxygen2 documentation for package data objects

#' MOIRAIS Dataset Catalog
#'
#' A data.frame listing all Canadian public health datasets available
#' through the MOIRAIS data management system. Each row describes one
#' dataset with its source, survey, year, format, and access metadata.
#'
#' @format A data.frame with columns:
#' \describe{
#'   \item{key}{Unique catalog key (e.g., \code{"opencanada_cpads_2021"})}
#'   \item{name}{Human-readable dataset name}
#'   \item{source}{Data source: opencanada, healthinfobase, or cihi}
#'   \item{survey}{Survey abbreviation: cpads, ccs, csads, csus, or indicators}
#'   \item{year}{Year or year range (e.g., \code{"2021-2022"})}
#'   \item{format}{File format: csv or xlsx}
#'   \item{type}{Data type: pumf, bootstrap, aggregate, or indicator}
#'   \item{large_file}{Logical; TRUE for bootstrap weight files (>100MB)}
#'   \item{local_path}{Relative path to the local data file}
#'   \item{table_name}{SQLite table name in the DBI cache}
#'   \item{ckan_resource_id}{CKAN DataStore resource ID (empty if unavailable)}
#' }
#' @source Health Canada, CIHI, Statistics Canada open data portals.
#' @examples
#' data(dataset_catalog)
#' head(dataset_catalog)
"dataset_catalog"

#' Substance Categories
#'
#' Canonical substance category mapping used across CSUS HealthInfobase
#' data files. Maps short keys to human-readable labels and source filenames.
#'
#' @format A data.frame with columns:
#' \describe{
#'   \item{key}{Short key (e.g., \code{"alcohol"}, \code{"cannabis"})}
#'   \item{label}{Display label (e.g., \code{"Alcohol"}, \code{"Cannabis"})}
#'   \item{source_file}{Filename in healthinfobase/CSUS/ directory}
#' }
#' @source Canadian Substance Use Survey (CSUS) via Health Infobase Canada.
#' @examples
#' data(substance_categories)
#' substance_categories$label
"substance_categories"

#' CKAN Metadata for Open Data APIs
#'
#' Package IDs and metadata URLs for accessing CPADS, CSADS, and CSUS
#' datasets via the Canadian Open Data CKAN API.
#'
#' @format A data.frame with columns:
#' \describe{
#'   \item{survey}{Survey abbreviation: cpads, csads, csus}
#'   \item{name}{Full survey name}
#'   \item{package_id}{CKAN package UUID}
#'   \item{metadata_url}{URL to retrieve full package metadata}
#' }
#' @source \url{https://open.canada.ca}
#' @examples
#' data(ckan_metadata)
#' ckan_metadata$metadata_url
"ckan_metadata"
