#!/usr/bin/env Rscript
# ingest_datasets.R — Ingest all MOIRAIS datasets into SQLite cache
#
# Usage:
#   Rscript data-raw/ingest_datasets.R               # ingest all
#   Rscript data-raw/ingest_datasets.R --skip-large   # skip bootstrap
#   Rscript data-raw/ingest_datasets.R --only opencanada_cpads_2021

library(moirais)

args <- commandArgs(trailingOnly = TRUE)
skip_large <- "--skip-large" %in% args
only_key <- NULL
if ("--only" %in% args) {
  idx <- which(args == "--only")
  if (idx < length(args)) only_key <- args[idx + 1L]
}

cat("MOIRAIS Dataset Ingest\n")
cat("===================\n\n")

catalog <- moirais_dataset_catalog()
con <- moirais_db_connect()
on.exit(DBI::dbDisconnect(con), add = TRUE)

# Ensure metadata table exists.
DBI::dbExecute(con, "
  CREATE TABLE IF NOT EXISTS _moirais_metadata (
    table_name TEXT PRIMARY KEY,
    source TEXT, survey TEXT, year TEXT, format TEXT,
    row_count INTEGER, col_count INTEGER,
    columns TEXT, ingested_at TEXT, file_hash TEXT
  )
")

ingested <- 0L
skipped <- 0L

for (i in seq_len(nrow(catalog))) {
  entry <- catalog[i, ]
  key <- entry$key

  if (!is.null(only_key) && key != only_key) next

  if (skip_large && entry$large_file) {
    cat(sprintf("  SKIP (large): %s\n", key))
    skipped <- skipped + 1L
    next
  }

  path <- entry$local_path
  if (!file.exists(path)) {
    cat(sprintf("  SKIP (missing): %s -> %s\n", key, path))
    skipped <- skipped + 1L
    next
  }

  size_mb <- round(file.info(path)$size / 1e6, 1)
  cat(sprintf("  Ingesting: %s (%s, %.1fMB)\n", key, entry$format, size_mb))
  table_name <- entry$table_name

  tryCatch({
    if (entry$format == "csv") {
      if (entry$large_file && requireNamespace("data.table", quietly = TRUE)) {
        # Chunked read for large files.
        header <- data.table::fread(path, nrows = 0)
        col_names <- names(header)
        offset <- 0L
        first <- TRUE
        repeat {
          chunk <- data.table::fread(path, skip = offset + 1L, nrows = 50000L,
                                      col.names = col_names, header = FALSE)
          if (nrow(chunk) == 0L) break
          DBI::dbWriteTable(con, table_name, as.data.frame(chunk),
                            overwrite = first, append = !first)
          first <- FALSE
          offset <- offset + nrow(chunk)
        }
        n <- offset
      } else {
        df <- utils::read.csv(path, stringsAsFactors = FALSE)
        DBI::dbWriteTable(con, table_name, df, overwrite = TRUE)
        n <- nrow(df)
      }
    } else if (entry$format == "xlsx") {
      if (!requireNamespace("readxl", quietly = TRUE)) {
        cat("    SKIP: readxl not installed\n")
        skipped <- skipped + 1L
        next
      }
      sheets <- readxl::excel_sheets(path)
      if (length(sheets) == 1L) {
        df <- readxl::read_excel(path, sheet = 1)
        DBI::dbWriteTable(con, table_name, as.data.frame(df), overwrite = TRUE)
        n <- nrow(df)
      } else {
        n <- 0L
        for (sheet in sheets) {
          df <- readxl::read_excel(path, sheet = sheet)
          if (nrow(df) == 0L) next
          safe <- tolower(gsub("[^a-zA-Z0-9]", "_", sheet))
          safe <- substr(safe, 1, 40)
          tbl <- paste0(table_name, "_", safe)
          DBI::dbWriteTable(con, tbl, as.data.frame(df), overwrite = TRUE)
          n <- n + nrow(df)
          cat(sprintf("    sheet '%s' -> %s (%d rows)\n", sheet, tbl, nrow(df)))
        }
      }
    } else {
      cat(sprintf("    Unknown format: %s\n", entry$format))
      skipped <- skipped + 1L
      next
    }

    # Write metadata.
    cols <- DBI::dbListFields(con, table_name)
    DBI::dbExecute(con, "INSERT OR REPLACE INTO _moirais_metadata VALUES (?,?,?,?,?,?,?,?,?,?)",
      params = list(table_name, entry$source, entry$survey, entry$year,
                    entry$format, n, length(cols),
                    jsonlite::toJSON(cols, auto_unbox = FALSE),
                    format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ"), ""))

    cat(sprintf("    OK: %s rows, %d cols -> %s\n", format(n, big.mark = ","), length(cols), table_name))
    ingested <- ingested + 1L
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", conditionMessage(e)))
    skipped <<- skipped + 1L
  })
}

cat(sprintf("\nDone: %d ingested, %d skipped\n", ingested, skipped))
