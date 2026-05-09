#!/usr/bin/env Rscript
# Export the canonical OTIS data frames from the cached RData
# environment to CSV mirrors that moirais.otis_analyze can read in
# Python. Idempotent — re-run any time the .RData fixture refreshes.

suppressWarnings({
  load("data/cache/correctional_stats_report_environment.RData")
})

write.csv(df, "data/cache/otis_main.csv", row.names = FALSE)
write.csv(dt, "data/cache/otis_dt.csv",   row.names = FALSE)

cat(sprintf("otis_main.csv: %d rows × %d cols\n", nrow(df), ncol(df)))
cat(sprintf("otis_dt.csv  : %d rows × %d cols\n", nrow(dt), ncol(dt)))
