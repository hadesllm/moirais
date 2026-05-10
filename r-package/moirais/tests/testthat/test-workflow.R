library(moirais)

test_that("run_pipeline supports custom workflow map", {
  root <- file.path(tempdir(), paste0("moirais-workflow-", as.integer(runif(1, 1, 1e8))))
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  file.create(file.path(root, "pyproject.toml"))
  dir.create(file.path(root, "libexec", "config"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(root, "docs", "source"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(root, "libexec", "config", "tests", "rtests"), recursive = TRUE, showWarnings = FALSE)

  writeLines(c("#!/usr/bin/env Rscript", "quit(status = 0)"), file.path(root, "libexec", "config", "tests", "rtests", "alpha.R"))
  writeLines(c("#!/usr/bin/env Rscript", "quit(status = 0)"), file.path(root, "libexec", "config", "tests", "rtests", "beta.R"))

  smap <- c(prepare = "libexec/config/tests/rtests/alpha.R", run = "libexec/config/tests/rtests/beta.R")

  res <- run_pipeline(project_root = root, script_map = smap, verbose = FALSE)

  expect_equal(as.character(res$step), c("prepare", "run"))
  expect_true(all(res$status == 0))
})

test_that("default workflow map returns a non-empty named character vector", {
  smap <- default_workflow_map()
  expect_type(smap, "character")
  expect_true(length(smap) >= 1)
  expect_false(is.null(names(smap)))
})
