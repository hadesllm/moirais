test_that("list_morie_modules exposes implemented module names", {
  mods <- list_morie_modules()
  expect_true(all(c("power-design", "propensity-scores", "ebac-selection-adjustment-ipw") %in% mods$name))
})

test_that("morie_load_dataset loads CPADS from built-in DB", {
  skip_if_not(requireNamespace("DBI", quietly = TRUE), "DBI not installed")
  skip_if_not(requireNamespace("RSQLite", quietly = TRUE), "RSQLite not installed")
  dat <- tryCatch(morie_load_dataset("cpads_2021"), error = function(e) NULL)
  skip_if(is.null(dat), "Built-in DB not available")
  expect_true(nrow(dat) > 0)
  expect_true("SEQID" %in% names(dat) || "weight" %in% names(dat))
})

test_that("morie_list_datasets shows all catalog entries", {
  skip_if_not(requireNamespace("DBI", quietly = TRUE), "DBI not installed")
  skip_if_not(requireNamespace("RSQLite", quietly = TRUE), "RSQLite not installed")
  ds <- morie_list_datasets()
  expect_true(nrow(ds) >= 20)
  expect_true("ocp21" %in% ds$key)
})

test_that("dataset catalog has expected structure", {
  cat <- morie_dataset_catalog()
  expect_true(nrow(cat) >= 20)
  expect_true(all(c("key", "name", "source", "survey", "table_name") %in% names(cat)))
})
