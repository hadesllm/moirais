library(morie)

test_that("synthetic generator returns expected generic schema", {
  dat <- generate_synthetic_data(n = 1000, seed = 123, special_code_rate = 0)

  req <- c(
    "id", "weight", "sex", "age_group", "region",
    "alcohol_ever", "alcohol_12m", "alcohol_freq_30d", "heavy_drinking_30d",
    "bac", "bac_legal", "risk_indicator", "cannabis_use",
    "physical_health", "mental_health"
  )

  expect_true(all(req %in% names(dat)))
  expect_equal(nrow(dat), 1000)
  expect_true(all(dat$weight > 0))
  expect_true(isTRUE(attr(dat, "synthetic")))
})

test_that("synthetic generation is deterministic by seed", {
  a <- generate_synthetic_data(n = 500, seed = 77, special_code_rate = 0)
  b <- generate_synthetic_data(n = 500, seed = 77, special_code_rate = 0)
  c <- generate_synthetic_data(n = 500, seed = 78, special_code_rate = 0)

  expect_identical(a, b)
  expect_false(identical(a, c))
})

test_that("custom name map works", {
  map <- default_synthetic_name_map("generic")
  map[["cannabis_use"]] <- "exposure_any"

  dat <- generate_synthetic_data(n = 400, seed = 9, special_code_rate = 0, name_map = map)

  expect_true("exposure_any" %in% names(dat))
  expect_false("cannabis_use" %in% names(dat))
})

test_that("write_synthetic_data writes a CSV", {
  out <- file.path(tempdir(), paste0("synthetic_data_", as.integer(runif(1, 1, 1e8)), ".csv"))
  p <- write_synthetic_data(path = out, n = 300, seed = 2, overwrite = TRUE)

  expect_true(file.exists(p))

  x <- utils::read.csv(p, check.names = FALSE)
  expect_equal(nrow(x), 300)
  expect_true(all(c("id", "weight", "cannabis_use") %in% names(x)))
})
