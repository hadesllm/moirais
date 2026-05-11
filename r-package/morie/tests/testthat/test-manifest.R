library(morie)

make_fixture_project <- function() {
  root <- file.path(tempdir(), paste0("morie-fixture-", as.integer(stats::runif(1, 1, 1e9))))
  dir.create(root, recursive = TRUE, showWarnings = FALSE)

  # Root markers for find_project_root()
  file.create(file.path(root, "pyproject.toml"))
  dir.create(file.path(root, "libexec", "config"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(root, "data", "manifest", "outputs", "figures"), recursive = TRUE, showWarnings = FALSE)

  writeLines("x,y\n1,2", file.path(root, "data", "manifest", "outputs", "table1.csv"))
  writeLines("notes", file.path(root, "data", "manifest", "outputs", "notes.txt"))
  root
}

test_that("build and read manifest", {
  root <- make_fixture_project()
  manifest_path <- file.path(root, "data", "manifest", "outputs_manifest.csv")

  built <- build_outputs_manifest(
    output_dir = file.path(root, "data", "manifest", "outputs"),
    manifest_path = manifest_path
  )

  expect_true(file.exists(manifest_path))
  expect_true(all(c("output", "public_path", "size_kb", "modified") %in% names(built)))

  read_back <- read_outputs_manifest(project_root = root)
  expect_equal(nrow(read_back), nrow(built))
  expect_true(validate_outputs_manifest(read_back))
})

test_that("audit captures missing and unexpected outputs", {
  root <- make_fixture_project()
  manifest_path <- file.path(root, "data", "manifest", "outputs_manifest.csv")

  manifest <- build_outputs_manifest(
    output_dir = file.path(root, "data", "manifest", "outputs"),
    manifest_path = manifest_path
  )

  unlink(file.path(root, "data", "manifest", "outputs", "notes.txt"))
  writeLines("new", file.path(root, "data", "manifest", "outputs", "extra.csv"))

  audit <- audit_public_outputs(project_root = root, manifest = manifest)
  summary <- summarize_output_audit(audit)

  expect_equal(summary$declared_missing, 1)
  expect_equal(summary$unexpected_files, 1)
})

test_that("find_project_root discovers parent project", {
  root <- make_fixture_project()
  nested <- file.path(root, "a", "b", "c")
  dir.create(nested, recursive = TRUE, showWarnings = FALSE)

  found <- find_project_root(start = nested)
  expect_equal(normalizePath(found), normalizePath(root))
})
