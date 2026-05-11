#!/usr/bin/env Rscript
# make_summary_data.R — Generate small .rda objects for the R package
#
# These ship with the package (under data/) and provide metadata
# without bundling the full 1.86GB of raw data.

library(morie)

# 1. Dataset catalog as data.frame
dataset_catalog <- morie_dataset_catalog()
cat("dataset_catalog:", nrow(dataset_catalog), "entries\n")

# 2. Substance categories (from CSUS healthinfobase files)
substance_categories <- data.frame(
  key = c("alcohol", "cannabis", "opioids", "stimulants", "sedatives",
          "polysubstance", "smoking_vaping", "illegal", "harms", "otc", "treatment"),
  label = c("Alcohol", "Cannabis", "Opioids", "Stimulants", "Sedatives",
            "Polysubstance Use", "Cigarette Smoking & Vaping",
            "Illegal Substances", "Substance Use Harms",
            "Over-the-Counter Products", "Treatment"),
  source_file = c("Alcohol.csv", "Cannabis.csv", "Opioids.csv", "Stimulants.csv",
                  "Sedatives.csv", "Polysubstance.csv", "Cigarette smoking and vaping.csv",
                  "Illegal substances.csv", "Substance use harms.csv",
                  "Over the counter products.csv", "Treatment.csv"),
  stringsAsFactors = FALSE
)
cat("substance_categories:", nrow(substance_categories), "entries\n")

# 3. CKAN metadata for open data API access
ckan_metadata <- data.frame(
  survey = c("cpads", "csads", "csus"),
  name = c("Canadian Postsecondary Education Alcohol and Drug Use Survey",
           "Canadian Student Alcohol and Drugs Survey",
           "Canadian Substance Use Survey"),
  package_id = c("736fa9b2-62e4-4e31-aea4-51869605b363",
                 "1f15ca45-8bfd-4f9c-9ec6-2c0c440e69c2",
                 "65e2d45e-efc6-4c29-9a9b-db59bc96aa0e"),
  metadata_url = c(
    "https://open.canada.ca/data/api/action/package_show?id=736fa9b2-62e4-4e31-aea4-51869605b363",
    "https://open.canada.ca/data/api/action/package_show?id=1f15ca45-8bfd-4f9c-9ec6-2c0c440e69c2",
    "https://open.canada.ca/data/api/action/package_show?id=65e2d45e-efc6-4c29-9a9b-db59bc96aa0e"
  ),
  stringsAsFactors = FALSE
)
cat("ckan_metadata:", nrow(ckan_metadata), "entries\n")

# Save all objects
data_dir <- file.path(dirname(getwd()), "data")
if (!dir.exists(data_dir)) dir.create(data_dir)

save(dataset_catalog, file = file.path(data_dir, "dataset_catalog.rda"), compress = "bzip2")
save(substance_categories, file = file.path(data_dir, "substance_categories.rda"), compress = "bzip2")
save(ckan_metadata, file = file.path(data_dir, "ckan_metadata.rda"), compress = "bzip2")

cat("\nSaved .rda files to:", data_dir, "\n")
