#!/usr/bin/env Rscript
# Native Detroit Berry reproduction vs the v2 Python translation.
# Sources local cmfproperty R files; does not install packages from the internet.
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
repo <- normalizePath(file.path(dirname(file_arg), "../../.."))
combined <- file.path(repo, "data/berry_cmf/raw/detroit_mi/box/qzz9nz9l81m1vku1q6luqzmxvdw9q9wb/combined files/combined.csv")
cmf <- file.path(repo, "data/berry_cmf/raw/_shared/cmfproperty/R")
out <- file.path(repo, "analysis/berry_attom_validation_v2/berry_reproduction/detroit_native_r.json")

ok <- TRUE
msg <- character()
try_lib <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    msg <<- c(msg, paste("missing_package", pkg))
    ok <<- FALSE
    FALSE
  } else {
    library(pkg, character.only = TRUE)
    TRUE
  }
}
for (pkg in c("dplyr", "readr", "lubridate", "magrittr")) try_lib(pkg)
if (!ok) {
  json <- sprintf('{"native_r_attempted": true, "native_r_status": "PACKAGES_MISSING", "notes": "%s"}\n', paste(msg, collapse="; "))
  writeLines(json, out)
  quit(status = 0)
}
helper <- file.path(cmf, "helper_fxns.R")
reformat <- file.path(cmf, "reformat_data.R")
iaao <- file.path(cmf, "iaao_stats.R")
if (!file.exists(helper) || !file.exists(reformat) || !file.exists(iaao)) {
  writeLines('{"native_r_attempted": true, "native_r_status": "CMFPROPERTY_SOURCES_MISSING"}\n', out)
  quit(status = 0)
}
source(helper)
source(reformat)
source(iaao)
assessor_data <- readr::read_csv(combined, show_col_types = FALSE)
assessor_data <- assessor_data %>%
  dplyr::mutate(SALE_YEAR = lubridate::year(`Sale Date`)) %>%
  dplyr::filter(`Property Class` == 401, `Terms of Sale` == "VALID ARMS LENGTH")
ratios <- reformat_data(assessor_data, "Adj. Sale $", "Asd. when Sold", "SALE_YEAR", filter_data = FALSE)
window <- ratios %>% dplyr::filter(SALE_YEAR >= 2016, SALE_YEAR <= 2018)
stats <- tryCatch(get_stats(window, bootstrap_iters = 1), error = function(e) e)
n <- nrow(window)
q2 <- sum(format(as.Date(window$`Sale Date`), "%Y") == "2016" & as.integer(format(as.Date(window$`Sale Date`), "%m")) %in% 4:6, na.rm = TRUE)
json <- sprintf(
  '{"native_r_attempted": true, "native_r_status": "RAN", "n": %d, "q2_2016_n": %d, "stats_class": "%s"}\n',
  n, q2, paste(class(stats), collapse=",")
)
writeLines(json, out)
cat(json)
