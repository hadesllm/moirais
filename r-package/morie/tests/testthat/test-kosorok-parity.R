# Smoke test for ksr01..ksr20: source each R file then call with the
# same canonical fixtures as the Python smoke test.
suppressMessages({
  for (f in sprintf("/tmp/morie-feature/r-package/morie/R/ksr%02d.R", 1:20))
    source(f)
})

xs <- c(0.1, 0.4, -0.3, 0.7, 0.05, -0.9, 1.2, -0.4, 0.6, -0.1,
        0.3, -0.2, 0.5, -0.7, 0.0, 0.2, -0.1, 0.4, -0.5, 0.8)
ys <- 1.5 * xs + c(0.2, -0.1, 0.05, 0.3, -0.2, 0.1, -0.3, 0.0, 0.1, -0.05,
                   -0.1, 0.0, 0.2, -0.2, 0.1, 0.05, -0.1, 0.2, -0.3, 0.1)
ts_ <- 1:10
ev  <- c(1, 1, 0, 1, 1, 0, 1, 1, 1, 0)
X3  <- matrix(0, 100, 3)

fmt <- function(d, keys) {
  parts <- sapply(keys, function(k) {
    v <- d[[k]]
    if (is.null(v) || !is.numeric(v)) paste0(k, "=NA")
    else sprintf("%s=%.10g", k, v)
  })
  paste(parts, collapse = " | ")
}

cat("ksr01:", fmt(ksr01_kosorok_empirical_process(xs, mu0 = 0), c("estimate","se")), "\n")
cat("ksr02:", fmt(ksr02_kosorok_donsker_class(xs), c("estimate")), "\n")
cat("ksr03:", fmt(ksr03_kosorok_glivenko_cantelli(xs), c("statistic","p_value")), "\n")
cat("ksr04:", fmt(ksr04_kosorok_vc_dimension(X3), c("estimate")), "\n")
cat("ksr05:", fmt(ksr05_kosorok_bracketing_number(xs, 0.1), c("estimate")), "\n")
cat("ksr06:", fmt(ksr06_kosorok_maximal_inequality(xs), c("estimate")), "\n")
cat("ksr07_se_approx:", fmt(ksr07_kosorok_bootstrap_empirical(xs, B = 2000, seed = 42), c("se")), "\n")
cat("ksr08_se_approx:", fmt(ksr08_kosorok_multiplier_bootstrap(xs, B = 2000, seed = 42), c("se")), "\n")
cat("ksr09:", fmt(ksr09_kosorok_z_estimator(xs, ys), c("estimate","se")), "\n")
cat("ksr10:", fmt(ksr10_kosorok_m_estimator(xs), c("estimate","se")), "\n")
cat("ksr11:", fmt(ksr11_kosorok_efficient_score(xs, ys), c("estimate","se")), "\n")
cat("ksr12:", fmt(ksr12_kosorok_information_bound(xs, ys), c("estimate")), "\n")
cat("ksr13:", fmt(ksr13_kosorok_tangent_space(xs), c("estimate")), "\n")
cat("ksr14:", fmt(ksr14_kosorok_profile_likelihood(xs, ys), c("estimate","se")), "\n")
cat("ksr15:", fmt(ksr15_kosorok_one_step_estimator(xs), c("estimate","se")), "\n")
cat("ksr16:", fmt(ksr16_kosorok_influence_function(xs, ys), c("estimate")), "\n")
cat("ksr17:", fmt(ksr17_kosorok_counting_process(ts_, ev), c("estimate")), "\n")
cat("ksr18:", fmt(ksr18_kosorok_nelson_aalen(ts_, ev), c("estimate","se")), "\n")
xs_cox <- xs[1:10]
cat("ksr19:", fmt(ksr19_kosorok_cox_partial_likelihood(xs_cox, ts_, ev), c("estimate","se")), "\n")
cat("ksr20:", fmt(ksr20_kosorok_censoring_survival(ts_, ev), c("estimate","se")), "\n")
