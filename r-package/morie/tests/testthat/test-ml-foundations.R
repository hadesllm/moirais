# Smoke tests for the ML Foundations R-parity wrappers.
# Mirrors the Python `if __name__ == "__main__":` blocks in src/morie/fn/*.

skip_if_no_pkg <- function(pkg) testthat::skip_if_not_installed(pkg)

test_that("linear_regression_ols recovers known coefficients", {
  set.seed(0)
  X <- matrix(rnorm(400), 200, 2)
  y <- 0.5 + X %*% c(1.5, -2.0) + rnorm(200, sd = 0.1)
  r <- linear_regression_ols(X, y)
  expect_equal(r$estimate, c(0.5, 1.5, -2.0), tolerance = 0.05)
  expect_length(r$se, 3)
  expect_equal(r$n, 200)
})

test_that("gradient_descent_vanilla converges to OLS", {
  set.seed(0)
  X <- matrix(rnorm(400), 200, 2)
  y <- 0.5 + X %*% c(1.5, -2.0) + rnorm(200, sd = 0.1)
  r <- gradient_descent_vanilla(X, y, lr = 0.05, n_iter = 5000)
  expect_equal(r$estimate, r$reference_ols, tolerance = 1e-4)
})

test_that("mini_batch_gradient roughly matches OLS", {
  set.seed(0)
  X <- matrix(rnorm(1500), 500, 3)
  y <- X %*% c(1.0, -0.5, 2.0) + 0.25 + rnorm(500, sd = 0.1)
  r <- mini_batch_gradient(X, y, lr = 0.02, n_epochs = 100, batch_size = 32L)
  expect_lt(max(abs(r$estimate - r$reference_ols)), 0.05)
})

test_that("polynomial_regression recovers known polynomial", {
  set.seed(0)
  x <- seq(-2, 2, length.out = 200)
  y <- 1 - 0.5 * x + 2 * x^2 + rnorm(200, sd = 0.1)
  r <- polynomial_regression(x, y, degree = 2L)
  expect_equal(r$estimate[1:3], c(1.0, -0.5, 2.0), tolerance = 0.1)
})

test_that("regularization_path returns a path matrix", {
  skip_if_no_pkg("glmnet")
  set.seed(0)
  X <- matrix(rnorm(1000), 200, 5)
  y <- X %*% c(1.0, 0.0, -0.5, 0.0, 2.0) + rnorm(200, sd = 0.1)
  r <- regularization_path(X, y, penalty = "ridge",
                            alphas = c(0.01, 0.1, 1.0, 10.0))
  expect_equal(dim(r$coef_path), c(4, 6))
  expect_length(r$alphas, 4)
})

test_that("learning_curve produces sensible curves", {
  set.seed(0)
  X <- matrix(rnorm(600), 200, 3)
  y <- X %*% c(1.0, -0.5, 2.0) + rnorm(200, sd = 0.5)
  r <- learning_curve(X, y, sizes = seq(0.2, 1.0, length.out = 4), cv = 3L)
  expect_length(r$train_scores, 4)
  expect_length(r$val_scores, 4)
})

test_that("svm_hinge_primal fits a linear SVM", {
  skip_if_no_pkg("e1071")
  set.seed(0)
  X <- matrix(rnorm(400), 200, 2)
  y <- as.integer(X[, 1] + X[, 2] > 0)
  r <- svm_hinge_primal(X, y, C = 1.0)
  expect_gt(r$train_accuracy, 0.85)
})

test_that("svm_kernel_trick fits an RBF SVM", {
  skip_if_no_pkg("e1071")
  set.seed(0)
  X <- matrix(rnorm(400), 200, 2)
  y <- as.integer(rowSums(X^2) < 1)
  r <- svm_kernel_trick(X, y, kernel = "rbf")
  expect_gt(r$train_accuracy, 0.85)
})

test_that("decision_tree_split builds a tree", {
  skip_if_no_pkg("rpart")
  set.seed(0)
  X <- matrix(rnorm(600), 200, 3)
  y <- as.integer(X[, 1] > 0)
  r <- decision_tree_split(X, y, criterion = "gini", max_depth = 3L)
  expect_gt(r$train_accuracy, 0.9)
})

test_that("random_forest_ensemble fits a random forest", {
  skip_if_no_pkg("randomForest")
  set.seed(0)
  X <- matrix(rnorm(1200), 300, 4)
  y <- as.integer(X[, 1] + X[, 2] - X[, 3] > 0)
  r <- random_forest_ensemble(X, y, n_estimators = 50L)
  expect_gt(r$train_score, 0.85)
})

test_that("gradient_boosting_ensemble fits a GBM", {
  testthat::skip_if(!requireNamespace("gbm", quietly = TRUE) &&
                    !requireNamespace("xgboost", quietly = TRUE))
  set.seed(0)
  X <- matrix(rnorm(1200), 300, 4)
  y <- as.integer(X[, 1] + 0.5 * X[, 2] - X[, 3] > 0)
  r <- gradient_boosting_ensemble(X, y, n_estimators = 50L)
  expect_gt(r$train_score, 0.85)
})

test_that("xgboost_objective fits boosted trees", {
  testthat::skip_if(!requireNamespace("xgboost", quietly = TRUE) &&
                    !requireNamespace("gbm", quietly = TRUE))
  set.seed(0)
  X <- matrix(rnorm(1200), 300, 4)
  y <- as.integer(X[, 1] + X[, 2] - X[, 3] > 0)
  r <- xgboost_objective(X, y, n_estimators = 50L)
  expect_gt(r$train_score, 0.85)
})

test_that("pca_dimension_reduction decomposes data", {
  set.seed(0)
  z <- matrix(rnorm(200), 200, 1)
  X <- z %*% t(c(1.0, 2.0, -1.0)) + 0.1 * matrix(rnorm(600), 200, 3)
  r <- pca_dimension_reduction(X, n_components = 3L)
  expect_gt(r$explained_variance_ratio[1], 0.9)
})

test_that("tsne_reduction embeds points", {
  skip_if_no_pkg("Rtsne")
  set.seed(0)
  X <- rbind(matrix(rnorm(250, mean = -3), 50, 5),
             matrix(rnorm(250, mean = +3), 50, 5))
  r <- tsne_reduction(X, n_components = 2L, perplexity = 10, n_iter = 500L)
  expect_equal(r$estimate, c(100L, 2L))
})

test_that("kmeans_clustering finds three clusters", {
  set.seed(0)
  centres <- rbind(c(0, 0), c(5, 5), c(-5, 5))
  X <- do.call(rbind, lapply(seq_len(3), function(i)
    matrix(rnorm(80, mean = 0), 40, 2) + matrix(rep(centres[i, ], 40), 40, 2, byrow = TRUE)))
  r <- kmeans_clustering(X, n_clusters = 3L)
  expect_equal(length(unique(r$labels)), 3L)
})

test_that("dbscan_clustering separates clusters from noise", {
  skip_if_no_pkg("dbscan")
  set.seed(0)
  blob1 <- matrix(rnorm(80, sd = 0.3), 40, 2)
  blob2 <- matrix(rnorm(80, mean = 5, sd = 0.3), 40, 2)
  noise <- matrix(runif(20, -2, 7), 10, 2)
  X <- rbind(blob1, blob2, noise)
  r <- dbscan_clustering(X, eps = 0.8, min_samples = 5L)
  expect_gte(r$n_clusters, 2L)
})

test_that("grid_search_cv finds best params", {
  skip_if_no_pkg("caret")
  set.seed(0)
  X <- matrix(rnorm(600), 200, 3)
  y <- as.integer(X[, 1] + X[, 2] > 0)
  r <- grid_search_cv(X, y, cv = 3L)
  expect_true(!is.null(r$best_params))
})

test_that("random_search_cv samples and scores", {
  skip_if_no_pkg("caret")
  set.seed(0)
  X <- matrix(rnorm(600), 200, 3)
  y <- as.integer(X[, 1] + X[, 2] > 0)
  r <- random_search_cv(X, y, n_iter = 5L, cv = 3L)
  expect_true(!is.null(r$best_params))
})

test_that("confusion_matrix_metrics matches sklearn semantics", {
  y_true <- c(0, 0, 1, 1, 1, 0, 1, 0, 1, 1)
  y_pred <- c(0, 1, 1, 1, 0, 0, 1, 0, 1, 1)
  r <- confusion_matrix_metrics(y_true, y_pred)
  expect_equal(r$accuracy, 0.8)
  expect_equal(r$confusion_matrix[1, 1], 3L)
  expect_equal(r$confusion_matrix[2, 2], 5L)
})

test_that("roc_auc_score computes AUC", {
  skip_if_no_pkg("pROC")
  set.seed(0)
  y <- sample(0:1, 200, replace = TRUE)
  s <- y + rnorm(200, sd = 0.6)
  r <- roc_auc_score(y, s)
  expect_gt(r$auc, 0.7)
})
