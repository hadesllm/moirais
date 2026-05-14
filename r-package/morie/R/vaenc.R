# SPDX-License-Identifier: MIT OR Apache-2.0

#' Variational autoencoder ELBO
#'
#' R parity for \code{morie.fn.vaenc.vae_elbo}.
#'
#' Gaussian encoder + standard-normal prior closed-form KL:
#' \deqn{D_\mathrm{KL}(q\|p) = -\tfrac{1}{2}\sum_j
#'       (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)}
#'
#' ELBO uses Gaussian (MSE) reconstruction term.
#'
#' @param x Original input.
#' @param x_recon Reconstruction.
#' @param mu Encoder mean.
#' @param log_var Encoder log-variance.
#' @param reduction \code{"mean"} (default) or \code{"sum"}.
#' @return Named list \code{(elbo, estimate, loss, recon_loss,
#'   kl_divergence, method)}.
#' @references Kingma & Welling (2014), ICLR.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
vaenc_vae_elbo <- function(x, x_recon, mu, log_var, reduction = "mean") {
  x <- as.array(x); x_recon <- as.array(x_recon)
  mu <- as.array(mu); log_var <- as.array(log_var)
  diff <- x - x_recon
  recon_per <- 0.5 * (diff * diff)
  kl_per <- -0.5 * (1 + log_var - mu^2 - exp(log_var))
  if (length(dim(recon_per)) > 1L) {
    recon_per <- rowSums(matrix(recon_per, nrow = dim(recon_per)[1L]))
    kl_per    <- rowSums(matrix(kl_per,    nrow = dim(kl_per)[1L]))
  } else {
    recon_per <- sum(recon_per)
    kl_per    <- sum(kl_per)
  }
  agg <- switch(reduction, "mean" = mean, "sum" = sum,
                stop(sprintf("reduction must be 'mean' or 'sum', got %s",
                             reduction)))
  recon_loss <- agg(recon_per)
  kl_div <- agg(kl_per)
  elbo <- -(recon_loss + kl_div)
  loss <- -elbo
  list(elbo = elbo, estimate = elbo, loss = loss,
       recon_loss = recon_loss, kl_divergence = kl_div,
       method = "VAE ELBO")
}

#' @rdname vaenc_vae_elbo
#' @keywords internal
#' @export
vae_elbo <- vaenc_vae_elbo
