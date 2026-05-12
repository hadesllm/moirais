# SPDX-License-Identifier: GPL-2.0-only

#' GAN minimax / non-saturating loss
#'
#' R parity for \code{morie.fn.ganls.gan_loss}.
#'
#' \deqn{V(D) = \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))]}
#'
#' so \code{D_loss = -V(D)}.  Two generator objectives are supported:
#' \code{kind="minimax"} (\code{G_loss = E[log(1 - D(G(z)))]}) and
#' \code{kind="nonsaturating"} (\code{G_loss = -E[log D(G(z))]}).
#'
#' @param D_real Discriminator outputs on real data (probabilities).
#' @param D_fake Discriminator outputs on generator samples.
#' @param kind One of \code{"minimax"} or \code{"nonsaturating"}.
#' @return Named list \code{(d_loss, g_loss, v, estimate, kind, method)}.
#' @references Goodfellow et al. (2014), NeurIPS.
#' @export
ganls_gan_loss <- function(D_real, D_fake, kind = "minimax") {
  D_real <- as.numeric(D_real); D_fake <- as.numeric(D_fake)
  clip_log <- function(p) log(pmin(pmax(p, 1e-12), 1.0))
  v_real <- mean(clip_log(D_real))
  v_fake_neg <- mean(clip_log(1 - D_fake))
  V <- v_real + v_fake_neg
  d_loss <- -V
  g_loss <- switch(kind,
    "minimax" = v_fake_neg,
    "nonsaturating" = -mean(clip_log(D_fake)),
    stop(sprintf("kind must be 'minimax' or 'nonsaturating', got %s", kind))
  )
  list(d_loss = d_loss, g_loss = g_loss, v = V,
       estimate = d_loss, kind = kind,
       method = sprintf("GAN %s loss", kind))
}

#' @rdname ganls_gan_loss
#' @keywords internal
#' @export
gan_loss <- ganls_gan_loss
