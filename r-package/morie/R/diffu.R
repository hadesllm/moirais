# SPDX-License-Identifier: MIT OR Apache-2.0

#' 1D heat diffusion solver (explicit finite differences)
#'
#' R parity for \code{morie.fn.diffu.heat_diffusion}.  Forward-Euler
#' update of \eqn{\partial_t T = \alpha \partial_x^2 T} with Dirichlet
#' (fixed-endpoint) boundary conditions.
#'
#' @param T0 Initial temperature profile (numeric vector, length >= 3).
#' @param alpha Thermal diffusivity (default 0.01).
#' @param dx Spatial step (default 0.1).
#' @param dt Time step (default 0.01).
#' @param n_steps Number of time steps (default 100).
#' @return Named list \code{(value, T_final, T_initial, history,
#'   r_stability, n_steps, alpha, method)}.
#' @references Crank (1975), Mathematics of Diffusion.
#' @export
diffu_heat_diffusion <- function(T0, alpha = 0.01, dx = 0.1, dt = 0.01,
                                 n_steps = 100L) {
  T0 <- as.numeric(T0)
  if (length(T0) < 3L) stop("T0 must have at least 3 points.")
  r <- alpha * dt / (dx^2)
  if (r > 0.5)
    stop(sprintf("CFL violated: r=%.4f > 0.5. Reduce dt or increase dx.", r))
  n_x <- length(T0)
  Tt <- T0
  history <- matrix(0, n_steps + 1L, n_x)
  history[1L, ] <- Tt
  for (k in seq_len(n_steps)) {
    Tn <- Tt
    Tn[2:(n_x - 1L)] <- Tt[2:(n_x - 1L)] +
      r * (Tt[3:n_x] - 2 * Tt[2:(n_x - 1L)] + Tt[1:(n_x - 2L)])
    Tt <- Tn
    history[k + 1L, ] <- Tt
  }
  list(value = mean(Tt), T_final = Tt, T_initial = T0,
       history = history, r_stability = r,
       n_steps = as.integer(n_steps), alpha = alpha,
       method = "1D Heat Diffusion (forward Euler)")
}

#' DDPM forward (noising) process
#'
#' R parity for \code{morie.fn.diffu.diffusion_forward}.
#'
#' \deqn{x_t = \sqrt{\bar\alpha_t}\, x_0 +
#'       \sqrt{1 - \bar\alpha_t}\, \varepsilon}
#'
#' with linear \eqn{\beta} schedule from \code{1e-4} to \code{0.02}.
#'
#' @param x0 Clean sample.
#' @param t Diffusion timestep (1..\code{num_steps}).
#' @param betas Optional custom \eqn{\beta} schedule.
#' @param num_steps Total diffusion steps (default 1000).
#' @param noise Pre-generated Gaussian noise.
#' @param seed RNG seed.
#' @return Named list \code{(x_t, estimate, noise, alpha_bar, beta, method)}.
#' @references Ho, Jain & Abbeel (2020), NeurIPS.
#' @export
diffu_diffusion_forward <- function(x0, t, betas = NULL, num_steps = 1000L,
                                    noise = NULL, seed = 0L) {
  x0 <- as.numeric(x0)
  if (is.null(betas)) betas <- seq(1e-4, 0.02, length.out = num_steps)
  betas <- as.numeric(betas)
  if (t < 1L || t > length(betas))
    stop(sprintf("t must be in [1, %d], got %d", length(betas), t))
  alphas <- 1 - betas
  alpha_bar <- prod(alphas[1:t])
  if (is.null(noise)) {
    set.seed(seed)
    noise <- stats::rnorm(length(x0))
  }
  noise <- as.numeric(noise)
  x_t <- sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
  list(x_t = x_t, estimate = x_t, noise = noise,
       alpha_bar = alpha_bar, beta = betas[t],
       method = "DDPM forward diffusion")
}

#' @rdname diffu_heat_diffusion
#' @keywords internal
#' @export
diffusion_forward <- diffu_heat_diffusion
