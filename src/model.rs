//! The inverse Bayes model of the centrality observable `n`.
//!
//! The distribution of `n` at fixed impact parameter is the fluctuation kernel
//! `P(n|c_b)`, with mean `n̄(c_b)` and variance `σ²(c_b)`, where
//! `c_b = ∫₀ᵇ P(b') db'` is the b-centrality. Since `c_b` is uniform on `[0, 1]`,
//! the measured distribution is
//!
//! ```text
//! P(n) = ∫₀¹ P(n|c_b) dc_b
//! ```
//!
//! (Das et al., PRC 97 (2018) 014905, arXiv:1708.00081; Rogly et al.,
//! PRC 98 (2018) 024902, arXiv:1804.03031; Frankland et al., PRC 104 (2021)
//! 034609, arXiv:2011.04496). The integral over `c_b` is done with a
//! Gauss-Legendre rule in `u = √c_b` (uniform in `b` for a geometric `P(b)`),
//! which puts more points at small `c_b`, where `P(n|c_b)` changes fastest.
//!
//! With pile-up, a recorded event is with probability `p` the sum of two
//! independent collisions:
//!
//! ```text
//! P_rec(n) = (1 - p) P(n) + p ∫₀¹∫₀¹ P(n | c₁, c₂) dc₁ dc₂
//! ```

use std::fmt;
use std::str::FromStr;

use rayon::prelude::*;
use serde::Deserialize;

use crate::error::Error;
use crate::numeric::{composite_gauss_legendre, gamma_pq, ln_gamma, normal_cdf, normal_sf};

/// Probabilities below this are treated as zero when walking into the tails.
const TAIL_CUT: f64 = 1e-22;
/// Points per panel of the composite Gauss-Legendre rule in `√c_b`.
const ORDER: usize = 8;
/// Nodes per parallel task. It does not depend on the number of threads, so
/// neither do the summation order and the results.
const CHUNK: usize = 16;
/// Points of the grid in the sum `M` of the means of two collisions, on which
/// the pile-up distribution is integrated.
const PILE_UP_GRID: usize = 1024;
/// Cells of the single collision distribution in the numerical convolution.
const CONVOLUTION_CELLS: usize = 8192;

/// Fluctuation kernel `P(n|c_b)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Kernel {
    /// Gamma distribution with shape `n̄²/σ²` and scale `σ²/n̄` (Rogly et al.).
    Gamma,
    /// Gaussian truncated to `n > 0` and renormalized (Das et al.).
    Gauss,
}

/// Parametrization of the mean `n̄(c_b)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum MeanModel {
    /// `n̄ = n_knee · exp(-Σ_{j=1}^{J} a_j c_b^j)` with `J` coefficients
    /// (Das et al. use `J = 3`, Rogly et al. `J = 2..4`).
    ExpPoly(usize),
    /// `n̄ = x_max · (1 - c_b^α)^γ + x_min` (Frankland et al.; their
    /// `k(c_b) = n̄/θ`).
    Indra,
}

/// Dependence of the variance `σ²(c_b)` on `c_b`; `θ` is the reduced variance
/// `σ²/n̄` of the most central collisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum VarianceModel {
    /// `σ² = θ · n̄(c_b)`: constant `θ`, variance proportional to the mean
    /// (scenario A of Das et al.; the choice of Rogly et al. and Frankland et al.).
    Proportional,
    /// `σ² = θ · n̄(0)`: constant width (scenario B of Das et al.).
    Constant,
}

/// How a histogram bin is compared to the model density.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum BinModel {
    /// Integral of the density over the bin (exact for any bin width).
    Integral,
    /// Density at the bin center times the bin width (`TH1::Fit` default).
    Center,
}

/// A complete model: kernel, mean and variance parametrizations, and
/// optionally pile-up.
///
/// The parameter vector is `[θ, <mean parameters>]`, followed by the pile-up
/// probability `p` if `pile_up` is set; see [`Model::param_names`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub kernel: Kernel,
    pub mean: MeanModel,
    pub variance: VarianceModel,
    /// With probability `p`, a recorded event is the sum of two independent
    /// collisions.
    pub pile_up: bool,
}

impl Model {
    /// Names of the parameters, in the order of the parameter vector.
    pub fn param_names(&self) -> Vec<String> {
        let mut names = vec!["theta".to_string()];
        match self.mean {
            MeanModel::ExpPoly(n) => {
                names.push("n_knee".into());
                names.extend((1..=n).map(|j| format!("a{j}")));
            }
            MeanModel::Indra => {
                names.extend(["x_max", "x_min", "alpha", "gamma"].map(String::from));
            }
        }
        if self.pile_up {
            names.push("p".into());
        }
        names
    }

    pub fn n_params(&self) -> usize {
        1 + self.n_mean_params() + usize::from(self.pile_up)
    }

    /// Number of parameters of the mean `n̄(c_b)`.
    pub fn n_mean_params(&self) -> usize {
        match self.mean {
            MeanModel::ExpPoly(n) => 1 + n,
            MeanModel::Indra => 4,
        }
    }

    /// Index of the pile-up probability in the parameter vector.
    pub fn pile_up_index(&self) -> Option<usize> {
        self.pile_up.then(|| self.n_params() - 1)
    }

    /// Pile-up probability (0 without pile-up).
    pub fn pile_up_probability(&self, p: &[f64]) -> f64 {
        self.pile_up_index().map_or(0., |i| p[i])
    }

    /// Mean `n̄(c_b)`.
    pub fn mean(&self, p: &[f64], cb: f64) -> f64 {
        match self.mean {
            MeanModel::ExpPoly(n) => {
                /* Horner: Σ a_j c^j = c (a1 + c (a2 + ...)) */
                let poly = p[2..2 + n].iter().rev().fold(0., |acc, a| (acc + a) * cb);
                p[1] * (-poly).exp()
            }
            MeanModel::Indra => {
                let (x_max, x_min, alpha, gamma) = (p[1], p[2], p[3], p[4]);
                let base = (1. - cb.powf(alpha)).max(0.);
                x_max * base.powf(gamma) + x_min
            }
        }
    }

    /// The knee: mean of `n` at zero impact parameter, `n̄(0)`.
    pub fn n_knee(&self, p: &[f64]) -> f64 {
        self.mean(p, 0.)
    }

    /// Variance `σ²(c_b)` for the mean `mean = n̄(c_b)`.
    pub fn variance(&self, p: &[f64], mean: f64) -> f64 {
        match self.variance {
            VarianceModel::Proportional => p[0] * mean,
            VarianceModel::Constant => p[0] * self.n_knee(p),
        }
    }

    /// Kernel `P(n|c_b)` at `c_b`.
    pub fn kernel_at(&self, p: &[f64], cb: f64) -> NodeKernel {
        let mean = self.mean(p, cb);
        NodeKernel::new(self.kernel, mean, self.variance(p, mean))
    }

    /// Whether `p` is a valid parameter vector (finite, positive `θ` and mean,
    /// pile-up probability in `[0, 1)`).
    pub fn is_valid(&self, p: &[f64]) -> bool {
        if p.len() != self.n_params() || !p.iter().all(|v| v.is_finite()) || !(p[0] > 0.) {
            return false;
        }
        if !(0. ..1.).contains(&self.pile_up_probability(p)) {
            return false;
        }
        match self.mean {
            MeanModel::ExpPoly(_) => p[1] > 0.,
            MeanModel::Indra => {
                p[1] >= 0. && p[2] >= 0. && p[1] + p[2] > 0. && p[3] > 0. && p[4] > 0.
            }
        }
    }

    /// Short description, e.g. `Gamma, ExpPoly(3), Proportional, pile-up`.
    pub fn describe(&self) -> String {
        format!(
            "{}, {}, {}{}",
            self.kernel,
            self.mean,
            self.variance,
            if self.pile_up { ", pile-up" } else { "" }
        )
    }

    /// Probabilities `P(edges[i] <= n < edges[i+1])` of all bins for the
    /// recorded events, `(1 - p) · P_single + p · P_pile-up`.
    pub fn bin_probabilities(
        &self,
        p: &[f64],
        quad: &Quadrature,
        edges: &[f64],
        bin_model: BinModel,
    ) -> Vec<f64> {
        let (single, pile_up) = self.bin_probability_parts(p, quad, edges, bin_model);
        match pile_up {
            Some(double) => single.iter().zip(&double).map(|(s, d)| s + d).collect(),
            None => single,
        }
    }

    /// Contributions of single collisions, `(1 - p) · P_single`, and of
    /// pile-up, `p · P_pile-up` (`None` without pile-up), to the bin
    /// probabilities of the recorded events.
    pub fn bin_probability_parts(
        &self,
        p: &[f64],
        quad: &Quadrature,
        edges: &[f64],
        bin_model: BinModel,
    ) -> (Vec<f64>, Option<Vec<f64>>) {
        let single = self.single_bin_probabilities(p, quad, edges, bin_model);
        if !self.pile_up {
            return (single, None);
        }
        let f = self.pile_up_probability(p);
        let double = self.pile_up_bin_probabilities(p, quad, edges, bin_model);
        (
            single.iter().map(|s| (1. - f) * s).collect(),
            Some(double.iter().map(|d| f * d).collect()),
        )
    }

    /// Probabilities `P(edges[i] <= n < edges[i+1])` of all bins for single
    /// collisions, integrated over `c_b` with `quad`.
    pub fn single_bin_probabilities(
        &self,
        p: &[f64],
        quad: &Quadrature,
        edges: &[f64],
        bin_model: BinModel,
    ) -> Vec<f64> {
        let nodes: Vec<(f64, f64)> = quad
            .nodes
            .iter()
            .copied()
            .zip(quad.weights.iter().copied())
            .collect();
        sum_kernels(&nodes, edges, bin_model, |cb| self.kernel_at(p, cb))
    }

    /// Probabilities of all bins for the sum of two independent collisions,
    /// `∫∫ P(n | c₁, c₂) dc₁ dc₂`.
    ///
    /// For the Gamma kernel with `Proportional` variance this is exact and
    /// fast: two collisions with means `n̄(c₁)` and `n̄(c₂)` share the scale
    /// `θ`, so their sum is a Gamma variable with mean `M = n̄(c₁) + n̄(c₂)` and
    /// scale `θ`. The weights of all pairs of `c_b` nodes are collected on a
    /// grid in `M`, each shared linearly between the two nearest points (which
    /// keeps its mean and makes the result continuous in the parameters), and
    /// the kernel is summed over that grid. Otherwise the single collision
    /// distribution is convolved with itself numerically, see
    /// [`Model::pile_up_by_convolution`].
    pub fn pile_up_bin_probabilities(
        &self,
        p: &[f64],
        quad: &Quadrature,
        edges: &[f64],
        bin_model: BinModel,
    ) -> Vec<f64> {
        if !(self.kernel == Kernel::Gamma && self.variance == VarianceModel::Proportional) {
            return self.pile_up_by_convolution(p, quad, edges);
        }
        let means: Vec<f64> = quad.nodes.iter().map(|&cb| self.mean(p, cb)).collect();
        let m_max = 2. * means.iter().copied().fold(0., f64::max);
        let n_grid = PILE_UP_GRID;
        let step = m_max / (n_grid - 1) as f64;
        let partial: Vec<Vec<f64>> = means
            .par_chunks(CHUNK)
            .zip(quad.weights.par_chunks(CHUNK))
            .map(|(m1s, w1s)| {
                let mut grid = vec![0.; n_grid];
                for (&m1, &w1) in m1s.iter().zip(w1s) {
                    for (&m2, &w2) in means.iter().zip(&quad.weights) {
                        let x = if step > 0. { (m1 + m2) / step } else { 0. };
                        let k = (x.floor() as usize).min(n_grid - 2);
                        let f = (x - k as f64).clamp(0., 1.);
                        grid[k] += w1 * w2 * (1. - f);
                        grid[k + 1] += w1 * w2 * f;
                    }
                }
                grid
            })
            .collect();
        let grid = sum_in_order(&partial, n_grid);
        let nodes: Vec<(f64, f64)> = grid
            .iter()
            .enumerate()
            .filter(|(_, w)| **w > 0.)
            .map(|(k, &w)| (k as f64 * step, w))
            .collect();
        let theta = p[0];
        sum_kernels(&nodes, edges, bin_model, |m| {
            NodeKernel::new(self.kernel, m, theta * m)
        })
    }

    /// Pile-up by numerical convolution, for any kernel: the single collision
    /// probabilities are computed in `CONVOLUTION_CELLS` uniform cells of width `h`
    /// from 0 to where the remaining fraction of events is below 1e-10; the
    /// sum of two values in cells `i` and `j`, uniform within the cells, is
    /// distributed triangularly on `[(i + j) h, (i + j + 2) h)`, and these
    /// triangles are integrated over the bins. The bins are always integrated
    /// (`bin_model` does not apply).
    pub fn pile_up_by_convolution(&self, p: &[f64], quad: &Quadrature, edges: &[f64]) -> Vec<f64> {
        let n_knee = self.n_knee(p);
        let mut x_top = n_knee + 10. * self.variance(p, n_knee).sqrt();
        for _ in 0..64 {
            if self.centrality(p, quad, x_top) < 1e-10 {
                break;
            }
            x_top *= 2.;
        }
        let n_cells = CONVOLUTION_CELLS;
        let h = x_top / n_cells as f64;
        let cell_edges: Vec<f64> = (0..=n_cells).map(|k| k as f64 * h).collect();
        let single = self.single_bin_probabilities(p, quad, &cell_edges, BinModel::Integral);
        /* conv[k] = Σ_{i+j=k} s_i s_j, in fixed chunks of k */
        let conv: Vec<f64> = (0..2 * n_cells - 1)
            .into_par_iter()
            .with_min_len(CHUNK)
            .map(|k| {
                let lo = k.saturating_sub(n_cells - 1);
                let hi = k.min(n_cells - 1);
                (lo..=hi).map(|i| single[i] * single[k - i]).sum()
            })
            .collect();
        /* cumulative distribution of the triangles at x */
        let mut prefix = vec![0.; conv.len() + 1];
        for (k, c) in conv.iter().enumerate() {
            prefix[k + 1] = prefix[k] + c;
        }
        let triangle_cdf = |u: f64| {
            if u <= 0. {
                0.
            } else if u < 1. {
                0.5 * u * u
            } else if u < 2. {
                1. - 0.5 * (2. - u) * (2. - u)
            } else {
                1.
            }
        };
        let cdf = |x: f64| -> f64 {
            if !(x > 0.) {
                return 0.;
            }
            let t = x / h;
            /* triangles starting at k end before x for k <= t - 2 */
            let full = ((t - 2.).floor() + 1.).clamp(0., conv.len() as f64) as usize;
            let mut sum = prefix[full];
            let last = (t.floor() as usize).min(conv.len() - 1);
            for (k, c) in conv.iter().enumerate().take(last + 1).skip(full) {
                sum += c * triangle_cdf(t - k as f64);
            }
            sum
        };
        let values: Vec<f64> = edges.iter().map(|&x| cdf(x)).collect();
        values.windows(2).map(|w| (w[1] - w[0]).max(0.)).collect()
    }

    /// Centrality of single collisions, `c(x) = P(n >= x) = ∫₀¹ P(n >= x | c_b) dc_b`.
    pub fn centrality(&self, p: &[f64], quad: &Quadrature, x: f64) -> f64 {
        quad.nodes
            .iter()
            .zip(&quad.weights)
            .map(|(&cb, &w)| w * self.kernel_at(p, cb).survival(x))
            .sum()
    }

    /// Density of `n` for single collisions, `P(x) = ∫₀¹ P(x|c_b) dc_b`.
    pub fn density(&self, p: &[f64], quad: &Quadrature, x: f64) -> f64 {
        quad.nodes
            .iter()
            .zip(&quad.weights)
            .map(|(&cb, &w)| w * self.kernel_at(p, cb).pdf(x))
            .sum()
    }
}

/// Sum of the partial results, in order.
fn sum_in_order(partial: &[Vec<f64>], n: usize) -> Vec<f64> {
    let mut total = vec![0.; n];
    for part in partial {
        for (t, v) in total.iter_mut().zip(part) {
            *t += v;
        }
    }
    total
}

/// `Σ w · P(bin | kernel(x))` over the nodes `(x, w)`. The nodes are processed
/// in parallel, in chunks of fixed size summed in a fixed order, so the result
/// is reproducible and does not depend on the number of threads.
fn sum_kernels(
    nodes: &[(f64, f64)],
    edges: &[f64],
    bin_model: BinModel,
    kernel: impl Fn(f64) -> NodeKernel + Sync,
) -> Vec<f64> {
    let n_bins = edges.len().saturating_sub(1);
    let partial: Vec<Vec<f64>> = nodes
        .par_chunks(CHUNK)
        .map(|chunk| {
            let mut out = vec![0.; n_bins];
            for &(x, w) in chunk {
                kernel(x).add_bin_probabilities(edges, w, bin_model, &mut out);
            }
            out
        })
        .collect();
    let mut total = vec![0.; n_bins];
    for part in &partial {
        for (t, v) in total.iter_mut().zip(part) {
            *t += v;
        }
    }
    total
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Display for VarianceModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Display for BinModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Display for MeanModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeanModel::ExpPoly(n) => write!(f, "ExpPoly({n})"),
            MeanModel::Indra => f.write_str("Indra"),
        }
    }
}

/// Parses the names written by `Display` (used to read back the fit file).
fn parse_name<T: Copy + fmt::Display>(all: &[T], s: &str, what: &str) -> Result<T, Error> {
    all.iter()
        .copied()
        .find(|v| v.to_string().eq_ignore_ascii_case(s.trim()))
        .ok_or_else(|| Error::Input(format!("unknown {what} '{s}'")))
}

impl FromStr for Kernel {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        parse_name(&[Kernel::Gamma, Kernel::Gauss], s, "kernel")
    }
}

impl FromStr for VarianceModel {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        parse_name(
            &[VarianceModel::Proportional, VarianceModel::Constant],
            s,
            "variance model",
        )
    }
}

impl FromStr for BinModel {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        parse_name(&[BinModel::Integral, BinModel::Center], s, "bin model")
    }
}

impl FromStr for MeanModel {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("Indra") {
            return Ok(MeanModel::Indra);
        }
        s.strip_prefix("ExpPoly(")
            .and_then(|r| r.strip_suffix(')'))
            .and_then(|n| n.trim().parse().ok())
            .map(MeanModel::ExpPoly)
            .ok_or_else(|| Error::Input(format!("unknown mean model '{s}'")))
    }
}

/// The kernel `P(n|c_b)` at one value of `c_b`.
#[derive(Debug, Clone, Copy)]
pub enum NodeKernel {
    Gamma {
        mean: f64,
        shape: f64,
        scale: f64,
        ln_gamma_shape: f64,
    },
    /// Normal distribution truncated to `n > 0`; `norm` is the untruncated
    /// probability of `n > 0`.
    Gauss { mean: f64, sigma: f64, norm: f64 },
    /// All probability at `n = 0` (vanishing mean).
    Zero,
}

impl NodeKernel {
    pub fn new(kernel: Kernel, mean: f64, variance: f64) -> Self {
        if !(mean > 1e-300) || !(variance > 0.) {
            return NodeKernel::Zero;
        }
        match kernel {
            Kernel::Gamma => {
                let shape = mean * mean / variance;
                NodeKernel::Gamma {
                    mean,
                    shape,
                    scale: variance / mean,
                    ln_gamma_shape: ln_gamma(shape),
                }
            }
            Kernel::Gauss => {
                let sigma = variance.sqrt();
                NodeKernel::Gauss {
                    mean,
                    sigma,
                    norm: normal_cdf(mean / sigma),
                }
            }
        }
    }

    pub fn mean(&self) -> f64 {
        match *self {
            NodeKernel::Gamma { mean, .. } | NodeKernel::Gauss { mean, .. } => mean,
            NodeKernel::Zero => 0.,
        }
    }

    /// `(P(n < x), P(n >= x))`; the smaller one is accurate in the tails.
    pub fn pq(&self, x: f64) -> (f64, f64) {
        match *self {
            NodeKernel::Gamma {
                shape,
                scale,
                ln_gamma_shape,
                ..
            } => gamma_pq(shape, x / scale, ln_gamma_shape),
            NodeKernel::Gauss { mean, sigma, norm } => {
                if !(x > 0.) {
                    return (0., 1.);
                }
                let z = (x - mean) / sigma;
                if x < mean {
                    let p = ((normal_cdf(z) - normal_cdf(-mean / sigma)) / norm).clamp(0., 1.);
                    (p, 1. - p)
                } else {
                    let q = (normal_sf(z) / norm).clamp(0., 1.);
                    (1. - q, q)
                }
            }
            NodeKernel::Zero => {
                if x > 0. {
                    (1., 0.)
                } else {
                    (0., 1.)
                }
            }
        }
    }

    /// `P(n >= x)`.
    pub fn survival(&self, x: f64) -> f64 {
        if x == f64::INFINITY {
            return 0.;
        }
        self.pq(x).1
    }

    /// Probability density at `x` (0 for the degenerate kernel).
    pub fn pdf(&self, x: f64) -> f64 {
        match *self {
            NodeKernel::Gamma {
                shape,
                scale,
                ln_gamma_shape,
                ..
            } => {
                if !(x > 0.) {
                    return 0.;
                }
                let t = x / scale;
                ((shape - 1.) * t.ln() - t - ln_gamma_shape).exp() / scale
            }
            NodeKernel::Gauss { mean, sigma, norm } => {
                if !(x > 0.) {
                    return 0.;
                }
                let z = (x - mean) / sigma;
                (-0.5 * z * z).exp() / (sigma * (2. * std::f64::consts::PI).sqrt() * norm)
            }
            NodeKernel::Zero => 0.,
        }
    }

    /// Adds `w · P(edges[i] <= n < edges[i+1])` to `out[i]` for every bin.
    /// Starting at the mean, each edge is evaluated once while walking
    /// outwards, until the remaining tail probability is negligible.
    pub fn add_bin_probabilities(
        &self,
        edges: &[f64],
        w: f64,
        bin_model: BinModel,
        out: &mut [f64],
    ) {
        let n_edges = edges.len();
        if n_edges < 2 {
            return;
        }
        let mean = self.mean();
        /* first edge at or above the mean */
        let j0 = edges.partition_point(|&e| e < mean);
        if bin_model == BinModel::Center {
            self.add_center_probabilities(edges, w, j0, out);
            return;
        }

        /* right of the mean: bins between edges j and j+1 from P(n >= edge) */
        let mut q_right = 1.;
        if j0 < n_edges {
            q_right = self.pq(edges[j0]).1;
            let mut prev = q_right;
            for j in j0..n_edges - 1 {
                let next = self.pq(edges[j + 1]).1;
                out[j] += w * (prev - next);
                prev = next;
                if next < TAIL_CUT {
                    break;
                }
            }
        }
        /* left of the mean, from P(n < edge) */
        if j0 > 0 {
            let p_left = self.pq(edges[j0 - 1]).0;
            if j0 < n_edges {
                /* the bin containing the mean */
                out[j0 - 1] += w * (1. - p_left - q_right).max(0.);
            }
            let mut prev = p_left;
            for i in (0..j0 - 1).rev() {
                if prev < TAIL_CUT {
                    break;
                }
                let next = self.pq(edges[i]).0;
                out[i] += w * (prev - next);
                prev = next;
            }
        }
    }

    fn add_center_probabilities(&self, edges: &[f64], w: f64, j0: usize, out: &mut [f64]) {
        let n_bins = edges.len() - 1;
        let term = |i: usize| self.pdf(0.5 * (edges[i] + edges[i + 1])) * (edges[i + 1] - edges[i]);
        /* bin containing the mean (or the nearest one) */
        let i0 = j0.saturating_sub(1).min(n_bins - 1);
        let mut prev = f64::INFINITY;
        for (i, o) in out.iter_mut().enumerate().take(n_bins).skip(i0) {
            let t = term(i);
            *o += w * t;
            if t < TAIL_CUT && t <= prev {
                break;
            }
            prev = t;
        }
        let mut prev = f64::INFINITY;
        for i in (0..i0).rev() {
            let t = term(i);
            out[i] += w * t;
            if t < TAIL_CUT && t <= prev {
                break;
            }
            prev = t;
        }
    }
}

/// Quadrature rule for integrals over `c_b`, with nodes uniform in `√c_b`.
#[derive(Debug, Clone)]
pub struct Quadrature {
    /// Values of `c_b`.
    pub nodes: Vec<f64>,
    /// Weights (they sum to the length of the `c_b` interval).
    pub weights: Vec<f64>,
}

impl Quadrature {
    /// Rule on `[0, 1]` with at least `points` nodes.
    pub fn new(points: usize) -> Self {
        Self::on_interval(0., 1., points)
    }

    /// Rule on `[cb_lo, cb_hi]` with at least `points` nodes.
    pub fn on_interval(cb_lo: f64, cb_hi: f64, points: usize) -> Self {
        let panels = points.div_ceil(ORDER).max(1);
        let (u, wu) =
            composite_gauss_legendre(cb_lo.max(0.).sqrt(), cb_hi.max(0.).sqrt(), panels, ORDER);
        /* c = u², dc = 2u du */
        let nodes = u.iter().map(|u| u * u).collect();
        let weights = u.iter().zip(&wu).map(|(u, w)| 2. * u * w).collect();
        Quadrature { nodes, weights }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    fn models() -> Vec<(Model, Vec<f64>)> {
        let mut v = Vec::new();
        for kernel in [Kernel::Gamma, Kernel::Gauss] {
            for variance in [VarianceModel::Proportional, VarianceModel::Constant] {
                v.push((
                    Model {
                        kernel,
                        mean: MeanModel::ExpPoly(3),
                        variance,
                        pile_up: false,
                    },
                    vec![1.2, 200., 3.5, 0.8, 1.6],
                ));
                v.push((
                    Model {
                        kernel,
                        mean: MeanModel::Indra,
                        variance,
                        pile_up: false,
                    },
                    vec![0.8, 150., 4., 1.2, 1.5],
                ));
            }
        }
        v
    }

    #[test]
    fn names_and_parsing() {
        for (m, p) in models() {
            assert_eq!(m.param_names().len(), m.n_params());
            assert_eq!(p.len(), m.n_params());
            assert!(m.is_valid(&p));
            assert_eq!(m.kernel.to_string().parse::<Kernel>().unwrap(), m.kernel);
            assert_eq!(m.mean.to_string().parse::<MeanModel>().unwrap(), m.mean);
            assert_eq!(
                m.variance.to_string().parse::<VarianceModel>().unwrap(),
                m.variance
            );
            let with = Model { pile_up: true, ..m };
            assert_eq!(with.n_params(), m.n_params() + 1);
            assert_eq!(with.param_names().last().unwrap(), "p");
            assert_eq!(with.pile_up_index(), Some(m.n_params()));
            let mut q = p.clone();
            q.push(0.01);
            assert!(with.is_valid(&q));
            /* the pile-up parameter does not enter the mean */
            assert_eq!(with.mean(&q, 0.3), m.mean(&p, 0.3));
            q[m.n_params()] = 1.;
            assert!(!with.is_valid(&q));
        }
        assert!("ExpPoly(x)".parse::<MeanModel>().is_err());
    }

    #[test]
    fn mean_parametrizations() {
        let m = Model {
            kernel: Kernel::Gamma,
            mean: MeanModel::ExpPoly(2),
            variance: VarianceModel::Proportional,
            pile_up: false,
        };
        let p = [1., 100., 2., 1.];
        assert!(close(m.mean(&p, 0.5), 100. * (-1.25f64).exp(), 1e-14));
        assert_eq!(m.n_knee(&p), 100.);
        let m = Model {
            mean: MeanModel::Indra,
            ..m
        };
        let p = [1., 100., 5., 2., 1.];
        assert!(close(m.mean(&p, 0.5), 100. * 0.75 + 5., 1e-14));
        assert!(close(m.mean(&p, 1.), 5., 1e-14));
    }

    #[test]
    fn bins_sum_to_one_and_match_centrality() {
        let quad = Quadrature::new(512);
        assert!(close(quad.weights.iter().sum::<f64>(), 1., 1e-13));
        /* wide range: the constant width model has a long tail at large c_b */
        let edges: Vec<f64> = (0..=20000).map(|i| i as f64).collect();
        for (m, p) in models() {
            let probs = m.bin_probabilities(&p, &quad, &edges, BinModel::Integral);
            let total: f64 = probs.iter().sum();
            assert!(close(total, 1., 1e-9), "{}: {total}", m.describe());
            /* the tail above an edge equals the centrality there */
            for x in [10usize, 100, 180, 230] {
                let above: f64 = probs[x..].iter().sum();
                let c = m.centrality(&p, &quad, x as f64);
                assert!(
                    close(above, c, 1e-8),
                    "{} x = {x}: {above} vs {c}",
                    m.describe()
                );
            }
            /* the density at the bin centers is close to the bin integral
             * away from n = 0, where a Gamma with shape < 1 diverges */
            let center = m.bin_probabilities(&p, &quad, &edges, BinModel::Center);
            let (c, i): (f64, f64) = (center[10..].iter().sum(), probs[10..].iter().sum());
            assert!(close(c, i, 1e-4), "{}: {c} vs {i}", m.describe());
        }
    }

    #[test]
    fn pile_up_is_the_sum_of_two_events() {
        let quad = Quadrature::new(256);
        let edges: Vec<f64> = (0..=5000).map(|i| i as f64).collect();
        for (m, p) in models() {
            let with = Model { pile_up: true, ..m };
            let mut q = p.clone();
            q.push(0.1);
            let single = m.bin_probabilities(&p, &quad, &edges, BinModel::Integral);
            let double = with.pile_up_bin_probabilities(&q, &quad, &edges, BinModel::Integral);
            /* normalized, with the mean and (for proportional variance) the
             * variance of the sum of two independent events */
            let moments = |v: &[f64]| {
                let (mut s0, mut s1, mut s2) = (0., 0., 0.);
                for (i, w) in v.iter().enumerate() {
                    let x = i as f64 + 0.5;
                    s0 += w;
                    s1 += w * x;
                    s2 += w * x * x;
                }
                (s0, s1 / s0, s2 / s0 - (s1 / s0).powi(2))
            };
            let (n1, mean1, var1) = moments(&single);
            let (n2, mean2, var2) = moments(&double);
            assert!(
                close(n1, 1., 1e-6) && close(n2, 1., 1e-6),
                "{}",
                m.describe()
            );
            /* binning adds 1/12 to each variance, not to the means; with
             * constant width much of the single-collision probability is in
             * the first bin, whose center overestimates its mean */
            let tolerance = match m.variance {
                VarianceModel::Proportional => 1e-3,
                VarianceModel::Constant => 5e-3,
            };
            assert!(
                close(mean2, 2. * mean1, tolerance),
                "{}: {mean2} vs {mean1}",
                m.describe()
            );
            if m.variance == VarianceModel::Proportional {
                assert!(
                    close(var2 - 1. / 12., 2. * (var1 - 1. / 12.), 2e-3),
                    "{}: {var2} vs {var1}",
                    m.describe()
                );
            }
            /* the mixture */
            let total = with.bin_probabilities(&q, &quad, &edges, BinModel::Integral);
            for i in [5, 50, 150, 250, 400] {
                let expected = 0.9 * single[i] + 0.1 * double[i];
                assert!(close(total[i], expected, 1e-12), "{}", m.describe());
            }
        }
    }

    #[test]
    fn kernel_moments() {
        for kernel in [Kernel::Gamma, Kernel::Gauss] {
            let k = NodeKernel::new(kernel, 50., 40.);
            let edges: Vec<f64> = (0..=4000).map(|i| i as f64 * 0.05).collect();
            let mut probs = vec![0.; edges.len() - 1];
            k.add_bin_probabilities(&edges, 1., BinModel::Integral, &mut probs);
            let (mut s0, mut s1, mut s2) = (0., 0., 0.);
            for (i, pr) in probs.iter().enumerate() {
                let x = 0.5 * (edges[i] + edges[i + 1]);
                s0 += pr;
                s1 += pr * x;
                s2 += pr * x * x;
            }
            let mean = s1 / s0;
            assert!(close(s0, 1., 1e-12));
            assert!(close(mean, 50., 1e-6), "{kernel}: {mean}");
            assert!(close(s2 / s0 - mean * mean, 40., 1e-3), "{kernel}");
            /* density integrates to the bin probabilities */
            assert!(close(k.pdf(50.) * 0.05, probs[1000], 1e-3));
        }
        let z = NodeKernel::new(Kernel::Gamma, 0., 1.);
        let mut probs = vec![0.; 2];
        z.add_bin_probabilities(&[0., 1., 2.], 1., BinModel::Integral, &mut probs);
        assert_eq!(probs, vec![1., 0.]);
    }
}
