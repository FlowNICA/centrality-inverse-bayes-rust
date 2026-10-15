//! Relation between the impact parameter `b` and the b-centrality
//! `c_b = ∫₀ᵇ P(b') db'` of inelastic events.

use std::f64::consts::PI;

use oxiroot::prelude::*;

use crate::config::ImpactParameter;
use crate::error::{Error, Result};
use crate::numeric::{bisect, composite_gauss_legendre};

/// Points of the tabulated `c_b(b)` of the Fermi distribution.
const FERMI_POINTS: usize = 20_000;
/// Bins of the histogram built from per-event values of `b`.
const TREE_BINS: usize = 4000;

/// `c_b(b)` and its inverse.
#[derive(Debug, Clone, PartialEq)]
pub enum ImpactProfile {
    /// `c_b = πb²/σ_inel`.
    Geometric { sigma_inel: f64 },
    /// Piecewise linear `c_b(b)` through the points `(b[i], cb[i])`, both
    /// ascending, from `cb = 0` to `cb = 1`.
    Table {
        b: Vec<f64>,
        cb: Vec<f64>,
        sigma_inel: f64,
        description: String,
    },
}

impl ImpactProfile {
    pub fn from_config(ip: &ImpactParameter) -> Result<Self> {
        match ip {
            ImpactParameter::Geometric { sigma_inel } => Ok(ImpactProfile::Geometric {
                sigma_inel: *sigma_inel,
            }),
            ImpactParameter::Fermi {
                sigma_inel,
                b0,
                width,
            } => {
                let b0 = match (b0, sigma_inel) {
                    (Some(b0), _) => *b0,
                    (None, Some(s)) => fermi_b0(*s, *width),
                    (None, None) => {
                        return Err(Error::Config(
                            "Fermi impact parameter needs sigma_inel or b0".into(),
                        ));
                    }
                };
                Ok(Self::fermi(b0, *width))
            }
            ImpactParameter::Histogram { file, hist } => {
                let f = FileReader::open(file)
                    .map_err(|e| Error::Input(format!("cannot open {}: {e}", file.display())))?;
                let h = Hist1D::read_root(&f, hist).map_err(|e| {
                    Error::Input(format!("cannot read '{hist}' from {}: {e}", file.display()))
                })?;
                Self::from_histogram(&h, format!("histogram '{hist}' of {}", file.display()))
            }
            ImpactParameter::Tree { file, tree, branch } => {
                let f = FileReader::open(file)
                    .map_err(|e| Error::Input(format!("cannot open {}: {e}", file.display())))?;
                let t = TreeReader::open(&f, tree)?;
                let values = crate::output::numbers(&t, &f, branch)?;
                Self::from_values(
                    &values,
                    format!("branch '{branch}' of tree '{tree}' in {}", file.display()),
                )
            }
            ImpactParameter::Glauber {
                nucleus_a,
                nucleus_b,
                sigma_nn,
                sigma_np,
                sqrt_snn,
                nuclei_file,
            } => Ok(crate::optical::OpticalGlauber::new(
                nucleus_a,
                nucleus_b,
                *sigma_nn,
                *sigma_np,
                *sqrt_snn,
                nuclei_file.as_deref(),
            )?
            .profile()),
        }
    }

    /// `P(b) ∝ 2πb / (1 + exp((b - b0)/width))`, tabulated.
    pub fn fermi(b0: f64, width: f64) -> Self {
        let b_end = b0 + 40. * width;
        let pdf = |b: f64| 2. * PI * b / (1. + ((b - b0) / width).exp());
        let h = b_end / FERMI_POINTS as f64;
        let (nodes, weights) = crate::numeric::gauss_legendre(4);
        let mut b = Vec::with_capacity(FERMI_POINTS + 1);
        let mut cum = Vec::with_capacity(FERMI_POINTS + 1);
        let mut total = 0.;
        b.push(0.);
        cum.push(0.);
        for k in 0..FERMI_POINTS {
            let mid = (k as f64 + 0.5) * h;
            total += nodes
                .iter()
                .zip(&weights)
                .map(|(x, w)| 0.5 * h * w * pdf(mid + 0.5 * h * x))
                .sum::<f64>();
            b.push((k + 1) as f64 * h);
            cum.push(total);
        }
        let cb = cum.iter().map(|c| c / total).collect();
        ImpactProfile::Table {
            b,
            cb,
            sigma_inel: total,
            description: format!(
                "Fermi P(b), b0 = {b0:.4} fm, width = {width} fm, sigma_inel = {total:.2} fm^2"
            ),
        }
    }

    /// From a histogram of `b`: `c_b` is its cumulative distribution,
    /// linear within the bins.
    pub fn from_histogram(h: &Hist1D, description: String) -> Result<Self> {
        let edges = h.edges();
        let n = edges.len() - 1;
        let counts: Vec<f64> = (1..=n).map(|i| h.contents[i].max(0.)).collect();
        let total: f64 = counts.iter().sum();
        if !(total > 0.) || !(edges[0] >= 0.) {
            return Err(Error::Input(format!(
                "{description}: needs a non-empty histogram of b >= 0"
            )));
        }
        let mut b = vec![edges[0]];
        let mut cb = vec![0.];
        let mut cum = 0.;
        for (i, c) in counts.iter().enumerate() {
            cum += c;
            b.push(edges[i + 1]);
            cb.push(cum / total);
        }
        if edges[0] > 0. {
            b.insert(0, 0.);
            cb.insert(0, 0.);
        }
        Ok(ImpactProfile::Table {
            b,
            cb,
            sigma_inel: f64::NAN,
            description,
        })
    }

    /// From per-event values of `b`, through a fine histogram.
    pub fn from_values(values: &[f64], description: String) -> Result<Self> {
        let max = values.iter().copied().fold(0., f64::max);
        if values.is_empty() || !(max > 0.) || values.iter().any(|v| !(*v >= 0.)) {
            return Err(Error::Input(format!(
                "{description}: needs values of b >= 0, not all zero"
            )));
        }
        let mut h = Hist::reg(TREE_BINS as i32, 0., max * (1. + 1e-9)).double();
        for &v in values {
            h.fill(v);
        }
        Self::from_histogram(&h, description)
    }

    /// `c_b(b)`.
    pub fn cb_of_b(&self, b: f64) -> f64 {
        match self {
            ImpactProfile::Geometric { sigma_inel } => (PI * b * b / sigma_inel).min(1.),
            ImpactProfile::Table { b: bs, cb, .. } => interpolate(bs, cb, b),
        }
    }

    /// Inverse of `c_b(b)` for `c_b` in `[0, 1]`.
    pub fn b_of_cb(&self, cb: f64) -> f64 {
        let cb = cb.clamp(0., 1.);
        match self {
            ImpactProfile::Geometric { sigma_inel } => (cb * sigma_inel / PI).sqrt(),
            ImpactProfile::Table { b, cb: cbs, .. } => interpolate(cbs, b, cb),
        }
    }

    /// Largest impact parameter: where `c_b` reaches 1 (`1 - 1e-6` for the
    /// Fermi tail).
    pub fn b_max(&self) -> f64 {
        match self {
            ImpactProfile::Geometric { .. } => self.b_of_cb(1.),
            ImpactProfile::Table { .. } => self.b_of_cb(1. - 1e-6),
        }
    }

    /// Inelastic cross section in fm², if known.
    pub fn sigma_inel(&self) -> Option<f64> {
        let s = match self {
            ImpactProfile::Geometric { sigma_inel } | ImpactProfile::Table { sigma_inel, .. } => {
                *sigma_inel
            }
        };
        s.is_finite().then_some(s)
    }

    pub fn description(&self) -> String {
        match self {
            ImpactProfile::Geometric { sigma_inel } => {
                format!("geometric P(b), sigma_inel = {sigma_inel} fm^2")
            }
            ImpactProfile::Table { description, .. } => description.clone(),
        }
    }
}

/// `σ_inel = ∫ 2πb / (1 + exp((b - b0)/width)) db`.
pub fn fermi_sigma(b0: f64, width: f64) -> f64 {
    let (nodes, weights) = composite_gauss_legendre(0., b0 + 40. * width, 2000, 4);
    nodes
        .iter()
        .zip(&weights)
        .map(|(&b, &w)| w * 2. * PI * b / (1. + ((b - b0) / width).exp()))
        .sum()
}

/// `b0` giving the cross section `sigma_inel` (fm²) for the given width.
pub fn fermi_b0(sigma_inel: f64, width: f64) -> f64 {
    let hi = (sigma_inel / PI).sqrt() + 10. * width;
    bisect(|b0| fermi_sigma(b0, width) - sigma_inel, 0., hi, 100)
}

/// Linear interpolation of `y(x)` through ascending points; flat outside.
fn interpolate(x: &[f64], y: &[f64], at: f64) -> f64 {
    let n = x.len();
    if at <= x[0] {
        return y[0];
    }
    if at >= x[n - 1] {
        return y[n - 1];
    }
    let i = x.partition_point(|&v| v <= at).clamp(1, n - 1);
    let (x0, x1, y0, y1) = (x[i - 1], x[i], y[i - 1], y[i]);
    if x1 > x0 {
        y0 + (y1 - y0) * (at - x0) / (x1 - x0)
    } else {
        y0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric() {
        let p = ImpactProfile::Geometric { sigma_inel: 700. };
        let b_max = (700. / PI).sqrt();
        assert!((p.b_max() - b_max).abs() < 1e-12);
        assert!((p.cb_of_b(b_max / 2.) - 0.25).abs() < 1e-12);
        assert!((p.b_of_cb(0.25) - b_max / 2.).abs() < 1e-12);
    }

    #[test]
    fn fermi_cross_section_and_inverse() {
        /* σ = -2π Δb² Li₂(-e^{b0/Δb}) ≈ π b0² + π³ Δb²/3 for b0 ≫ Δb */
        let (b0, width) = (10.1, 0.4);
        let approx = PI * b0 * b0 + PI.powi(3) * width * width / 3.;
        assert!((fermi_sigma(b0, width) - approx).abs() < 1e-6 * approx);
        let b0_back = fermi_b0(fermi_sigma(b0, width), width);
        assert!((b0_back - b0).abs() < 1e-9);
        let p = ImpactProfile::fermi(b0, width);
        assert!((p.sigma_inel().unwrap() - approx).abs() < 1e-6 * approx);
        /* well inside, c_b ≈ πb²/σ */
        let cb = p.cb_of_b(5.);
        assert!((cb - PI * 25. / approx).abs() < 1e-6);
        assert!((p.b_of_cb(cb) - 5.).abs() < 1e-6);
    }

    #[test]
    fn from_values_matches_geometric() {
        /* b distributed as 2b/b_max² on [0, b_max] */
        let b_max: f64 = 15.;
        let values: Vec<f64> = (0..200_000)
            .map(|i| b_max * ((i as f64 + 0.5) / 200_000.).sqrt())
            .collect();
        let p = ImpactProfile::from_values(&values, "test".into()).unwrap();
        for b in [1., 5., 10., 14.] {
            let expected = (b / b_max).powi(2);
            assert!((p.cb_of_b(b) - expected).abs() < 1e-3, "b = {b}");
        }
        assert!((p.b_of_cb(0.5) - b_max * 0.5f64.sqrt()).abs() < 0.01);
        assert!(ImpactProfile::from_values(&[], "e".into()).is_err());
    }
}
