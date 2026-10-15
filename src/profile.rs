//! Mean and variance of the observable `n` versus the impact parameter `b`
//! from a 2-D histogram of `n` and `b`, e.g. of a model, for the `Histogram`
//! mean and variance.
//!
//! In each bin of `b` the mean and variance of `n` are computed from the bin
//! centers of `n` (bins with integer lower edges, `[k, k + 1)`, are at `k + 0.5`,
//! as in the model, where a bin is the integral of a continuous density).
//! Between the centers of the `b` bins they are interpolated linearly, outside
//! they are constant.
//!
//! The histogram must contain all inelastic events: its projection on `b` is
//! the impact parameter distribution `P(b)` that turns `b` into `c_b`, the
//! fraction of all inelastic events with a smaller `b`.

use std::fmt;
use std::path::{Path, PathBuf};

use oxiroot::prelude::*;
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::impact::ImpactProfile;

/// What an axis of the 2-D histogram holds: the observable `"n"` or the
/// impact parameter `"b"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub enum Axis {
    N,
    B,
}

impl TryFrom<String> for Axis {
    type Error = String;
    fn try_from(s: String) -> std::result::Result<Self, String> {
        match s.trim() {
            "n" | "N" => Ok(Axis::N),
            "b" | "B" => Ok(Axis::B),
            _ => Err(format!("axis must be \"n\" or \"b\", got \"{s}\"")),
        }
    }
}

impl fmt::Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Axis::N => "n",
            Axis::B => "b",
        })
    }
}

/// A 2-D histogram of `n` and `b` (fm) of all inelastic events, and which
/// of its axes holds which.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramSpec {
    pub file: PathBuf,
    pub hist: String,
    pub x_axis: Axis,
    pub y_axis: Axis,
}

impl HistogramSpec {
    pub fn validate(&self) -> Result<()> {
        if self.x_axis == self.y_axis {
            return Err(Error::Config(format!(
                "histogram '{}': x_axis and y_axis must differ, one \"n\" and one \"b\", \
                 got \"{}\" for both",
                self.hist, self.x_axis
            )));
        }
        Ok(())
    }

    /// RON form of the fields, `file: "f.root", hist: "h", x_axis: "b", y_axis: "n"`.
    pub fn ron_fields(&self) -> String {
        format!(
            "file: {:?}, hist: {:?}, x_axis: \"{}\", y_axis: \"{}\"",
            self.file.display().to_string(),
            self.hist,
            self.x_axis,
            self.y_axis
        )
    }

    /// Reads the histogram: the mean and variance of `n` in each bin of `b`,
    /// and `P(b)` from its projection on `b`, which together give them versus
    /// `c_b`.
    pub fn build(&self) -> Result<CbProfile> {
        let h = self.read()?;
        let b_on_x = self.x_axis == Axis::B;
        let description = self.description(&self.file);
        Ok(CbProfile {
            profile: BProfile::from_histogram(&h, b_on_x, &description)?,
            impact: b_projection(&h, b_on_x, &description)?,
        })
    }

    /// Reads the histogram: `P(b)` from its projection on `b`.
    pub fn impact_profile(&self) -> Result<ImpactProfile> {
        let h = self.read()?;
        b_projection(&h, self.x_axis == Axis::B, &self.description(&self.file))
    }

    fn read(&self) -> Result<Hist2D> {
        self.validate()?;
        let f = FileReader::open(&self.file)
            .map_err(|e| Error::Input(format!("cannot open {}: {e}", self.file.display())))?;
        Hist2D::read_root(&f, &self.hist).map_err(|e| {
            Error::Input(format!(
                "cannot read 2-D histogram '{}' from {}: {e}",
                self.hist,
                self.file.display()
            ))
        })
    }

    fn description(&self, file: &Path) -> String {
        format!(
            "histogram '{}' of {} (x: {}, y: {})",
            self.hist,
            file.display(),
            self.x_axis,
            self.y_axis
        )
    }
}

/// Mean and variance of `n` in the bins of `b`.
#[derive(Debug, Clone, PartialEq)]
pub struct BProfile {
    /// Centers of the filled bins of `b`, ascending.
    b: Vec<f64>,
    mean: Vec<f64>,
    /// Centers of the bins of `b` with a positive variance of `n`, ascending.
    b_var: Vec<f64>,
    variance: Vec<f64>,
    pub description: String,
}

impl BProfile {
    /// From a 2-D histogram with `b` on the x axis (`b_on_x`) or on the y axis.
    pub fn from_histogram(h: &Hist2D, b_on_x: bool, description: &str) -> Result<Self> {
        let (nx, ny) = (h.nx(), h.ny());
        let cell = |ix: usize, iy: usize| h.contents[ix + (nx + 2) * iy];
        let (b_axis, n_axis) = if b_on_x {
            (&h.xaxis, &h.yaxis)
        } else {
            (&h.yaxis, &h.xaxis)
        };
        let (nb, nn) = if b_on_x { (nx, ny) } else { (ny, nx) };
        let at = |ib: usize, i_n: usize| {
            if b_on_x { cell(ib, i_n) } else { cell(i_n, ib) }
        };
        let (mut b, mut mean, mut b_var, mut variance) = (vec![], vec![], vec![], vec![]);
        let mut n_overflow = 0.;
        for ib in 1..=nb {
            let (mut sw, mut swn, mut swn2) = (0., 0., 0.);
            for i_n in 1..=nn {
                let w = at(ib, i_n);
                let x = n_axis.bin_center(i_n);
                sw += w;
                swn += w * x;
                swn2 += w * x * x;
            }
            n_overflow += at(ib, 0) + at(ib, nn + 1);
            if !(sw > 0.) {
                continue;
            }
            let m = swn / sw;
            let v = (swn2 / sw - m * m).max(0.);
            let bc = b_axis.bin_center(ib);
            b.push(bc);
            mean.push(m);
            if v > 0. {
                b_var.push(bc);
                variance.push(v);
            }
        }
        if b.is_empty() {
            return Err(Error::Input(format!("{description}: no entries")));
        }
        if variance.is_empty() {
            return Err(Error::Input(format!(
                "{description}: n has no spread in any bin of b"
            )));
        }
        if b[0] < 0. {
            return Err(Error::Input(format!(
                "{description}: the impact parameter axis must start at b >= 0"
            )));
        }
        if n_overflow > 0. {
            eprintln!(
                "Warning: {description}: {n_overflow} entries in the under- or overflow of n \
                 are not used for the mean and variance of n"
            );
        }
        Ok(BProfile {
            b,
            mean,
            b_var,
            variance,
            description: description.to_string(),
        })
    }

    /// Mean of `n` at `b`.
    pub fn mean_at(&self, b: f64) -> f64 {
        interpolate(&self.b, &self.mean, b)
    }

    /// Variance of `n` at `b`.
    pub fn variance_at(&self, b: f64) -> f64 {
        interpolate(&self.b_var, &self.variance, b)
    }
}

/// `P(b)` from the projection of the histogram on `b`: all entries in each
/// bin of `b`, including those in the under- and overflow of `n`. Entries
/// outside the axis of `b` are an error, as `P(b)` would miss them.
pub fn b_projection(h: &Hist2D, b_on_x: bool, description: &str) -> Result<ImpactProfile> {
    let (nx, ny) = (h.nx(), h.ny());
    let cell = |ix: usize, iy: usize| h.contents[ix + (nx + 2) * iy];
    let (nb, nn, b_axis) = if b_on_x {
        (nx, ny, &h.xaxis)
    } else {
        (ny, nx, &h.yaxis)
    };
    let slice = |ib: usize| -> f64 {
        (0..=nn + 1)
            .map(|i_n| if b_on_x { cell(ib, i_n) } else { cell(i_n, ib) })
            .sum()
    };
    let outside = slice(0) + slice(nb + 1);
    if outside > 0. {
        return Err(Error::Input(format!(
            "{description}: {outside} entries lie outside the axis of b, so its projection \
             is not P(b) of all events; the histogram must contain all inelastic events"
        )));
    }
    let counts: Vec<f64> = (1..=nb).map(slice).collect();
    ImpactProfile::from_counts(
        &b_axis.edges(),
        &counts,
        format!("projection on b of {description}"),
    )
}

/// A `BProfile` as a function of `c_b`, through `b(c_b)` of `P(b)`.
#[derive(Debug, Clone, PartialEq)]
pub struct CbProfile {
    pub profile: BProfile,
    pub impact: ImpactProfile,
}

impl CbProfile {
    pub fn mean(&self, cb: f64) -> f64 {
        self.profile.mean_at(self.impact.b_of_cb(cb))
    }

    pub fn variance(&self, cb: f64) -> f64 {
        self.profile.variance_at(self.impact.b_of_cb(cb))
    }
}

/// Linear interpolation of `y(x)` through ascending points; constant outside.
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
    y0 + (y1 - y0) * (at - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 bins of b on [0, 4], 10 bins of n on [0, 10]; in bin `ib` of b,
    /// `n` is in bins `10 - 2 ib` (weight 1) and `11 - 2 ib` (weight 3).
    fn filled(b_on_x: bool) -> Hist2D {
        let mut h = if b_on_x {
            Hist::reg(4, 0., 4.).reg(10, 0., 10.).double()
        } else {
            Hist::reg(10, 0., 10.).reg(4, 0., 4.).double()
        };
        for ib in 1..=3 {
            let b = ib as f64 - 0.5;
            for (n_bin, w) in [(10 - 2 * ib, 1.), (11 - 2 * ib, 3.)] {
                let n = n_bin as f64 - 0.5;
                let (x, y) = if b_on_x { (b, n) } else { (n, b) };
                h.fill_weight(x, y, w);
            }
        }
        h
    }

    #[test]
    fn mean_and_variance_in_bins_of_b_for_both_orientations() {
        for b_on_x in [true, false] {
            let p = BProfile::from_histogram(&filled(b_on_x), b_on_x, "test").unwrap();
            /* bin 1 of b: n = 7.5 (w 1) and 8.5 (w 3): mean 8.25, variance 3/16 */
            assert_eq!(p.b, [0.5, 1.5, 2.5]);
            assert!((p.mean_at(0.5) - 8.25).abs() < 1e-12);
            assert!((p.variance_at(0.5) - 0.1875).abs() < 1e-12);
            assert!((p.mean_at(1.5) - 6.25).abs() < 1e-12);
            /* linear between centers, constant outside, empty bin 4 skipped */
            assert!((p.mean_at(1.) - 7.25).abs() < 1e-12);
            assert!((p.mean_at(0.) - 8.25).abs() < 1e-12);
            assert!((p.mean_at(3.9) - 4.25).abs() < 1e-12);
            /* P(b): 4 entries in each of the bins [0, 1), [1, 2), [2, 3) */
            let ip = b_projection(&filled(b_on_x), b_on_x, "test").unwrap();
            assert!((ip.cb_of_b(1.) - 1. / 3.).abs() < 1e-12);
            assert!((ip.cb_of_b(1.5) - 0.5).abs() < 1e-12);
            assert!((ip.b_max() - 3.).abs() < 1e-5);
        }
        /* entries outside the axis of b */
        let mut h = filled(true);
        h.fill(20., 5.);
        assert!(b_projection(&h, true, "test").is_err());
    }

    #[test]
    fn axes() {
        let parse = |s: &str| Axis::try_from(s.to_string());
        assert_eq!(parse("b"), Ok(Axis::B));
        assert_eq!(parse("n"), Ok(Axis::N));
        assert!(parse("x").is_err());
        let spec = HistogramSpec {
            file: "f.root".into(),
            hist: "h".into(),
            x_axis: Axis::B,
            y_axis: Axis::B,
        };
        assert!(spec.validate().is_err());
    }
}
