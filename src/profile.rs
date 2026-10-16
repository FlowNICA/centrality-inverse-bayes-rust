//! Mean and variance of the observable `n` versus the impact parameter `b`
//! from a 2-D histogram of `n` and `b` (`Histogram` mean and variance) or
//! from the values of `b` and `n` of each event in a tree (`Tree`), e.g. of a
//! model.
//!
//! For a tree, the profile is built versus `c_b` directly: the `c_b` of an
//! event is the fraction of events with a smaller `b`. The events are grouped
//! into `b_bins` bins with equal numbers of events (equal width in `c_b`);
//! the mean of `n` of each bin (no binning of `n`) is placed at the bin
//! center in `c_b`, interpolated linearly and extrapolated linearly to
//! `c_b = 0` and 1, and the variance is that of `n` around this mean at the
//! `c_b` of each event, so the change of the mean across a bin is not counted
//! as a fluctuation. `P(b)` is that of all events in the tree, as for
//! `impact_parameter: Tree`.
//!
//! For a histogram:
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

/// Default number of bins of `b` for a tree.
pub const DEFAULT_TREE_B_BINS: usize = 100;
/// Fewest events per bin of `b` for a tree.
const MIN_EVENTS_PER_BIN: usize = 10;

/// Branches with `b` (fm) and `n` of all inelastic events in a tree.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeSpec {
    pub file: PathBuf,
    pub tree: String,
    pub b_branch: String,
    pub n_branch: String,
    /// Bins of `b` with equal numbers of events (default
    /// [`DEFAULT_TREE_B_BINS`]).
    pub b_bins: Option<usize>,
}

impl TreeSpec {
    pub fn validate(&self) -> Result<()> {
        if self.b_bins.is_some_and(|n| n < 2) {
            return Err(Error::Config(format!(
                "tree '{}': b_bins must be at least 2",
                self.tree
            )));
        }
        Ok(())
    }

    /// RON form of the fields.
    pub fn ron_fields(&self) -> String {
        let mut s = format!(
            "file: {:?}, tree: {:?}, b_branch: {:?}, n_branch: {:?}",
            self.file.display().to_string(),
            self.tree,
            self.b_branch,
            self.n_branch
        );
        if let Some(n) = self.b_bins {
            s.push_str(&format!(", b_bins: {n}"));
        }
        s
    }

    /// Reads the branches: the mean and variance of `n` in bins of `b`, and
    /// `P(b)` of all events.
    pub fn build(&self) -> Result<CbProfile> {
        self.validate()?;
        let f = FileReader::open(&self.file)
            .map_err(|e| Error::Input(format!("cannot open {}: {e}", self.file.display())))?;
        let t = TreeReader::open(&f, &self.tree).map_err(|e| {
            Error::Input(format!(
                "cannot read tree '{}' from {}: {e}",
                self.tree,
                self.file.display()
            ))
        })?;
        let b = crate::output::numbers(&t, &f, &self.b_branch)?;
        let n = crate::output::numbers(&t, &f, &self.n_branch)?;
        let description = format!(
            "branches '{}' (b) and '{}' (n) of tree '{}' in {}",
            self.b_branch,
            self.n_branch,
            self.tree,
            self.file.display()
        );
        Ok(CbProfile {
            profile: BProfile::from_values(
                &b,
                &n,
                self.b_bins.unwrap_or(DEFAULT_TREE_B_BINS),
                &description,
            )?,
            impact: ImpactProfile::from_values(&b, format!("P(b) of {description}"))?,
        })
    }
}

/// Where the mean or variance of `n` versus `b` come from.
#[derive(Debug, Clone, PartialEq)]
pub enum ProfileSpec {
    Histogram(HistogramSpec),
    Tree(TreeSpec),
}

impl ProfileSpec {
    pub fn validate(&self) -> Result<()> {
        match self {
            ProfileSpec::Histogram(s) => s.validate(),
            ProfileSpec::Tree(s) => s.validate(),
        }
    }

    pub fn build(&self) -> Result<CbProfile> {
        match self {
            ProfileSpec::Histogram(s) => s.build(),
            ProfileSpec::Tree(s) => s.build(),
        }
    }
}

/// Mean and variance of `n` in the bins of `b`.
#[derive(Debug, Clone, PartialEq)]
pub struct BProfile {
    /// Centers of the filled bins of `b` (of `c_b` if `in_cb`), ascending.
    b: Vec<f64>,
    mean: Vec<f64>,
    /// Centers of the bins of `b` with a positive variance of `n`, ascending.
    b_var: Vec<f64>,
    variance: Vec<f64>,
    /// The points are at values of `c_b` instead of `b` (from a tree); they
    /// are then extrapolated linearly to `c_b = 0` and 1.
    in_cb: bool,
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
            in_cb: false,
            description: description.to_string(),
        })
    }

    /// From the values of `b` and `n` of each event, versus `c_b`: the
    /// `c_b` of an event is the fraction of events with a smaller `b` (its
    /// rank), and the events are grouped into `bins` bins of equal numbers of
    /// events, i.e. of equal width in `c_b`. The mean of `n` of each bin is
    /// placed at the center of the bin in `c_b`, linearly interpolated
    /// between them and extrapolated linearly to `c_b = 0` (where the mean
    /// sets the knee) and to 1. The
    /// variance of a bin is that of `n` around this interpolated mean at the
    /// `c_b` of each event, so that the change of the mean across the bin
    /// does not count as a fluctuation.
    pub fn from_values(b: &[f64], n: &[f64], bins: usize, description: &str) -> Result<Self> {
        if b.len() != n.len() {
            return Err(Error::Input(format!(
                "{description}: {} values of b but {} of n",
                b.len(),
                n.len()
            )));
        }
        if b.iter().chain(n).any(|v| !v.is_finite()) || b.iter().any(|v| *v < 0.) {
            return Err(Error::Input(format!(
                "{description}: needs finite values, and b >= 0"
            )));
        }
        if b.len() < bins * MIN_EVENTS_PER_BIN {
            return Err(Error::Input(format!(
                "{description}: {} events are too few for {bins} bins of b (at least \
                 {MIN_EVENTS_PER_BIN} per bin)",
                b.len()
            )));
        }
        let mut events: Vec<(f64, f64)> = b.iter().copied().zip(n.iter().copied()).collect();
        events.sort_by(|x, y| x.0.total_cmp(&y.0));
        let total = events.len();
        let cb = |i: usize| (i as f64 + 0.5) / total as f64;
        let range = |k: usize| k * total / bins..(k + 1) * total / bins;
        let (mut centers, mut mean) = (Vec::with_capacity(bins), Vec::with_capacity(bins));
        for k in 0..bins {
            let r = range(k);
            let count = r.len() as f64;
            centers.push(r.clone().map(cb).sum::<f64>() / count);
            mean.push(events[r].iter().map(|e| e.1).sum::<f64>() / count);
        }
        let (mut c_var, mut variance) = (Vec::new(), Vec::new());
        for k in 0..bins {
            let r = range(k);
            let count = r.len() as f64;
            let v = r
                .clone()
                .map(|i| (events[i].1 - interpolate_cb(&centers, &mean, cb(i))).powi(2))
                .sum::<f64>()
                / (count - 1.);
            if v > 0. {
                c_var.push(centers[k]);
                variance.push(v);
            }
        }
        if variance.is_empty() {
            return Err(Error::Input(format!(
                "{description}: n has no spread in any bin of b"
            )));
        }
        Ok(BProfile {
            b: centers,
            mean,
            b_var: c_var,
            variance,
            in_cb: true,
            description: description.to_string(),
        })
    }

    /// Mean of `n` at `b`.
    pub fn mean_at(&self, b: f64) -> f64 {
        if self.in_cb {
            interpolate_cb(&self.b, &self.mean, b).max(0.)
        } else {
            interpolate(&self.b, &self.mean, b)
        }
    }

    /// Variance of `n` at `b`.
    pub fn variance_at(&self, b: f64) -> f64 {
        if self.in_cb {
            interpolate_cb(&self.b_var, &self.variance, b).max(0.)
        } else {
            interpolate(&self.b_var, &self.variance, b)
        }
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
    /// The value of `b` or `c_b` at which the profile is tabulated.
    fn abscissa(&self, cb: f64) -> f64 {
        if self.profile.in_cb {
            cb
        } else {
            self.impact.b_of_cb(cb)
        }
    }

    pub fn mean(&self, cb: f64) -> f64 {
        self.profile.mean_at(self.abscissa(cb))
    }

    pub fn variance(&self, cb: f64) -> f64 {
        self.profile.variance_at(self.abscissa(cb))
    }
}

/// Linear interpolation of `y(c)` through ascending points, extrapolated
/// linearly beyond the first and last points (by half a bin at most, to
/// `c_b = 0` and 1).
fn interpolate_cb(c: &[f64], y: &[f64], at: f64) -> f64 {
    let n = c.len();
    if n >= 2 && at < c[0] {
        return y[0] + (y[1] - y[0]) * (at - c[0]) / (c[1] - c[0]);
    }
    if n >= 2 && at > c[n - 1] {
        return y[n - 1] + (y[n - 1] - y[n - 2]) * (at - c[n - 1]) / (c[n - 1] - c[n - 2]);
    }
    interpolate(c, y, at)
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
    fn mean_and_variance_from_the_values_of_each_event() {
        /* b uniform in [0, 10] (c_b = b/10, as in the tree itself) and
         * n = 100 c_b + noise ±1 (variance 1) */
        let (mut b, mut n) = (Vec::new(), Vec::new());
        for i in 0..10_000 {
            let bi = 10. * (i as f64 + 0.5) / 10_000.;
            b.push(bi);
            n.push(10. * bi + if i % 2 == 0 { 1. } else { -1. });
        }
        let p = BProfile::from_values(&b, &n, 50, "test").unwrap();
        assert_eq!(p.b.len(), 50);
        /* points at the centers of the bins in c_b, the first at 0.01 */
        assert!((p.b[0] - 0.01).abs() < 1e-9, "{}", p.b[0]);
        for c in [0., 0.002, 0.05, 0.23, 0.77] {
            assert!(
                (p.mean_at(c) - 100. * c).abs() < 1e-6,
                "{c}: {}",
                p.mean_at(c)
            );
        }
        /* the change of the mean across a bin (100 · 0.02 = 2, which would
         * add 2²/12 = 0.33) is not counted as a fluctuation */
        for c in [0.005, 0.5, 0.995] {
            assert!(
                (p.variance_at(c) - 1.).abs() < 0.01,
                "{c}: {}",
                p.variance_at(c)
            );
        }
        assert!(BProfile::from_values(&b[..10], &n[..10], 50, "test").is_err());
        assert!(BProfile::from_values(&b, &n[..9_999], 50, "test").is_err());
        let mut negative = b.clone();
        negative[0] = -1.;
        assert!(BProfile::from_values(&negative, &n, 50, "test").is_err());
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
