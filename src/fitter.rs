//! Fit of the measured distribution `P(n)` with the inverse Bayes model, using
//! Minuit (Migrad + Hesse) from the `minuit2` crate.
//!
//! Each evaluation of the fit statistic integrates the kernel over `c_b` for
//! all bins in the fit range; the quadrature nodes are processed in parallel.
//! The normalization of the model is profiled analytically, so Minuit only
//! sees the shape parameters.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use minuit2::{FCN, FunctionMinimum, MnHesse, MnMigrad, MnMinimize};
use rayon::prelude::*;

use crate::config::{FitConfig, FitMethod, Normalization, ParamConfig};
use crate::data::DataHistogram;
use crate::error::{Error, Result};
use crate::model::{BinModel, MeanModel, Model, Quadrature};
use crate::numeric::solve;

/// Statistic returned for parameters where the model is undefined.
const PENALTY: f64 = 1e30;

/// Progress of a fit, reported to a callback.
#[derive(Debug, Clone)]
pub enum FitProgress {
    /// A new stage of the fit begins.
    Stage(String),
    /// The best statistic so far, after `calls` evaluations.
    Calls { calls: usize, best: f64 },
}

impl FitProgress {
    /// Prints stages to stdout and every `Calls` event to stderr.
    pub fn print(p: FitProgress) {
        match p {
            FitProgress::Stage(s) => println!("{s}"),
            FitProgress::Calls { calls, best } => {
                eprintln!("  {calls} calls, best statistic {best:.6}")
            }
        }
    }
}

/// Result of the fit.
#[derive(Debug, Clone)]
pub struct FitResult {
    pub model: Model,
    pub names: Vec<String>,
    pub values: Vec<f64>,
    pub errors: Vec<f64>,
    pub fixed: Vec<bool>,
    /// Covariance of all parameters, row-major (zero rows for fixed ones).
    pub covariance: Vec<f64>,
    pub method: FitMethod,
    /// Minimized statistic (χ² or -2 ln(L/L_sat)).
    pub statistic: f64,
    pub ndf: usize,
    /// Number of events `N` of the model (all centralities).
    pub norm: f64,
    /// Entries of the data histogram.
    pub data_total: f64,
    /// Whether Minuit reports a valid minimum.
    pub valid: bool,
    pub edm: f64,
    pub n_calls: usize,
    /// Lower and upper edge of the fit range.
    pub fit_range: (f64, f64),
    /// `n̄(c_b = 0)`.
    pub n_knee: f64,
    /// `σ(c_b = 0)`.
    pub sigma_knee: f64,
    /// Centrality of the knee: fraction of (single collision) events with
    /// `n > n_knee`.
    pub c_knee: f64,
    /// Expected counts `N · P(bin)` of the recorded events in all data bins.
    pub model_counts: Vec<f64>,
    /// The part of `model_counts` from single collisions, `N (1 - p) P_single`.
    pub single_counts: Vec<f64>,
    /// The part from pile-up, `N p P_pile-up` (`None` without pile-up).
    pub pile_up_counts: Option<Vec<f64>>,
}

impl FitResult {
    /// `statistic / ndf`.
    pub fn reduced_statistic(&self) -> f64 {
        self.statistic / self.ndf.max(1) as f64
    }
}

/// The fit of one data histogram.
pub struct Fitter {
    config: FitConfig,
    model: Model,
    data: DataHistogram,
    quad: Quadrature,
    /// First and last fit bin (0-based, inclusive).
    fit_bins: (usize, usize),
}

impl Fitter {
    /// Reads the data histogram given in `config`.
    pub fn new(config: FitConfig) -> Result<Self> {
        let data = DataHistogram::read(&config.data_file, &config.data_hist)?;
        Self::with_data(config, data)
    }

    pub fn with_data(config: FitConfig, data: DataHistogram) -> Result<Self> {
        config.validate()?;
        let fit_bins = data.fit_bins(config.fit_min, config.fit_max)?;
        if config.fit_method == FitMethod::Likelihood && data.weighted {
            eprintln!(
                "Warning: the errors of '{}' are not Poisson (weighted or scaled histogram); \
                 the likelihood fit assumes raw counts",
                data.hist.name
            );
        }
        if config.bin_model == BinModel::Center && data.edges[fit_bins.0] < 0. {
            eprintln!("Warning: bins below 0 are not described by the model");
        }
        Ok(Fitter {
            model: config.model()?,
            quad: Quadrature::new(config.cb_points),
            config,
            data,
            fit_bins,
        })
    }

    pub fn config(&self) -> &FitConfig {
        &self.config
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn data(&self) -> &DataHistogram {
        &self.data
    }

    pub fn quadrature(&self) -> &Quadrature {
        &self.quad
    }

    /// The fit statistic as a function of the parameters.
    pub fn objective(&self) -> Objective<'_> {
        let (lo, hi) = self.fit_bins;
        Objective {
            model: self.model.clone(),
            quad: &self.quad,
            edges: &self.data.edges[lo..=hi + 1],
            counts: &self.data.counts[lo..=hi],
            errors: &self.data.errors[lo..=hi],
            method: self.config.fit_method,
            normalization: self.config.normalization,
            data_total: self.data.total,
            bin_model: self.config.bin_model,
            calls: AtomicUsize::new(0),
            best: Mutex::new(f64::INFINITY),
            progress: None,
        }
    }

    /// Runs the fit, printing the progress to stdout.
    pub fn fit(&self) -> Result<FitResult> {
        self.fit_with_progress(&FitProgress::print)
    }

    /// Runs the fit, reporting the progress to `progress`.
    pub fn fit_with_progress(&self, progress: &(dyn Fn(FitProgress) + Sync)) -> Result<FitResult> {
        let names = self.model.param_names();
        let user: Vec<ParamConfig> = names
            .iter()
            .map(|n| self.config.params.get(n).cloned().unwrap_or_default())
            .collect();
        let (lo, hi) = self.fit_bins;
        progress(FitProgress::Stage(format!(
            "Fit of '{}' with {} in [{}, {}) ({} bins), {} c_b points",
            self.data.hist.name,
            self.model.describe(),
            self.data.edges[lo],
            self.data.edges[hi + 1],
            hi + 1 - lo,
            self.quad.len()
        )));

        let start = self.initial_values(&user, progress)?;
        progress(FitProgress::Stage(format!(
            "Start values: {}",
            format_params(&names, &start)
        )));

        let mut objective = self.objective();
        objective.progress = Some(progress);
        let specs = param_specs(&self.model, &names, &start, &user);

        progress(FitProgress::Stage("Migrad".into()));
        let mut minimum = run_migrad(&objective, &specs, &self.config);
        if !minimum.is_valid() {
            progress(FitProgress::Stage(
                "Migrad did not converge, retrying with Simplex + Migrad".into(),
            ));
            let restart: Vec<ParamSpec> = specs
                .iter()
                .zip(minimum.user_state().params().params())
                .map(|(s, p)| ParamSpec {
                    value: p.value(),
                    ..s.clone()
                })
                .collect();
            let second = run_minimize(&objective, &restart, &self.config);
            if second.is_valid() || second.fval() < minimum.fval() {
                minimum = second;
            }
        }
        progress(FitProgress::Stage("Hesse".into()));
        let minimum = MnHesse::new()
            .with_strategy(self.config.strategy)
            .calculate(&objective, &minimum);

        let state = minimum.user_state();
        let params = state.params().params();
        let values: Vec<f64> = params.iter().map(|p| p.value()).collect();
        let fixed: Vec<bool> = specs.iter().map(|s| s.fixed).collect();
        let errors: Vec<f64> = params
            .iter()
            .zip(&fixed)
            .map(|(p, &f)| if f { 0. } else { p.error() })
            .collect();
        let n = values.len();
        let mut covariance = vec![0.; n * n];
        if let Some(cov) = state.covariance() {
            let trafo = state.params().trafo();
            for i in 0..cov.nrow() {
                for j in 0..cov.nrow() {
                    let (ei, ej) = (trafo.ext_of_int(i), trafo.ext_of_int(j));
                    covariance[ei * n + ej] = cov.get(i, j);
                }
            }
        } else {
            for i in 0..n {
                covariance[i * n + i] = errors[i] * errors[i];
            }
        }

        let eval = objective
            .evaluate(&values)
            .ok_or_else(|| Error::Fit(format!("invalid parameters at the minimum: {values:?}")))?;
        let n_free = fixed.iter().filter(|f| !**f).count()
            + usize::from(self.config.normalization == Normalization::Free);
        let ndf = objective.n_used_bins().saturating_sub(n_free);

        let (single, pile_up) = self.model.bin_probability_parts(
            &values,
            &self.quad,
            &self.data.edges,
            self.config.bin_model,
        );
        let counts = |v: &[f64]| v.iter().map(|p| p * eval.norm).collect::<Vec<f64>>();
        let single_counts = counts(&single);
        let pile_up_counts = pile_up.as_deref().map(counts);
        let model_counts = match &pile_up_counts {
            Some(d) => single_counts.iter().zip(d).map(|(s, d)| s + d).collect(),
            None => single_counts.clone(),
        };
        let n_knee = self.model.n_knee(&values);
        let sigma_knee = self.model.variance(&values, n_knee).sqrt();
        Ok(FitResult {
            model: self.model.clone(),
            names,
            errors,
            fixed,
            covariance,
            method: self.config.fit_method,
            statistic: eval.statistic,
            ndf,
            norm: eval.norm,
            data_total: self.data.total,
            valid: minimum.is_valid(),
            edm: minimum.edm(),
            n_calls: objective.calls.load(Ordering::Relaxed),
            fit_range: (self.data.edges[lo], self.data.edges[hi + 1]),
            n_knee,
            sigma_knee,
            c_knee: self.model.centrality(&values, &self.quad, n_knee),
            model_counts,
            single_counts,
            pile_up_counts,
            values,
        })
    }

    /// Start values: user values where given; otherwise the mean is
    /// pre-fitted to the data in the limit of vanishing fluctuations, where
    /// `n̄(c)` is the inverse of the measured centrality `c(n)` (Das et al.),
    /// and `θ` and the overall scale of the mean are scanned on a grid.
    pub fn initial_values(
        &self,
        user: &[ParamConfig],
        progress: &(dyn Fn(FitProgress) + Sync),
    ) -> Result<Vec<f64>> {
        let n = self.model.n_params();
        let n_mean = self.model.n_mean_params();
        let mut start = vec![0.; n];
        let prefit = self.prefit_mean()?;
        start[1..=n_mean].copy_from_slice(&prefit);
        let pile_up = self.model.pile_up_index();
        if let Some(i) = pile_up {
            start[i] = 1e-3;
        }
        for (s, u) in start.iter_mut().zip(user) {
            if let Some(v) = u.value {
                *s = v;
            }
        }
        clamp_to_limits(&mut start, user);
        let theta_given = user[0].value.is_some();
        let mean_given = user[1..=n_mean].iter().any(|u| u.value.is_some());
        let pile_up_given = pile_up.is_none_or(|i| user[i].value.is_some());
        let n_knee = self.model.n_knee(&start);
        if theta_given && mean_given && pile_up_given {
            return Ok(start);
        }
        let objective = self.objective();
        let best_of = |candidates: Vec<Vec<f64>>| -> Vec<f64> {
            candidates
                .par_iter()
                .map(|p| (objective.value(p), p))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, p)| p.clone())
                .expect("at least one candidate")
        };
        if !(theta_given && mean_given) {
            start =
                best_of(self.width_scan(&start, n_knee, theta_given, mean_given, user, progress));
        }
        /* then the pile-up probability, at the best width and scale */
        if let (Some(i), false) = (pile_up, pile_up_given) {
            let candidates = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
                .iter()
                .map(|&f| {
                    let mut p = start.clone();
                    p[i] = f;
                    clamp_to_limits(&mut p, user);
                    p
                })
                .collect();
            start = best_of(candidates);
        }
        Ok(start)
    }

    /// Start points on a grid of `θ` (relative width `σ(0)/n_knee` from 0.4%
    /// to 60%) and of the scale of the mean.
    fn width_scan(
        &self,
        start: &[f64],
        n_knee: f64,
        theta_given: bool,
        mean_given: bool,
        user: &[ParamConfig],
        progress: &(dyn Fn(FitProgress) + Sync),
    ) -> Vec<Vec<f64>> {
        /* relative width σ(0)/n_knee and scale of the mean */
        let widths: Vec<f64> = if theta_given {
            vec![f64::NAN]
        } else {
            (0..16)
                .map(|i| 0.004 * (0.6f64 / 0.004).powf(i as f64 / 15.))
                .collect()
        };
        let scales: &[f64] = if mean_given {
            &[1.]
        } else {
            &[0.9, 0.95, 1., 1.05, 1.1]
        };
        progress(FitProgress::Stage(format!(
            "Scanning {} start points",
            widths.len() * scales.len()
        )));
        widths
            .iter()
            .flat_map(|&r| {
                scales.iter().map(move |&s| {
                    let mut p = start.to_vec();
                    scale_mean(&self.model, &mut p, s);
                    if r.is_finite() {
                        /* σ(0)² = θ n̄(0) for both variance models */
                        p[0] = r * r * n_knee * s;
                    }
                    clamp_to_limits(&mut p, user);
                    p
                })
            })
            .collect()
    }

    /// Mean parameters fitted to the inverse of the measured centrality, using
    /// the bins in the fit range between 2% and 95% centrality.
    fn prefit_mean(&self) -> Result<Vec<f64>> {
        let fractions = self.data.upper_fractions();
        let (lo, hi) = self.fit_bins;
        let point = |i: usize| (fractions[i], self.data.center(i));
        let usable = |i: &usize| {
            self.data.counts[*i] > 0. && self.data.center(*i) > 0. && fractions[*i] > 0.
        };
        let mut points: Vec<(f64, f64)> = (lo..=hi)
            .filter(usable)
            .map(point)
            .filter(|(c, _)| (0.02..=0.95).contains(c))
            .collect();
        if points.len() < self.model.n_params() + 1 {
            points = (lo..=hi).filter(usable).map(point).collect();
        }
        if points.len() < 2 {
            return Err(Error::Input(format!(
                "too few filled bins with positive values in the fit range of '{}'",
                self.data.hist.name
            )));
        }
        match self.model.mean {
            MeanModel::ExpPoly(n) => Ok(prefit_exp_poly(&points, n)),
            MeanModel::Indra => Ok(prefit_indra(&points)),
            MeanModel::Glauber { .. } => Ok(prefit_glauber(
                &points,
                self.model.glauber_table().expect("Glauber table"),
            )),
        }
    }

    /// Expected counts `N · P(bin)` in all data bins, for parameters `p` and
    /// normalization `norm`.
    pub fn model_counts(&self, p: &[f64], norm: f64) -> Vec<f64> {
        self.model
            .bin_probabilities(p, &self.quad, &self.data.edges, self.config.bin_model)
            .into_iter()
            .map(|v| v * norm)
            .collect()
    }
}

/// `ln n̄ = ln n_knee - Σ a_j c^j`, a linear least squares problem; if it is
/// singular, a pure exponential through the end points is used.
fn prefit_exp_poly(points: &[(f64, f64)], n: usize) -> Vec<f64> {
    let m = n + 1;
    let row = |c: f64| -> Vec<f64> {
        std::iter::once(1.)
            .chain((1..=n).map(|j| -c.powi(j as i32)))
            .collect()
    };
    let mut ata = vec![0.; m * m];
    let mut atb = vec![0.; m];
    for &(c, x) in points {
        let r = row(c);
        for i in 0..m {
            atb[i] += r[i] * x.ln();
            for j in 0..m {
                ata[i * m + j] += r[i] * r[j];
            }
        }
    }
    let fitted = if points.len() >= m {
        solve(ata, atb)
    } else {
        None
    };
    match fitted {
        Some(beta) => std::iter::once(beta[0].exp())
            .chain(beta[1..].iter().copied())
            .collect(),
        None => {
            let (c0, x0) = points[0];
            let (c1, x1) = points[points.len() - 1];
            let a1 = if c1 > c0 {
                (x0 / x1).ln() / (c1 - c0)
            } else {
                3.
            };
            let mut p = vec![x0 * (a1 * c0).exp(), a1];
            p.resize(m, 0.);
            p
        }
    }
}

/// `n̄ = μ [f N_part + (1 - f) N_coll]` fitted to the points: linear least
/// squares in `μf` and `μ(1 - f)` with relative residuals, `f` clamped to
/// `[0, 1]`.
fn prefit_glauber(points: &[(f64, f64)], table: &crate::optical::GlauberMean) -> Vec<f64> {
    /* minimize Σ ((x - α N_part - β N_coll) / x)² */
    let (mut s11, mut s12, mut s22, mut r1, mut r2) = (0., 0., 0., 0., 0.);
    for &(c, x) in points {
        let (np, nc) = table.at(c);
        let (u, v) = (np / x, nc / x);
        s11 += u * u;
        s12 += u * v;
        s22 += v * v;
        r1 += u;
        r2 += v;
    }
    let f = match solve(vec![s11, s12, s12, s22], vec![r1, r2]) {
        Some(ab) if ab[0] + ab[1] > 0. => (ab[0] / (ab[0] + ab[1])).clamp(0., 1.),
        _ => 0.5,
    };
    /* μ for this f: minimize Σ ((x - μ g) / x)² */
    let (mut num, mut den) = (0., 0.);
    for &(c, x) in points {
        let (np, nc) = table.at(c);
        let g = (f * np + (1. - f) * nc) / x;
        num += g;
        den += g * g;
    }
    vec![if den > 0. { num / den } else { 1. }, f]
}

/// `n̄ = x_max (1 - c^α)^γ + x_min` fitted to the logarithm of the points.
fn prefit_indra(points: &[(f64, f64)]) -> Vec<f64> {
    let x_top = points
        .iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(1., |p| p.1);
    let model = Model::new(
        crate::model::Kernel::Gamma,
        MeanModel::Indra,
        crate::model::VarianceModel::Proportional,
        false,
    )
    .expect("the Indra model needs no table");
    let fcn = |q: &[f64]| -> f64 {
        let p = [1., q[0], q[1], q[2], q[3]];
        if !model.is_valid(&p) {
            return PENALTY;
        }
        points
            .iter()
            .map(|&(c, x)| {
                let m = model.mean(&p, c);
                if m > 0. {
                    (m.ln() - x.ln()).powi(2)
                } else {
                    1e6
                }
            })
            .sum()
    };
    let min = MnMinimize::new()
        .add_lower_limited("x_max", x_top, 0.1 * x_top, 0.)
        .add_lower_limited("x_min", 0.01 * x_top, 0.01 * x_top, 0.)
        .add_lower_limited("alpha", 1., 0.1, 1e-3)
        .add_lower_limited("gamma", 1., 0.1, 1e-3)
        .minimize(&fcn);
    min.user_state()
        .params()
        .params()
        .iter()
        .map(|p| p.value())
        .collect()
}

/// Multiplies the mean `n̄(c_b)` by `s`.
fn scale_mean(model: &Model, p: &mut [f64], s: f64) {
    match model.mean {
        MeanModel::ExpPoly(_) => p[1] *= s,
        MeanModel::Indra => {
            p[1] *= s;
            p[2] *= s;
        }
        MeanModel::Glauber { .. } => p[1] *= s,
    }
}

fn clamp_to_limits(p: &mut [f64], user: &[ParamConfig]) {
    for (v, u) in p.iter_mut().zip(user) {
        if let Some(lo) = u.min {
            *v = v.max(lo);
        }
        if let Some(hi) = u.max {
            *v = v.min(hi);
        }
    }
}

/// A parameter as given to Minuit.
#[derive(Debug, Clone)]
struct ParamSpec {
    name: String,
    value: f64,
    step: f64,
    min: Option<f64>,
    max: Option<f64>,
    fixed: bool,
}

/// Steps and limits: user settings where given, otherwise positivity of `θ`
/// and of the scale parameters of the mean.
fn param_specs(
    model: &Model,
    names: &[String],
    start: &[f64],
    user: &[ParamConfig],
) -> Vec<ParamSpec> {
    names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let v = start[i];
            let is_pile_up = model.pile_up_index() == Some(i);
            let glauber = matches!(model.mean, MeanModel::Glauber { .. });
            /* the fraction f of the Glauber mean is limited to [0, 1] */
            let fraction = glauber && i == 2;
            let positive = match model.mean {
                _ if is_pile_up => true,
                MeanModel::ExpPoly(_) => i <= 1,
                MeanModel::Indra | MeanModel::Glauber { .. } => true,
            };
            let default_step = match (&model.mean, i) {
                _ if is_pile_up => 0.5 * v.abs().max(1e-4),
                _ if fraction => 0.05,
                (_, 0) => 0.1 * v.abs(),
                (_, 1) => 0.02 * v.abs(),
                (MeanModel::Indra, 2) => 0.1 * v.abs().max(1e-3 * start[1].abs()),
                _ => 0.1 * v.abs().max(0.1),
            };
            let u = &user[i];
            ParamSpec {
                name: name.clone(),
                value: v,
                step: u.step.unwrap_or(default_step).max(1e-12),
                min: u.min.or(positive.then_some(0.)),
                max: u
                    .max
                    .or(is_pile_up.then_some(0.5))
                    .or(fraction.then_some(1.)),
                fixed: u.fixed,
            }
        })
        .collect()
}

/// Adds the parameters to a Migrad or Minimize builder (they share the method
/// names but no public trait).
macro_rules! add_params {
    ($builder:expr, $specs:expr) => {{
        let mut b = $builder;
        for s in $specs {
            b = if s.fixed {
                b.add_const(s.name.as_str(), s.value)
            } else {
                match (s.min, s.max) {
                    (Some(lo), Some(hi)) => b.add_limited(s.name.as_str(), s.value, s.step, lo, hi),
                    (Some(lo), None) => b.add_lower_limited(s.name.as_str(), s.value, s.step, lo),
                    (None, Some(hi)) => b.add_upper_limited(s.name.as_str(), s.value, s.step, hi),
                    (None, None) => b.add(s.name.as_str(), s.value, s.step),
                }
            };
        }
        b
    }};
}

fn run_migrad(fcn: &dyn FCN, specs: &[ParamSpec], config: &FitConfig) -> FunctionMinimum {
    let mut migrad = add_params!(MnMigrad::new().with_strategy(config.strategy), specs);
    if let Some(n) = config.max_calls {
        migrad = migrad.max_fcn(n);
    }
    if let Some(t) = config.tolerance {
        migrad = migrad.tolerance(t);
    }
    migrad.minimize(fcn)
}

fn run_minimize(fcn: &dyn FCN, specs: &[ParamSpec], config: &FitConfig) -> FunctionMinimum {
    let mut minimize = add_params!(MnMinimize::new().with_strategy(config.strategy), specs);
    if let Some(n) = config.max_calls {
        minimize = minimize.max_fcn(n);
    }
    if let Some(t) = config.tolerance {
        minimize = minimize.tolerance(t);
    }
    minimize.minimize(fcn)
}

pub fn format_params(names: &[String], values: &[f64]) -> String {
    names
        .iter()
        .zip(values)
        .map(|(n, v)| format!("{n} = {v:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Value of the statistic and the profiled normalization.
#[derive(Debug, Clone, Copy)]
pub struct Evaluation {
    pub statistic: f64,
    pub norm: f64,
}

/// The fit statistic for the bins in the fit range.
pub struct Objective<'a> {
    model: Model,
    quad: &'a Quadrature,
    edges: &'a [f64],
    counts: &'a [f64],
    errors: &'a [f64],
    method: FitMethod,
    normalization: Normalization,
    data_total: f64,
    bin_model: BinModel,
    calls: AtomicUsize,
    best: Mutex<f64>,
    progress: Option<&'a (dyn Fn(FitProgress) + Sync)>,
}

impl Objective<'_> {
    /// Statistic and normalization, `None` for invalid parameters.
    pub fn evaluate(&self, p: &[f64]) -> Option<Evaluation> {
        if !self.model.is_valid(p) {
            return None;
        }
        let probs = self
            .model
            .bin_probabilities(p, self.quad, self.edges, self.bin_model);
        let norm = match (self.normalization, self.method) {
            (Normalization::Data, _) => self.data_total,
            (Normalization::Free, FitMethod::Chi2) => {
                let (mut num, mut den) = (0., 0.);
                for ((&d, &e), &q) in self.counts.iter().zip(self.errors).zip(&probs) {
                    if e > 0. {
                        num += d * q / (e * e);
                        den += q * q / (e * e);
                    }
                }
                num / den
            }
            (Normalization::Free, FitMethod::Likelihood) => {
                self.counts.iter().sum::<f64>() / probs.iter().sum::<f64>()
            }
        };
        if !(norm.is_finite() && norm > 0.) {
            return None;
        }
        let statistic = match self.method {
            FitMethod::Chi2 => self
                .counts
                .iter()
                .zip(self.errors)
                .zip(&probs)
                .filter(|((_, e), _)| **e > 0.)
                .map(|((&d, &e), &q)| ((d - norm * q) / e).powi(2))
                .sum(),
            FitMethod::Likelihood => {
                2. * self
                    .counts
                    .iter()
                    .zip(&probs)
                    .map(|(&d, &q)| {
                        let m = (norm * q).max(1e-300);
                        if d > 0. { m - d + d * (d / m).ln() } else { m }
                    })
                    .sum::<f64>()
            }
        };
        statistic
            .is_finite()
            .then_some(Evaluation { statistic, norm })
    }

    /// Number of bins entering the statistic.
    pub fn n_used_bins(&self) -> usize {
        match self.method {
            FitMethod::Chi2 => self.errors.iter().filter(|e| **e > 0.).count(),
            FitMethod::Likelihood => self.counts.len(),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl FCN for Objective<'_> {
    fn value(&self, p: &[f64]) -> f64 {
        let v = self.evaluate(p).map_or(PENALTY, |e| e.statistic);
        let calls = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(progress) = self.progress {
            let mut best = self.best.lock().expect("progress lock");
            if v < *best {
                *best = v;
            }
            if calls.is_multiple_of(100) {
                progress(FitProgress::Calls { calls, best: *best });
            }
        }
        v
    }

    /// Both statistics are `-2 ln L` up to a constant.
    fn error_def(&self) -> f64 {
        1.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_poly_prefit_recovers_exact_curve() {
        let truth = [200., 3., -1., 2.];
        let points: Vec<(f64, f64)> = (1..50)
            .map(|i| {
                let c = i as f64 / 50.;
                (
                    c,
                    truth[0] * (-(truth[1] * c + truth[2] * c * c + truth[3] * c * c * c)).exp(),
                )
            })
            .collect();
        let p = prefit_exp_poly(&points, 3);
        for (a, b) in p.iter().zip(&truth) {
            assert!((a - b).abs() < 1e-8 * b.abs().max(1.), "{p:?}");
        }
    }

    #[test]
    fn indra_prefit_recovers_curve() {
        let model = Model::new(
            crate::model::Kernel::Gamma,
            MeanModel::Indra,
            crate::model::VarianceModel::Proportional,
            false,
        )
        .unwrap();
        let truth = [1., 40., 3., 1.5, 1.3];
        let points: Vec<(f64, f64)> = (1..40)
            .map(|i| {
                let c = i as f64 / 40.;
                (c, model.mean(&truth, c))
            })
            .collect();
        let p = prefit_indra(&points);
        let q = [1., p[0], p[1], p[2], p[3]];
        for &(c, x) in &points {
            assert!((model.mean(&q, c) - x).abs() < 0.01 * x, "{p:?}");
        }
    }
}
