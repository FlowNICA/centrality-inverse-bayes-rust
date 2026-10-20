//! Centrality classes from the fitted model.
//!
//! The centrality of a value `x` of the observable is the fraction of events
//! above it, `c(x) = ∫₀¹ P(n >= x | c_b) dc_b`. The class borders are the
//! values with the requested fractions above them. Within a class
//! `x_lo <= n < x_hi`, Bayes' theorem gives (Rogly et al., Frankland et al.)
//!
//! ```text
//! P(c_b | class) = [P(n >= x_lo | c_b) - P(n >= x_hi | c_b)] / [c(x_lo) - c(x_hi)]
//! P(b | class)   = P(b) · P(c_b(b) | class)
//! ```
//!
//! from which the mean and RMS of `c_b` and `b` in each class follow.

use std::path::PathBuf;

use oxiroot::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::config::{BorderMode, CentralityConfig};
use crate::error::{Error, Result};
use crate::impact::ImpactProfile;
use crate::model::{Model, NodeKernel, Quadrature};
use crate::numeric::cholesky;
use crate::output::{
    StoredFit, best_result_tree, fit_histogram, graph, histogram_2d, histogram_from_edges,
    histogram_like, joint_counts, read_fit, set_2d, uniform_histogram,
};

/// Mean, RMS and the range selected by a sharp cut at the class percents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub mean: f64,
    pub rms: f64,
    pub min: f64,
    pub max: f64,
}

/// Standard deviations from the propagation of the fit uncertainties.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClassErrors {
    /// Of the exact borders with the nominal percents.
    pub min_border: f64,
    pub max_border: f64,
    pub cb_mean: f64,
    pub cb_rms: f64,
    pub b_mean: f64,
    pub b_rms: f64,
}

/// One centrality class.
#[derive(Debug, Clone)]
pub struct CentralityClass {
    pub number: usize,
    /// Nominal percents.
    pub min_percent: f64,
    pub max_percent: f64,
    /// Range `[min_border, max_border)` of the observable; the most central
    /// class extends to infinity, the most peripheral one to zero, but the
    /// borders show the ends of the data histogram there.
    pub min_border: f64,
    pub max_border: f64,
    /// Percents of the fitted distribution above the borders: the actual
    /// class limits (they differ from the nominal ones for snapped borders).
    pub fit_min_percent: f64,
    pub fit_max_percent: f64,
    /// Percent of the data histogram entries in the class.
    pub data_percent: f64,
    /// Trigger efficiency in the class: data entries over the recorded events
    /// expected from the fit, in the same bins.
    pub efficiency: f64,
    /// Percent of the recorded events of the fit in the class that are
    /// pile-up (0 without pile-up).
    pub pile_up_percent: f64,
    pub cb: Stats,
    pub b: Option<Stats>,
    pub errors: Option<ClassErrors>,
}

impl CentralityClass {
    pub fn label(&self) -> String {
        class_label(self.min_percent, self.max_percent)
    }
}

/// Result of the centrality determination.
#[derive(Debug, Clone)]
pub struct CentralityResult {
    pub classes: Vec<CentralityClass>,
    /// The fit file.
    pub source: PathBuf,
    /// Description of `P(b)`, if given.
    pub impact: Option<String>,
    /// Largest impact parameter of `P(b)` (fm), and the upper edge of the
    /// histograms of `b` (`b_max` of the configuration, or the former).
    pub b_range: Option<(f64, f64)>,
    pub n_knee: f64,
    pub sigma_knee: f64,
    pub c_knee: f64,
    /// Parameter sets used for the error propagation.
    pub n_error_samples: usize,
    /// Fitted pile-up probability (0 without pile-up).
    pub pile_up_probability: f64,
    /// Start of the pile-up dominated tail, excluded from the classes.
    pub pile_up_tail: Option<f64>,
    /// Data entries over all events of the fit.
    pub efficiency: f64,
    /// Normalization of the fit.
    pub normalization: crate::config::Normalization,
    /// Expected recorded events of the fit in the data bins.
    pub model_counts: Vec<f64>,
}

/// The model with fixed parameters, with the kernels at the quadrature nodes.
struct Calc<'a> {
    model: Model,
    p: &'a [f64],
    quad: &'a Quadrature,
    kernels: Vec<NodeKernel>,
}

impl<'a> Calc<'a> {
    fn new(model: Model, p: &'a [f64], quad: &'a Quadrature) -> Self {
        let kernels = quad
            .nodes
            .iter()
            .map(|&cb| model.kernel_at(p, cb))
            .collect();
        Calc {
            model,
            p,
            quad,
            kernels,
        }
    }

    /// `P(n >= x)`, 1 for `x = -∞`, 0 for `x = +∞`.
    fn centrality(&self, x: f64) -> f64 {
        if x == f64::NEG_INFINITY {
            return 1.;
        }
        self.kernels
            .iter()
            .zip(&self.quad.weights)
            .map(|(k, w)| w * k.survival(x))
            .sum()
    }

    fn density(&self, x: f64) -> f64 {
        self.kernels
            .iter()
            .zip(&self.quad.weights)
            .map(|(k, w)| w * k.pdf(x))
            .sum()
    }

    /// The value with the fraction `f` of events above it: `+∞` for 0, `-∞`
    /// for 1; Newton's method, safeguarded by bisection.
    fn border(&self, f: f64) -> f64 {
        if f <= 0. {
            return f64::INFINITY;
        }
        if f >= 1. {
            return f64::NEG_INFINITY;
        }
        let n_knee = self.model.n_knee(self.p);
        let mut hi = n_knee + 10. * self.model.sigma_knee(self.p);
        while self.centrality(hi) > f && hi < 1e300 {
            hi *= 2.;
        }
        let mut lo = 0.;
        let mut x = 0.5 * (lo + hi);
        for _ in 0..200 {
            let g = self.centrality(x) - f;
            if g > 0. {
                lo = x;
            } else {
                hi = x;
            }
            let d = self.density(x);
            let newton = x + g / d;
            let next = if d > 0. && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            if (next - x).abs() <= 1e-12 * x.abs().max(1e-300) || hi - lo <= 1e-12 * hi.abs() {
                return next;
            }
            x = next;
        }
        x
    }

    /// `P(recorded in the class | c_b)` times the node weights `w`.
    fn class_weights(&self, sel: &Selection) -> Vec<f64> {
        self.kernels
            .iter()
            .zip(&self.quad.weights)
            .map(|(k, w)| w * sel.probability(k))
            .collect()
    }

    /// Mean and RMS of `c_b` and of `b` of the recorded events in the class,
    /// and their fraction of all events.
    fn class_moments(
        &self,
        sel: &Selection,
        impact: Option<&ImpactProfile>,
    ) -> (f64, (f64, f64), Option<(f64, f64)>) {
        let weights = self.class_weights(sel);
        let total: f64 = weights.iter().sum();
        let moments = |f: &dyn Fn(f64) -> f64| {
            let (mut s1, mut s2) = (0., 0.);
            for (&cb, &w) in self.quad.nodes.iter().zip(&weights) {
                let v = f(cb);
                s1 += w * v;
                s2 += w * v * v;
            }
            let mean = s1 / total;
            (mean, (s2 / total - mean * mean).max(0.).sqrt())
        };
        let cb = moments(&|c| c);
        let b = impact.map(|ip| moments(&|c| ip.b_of_cb(c)));
        (total, cb, b)
    }

    /// Mean of `P(recorded in the class | c_b)` over `[c0, c1]` (its value at
    /// `c0` for an empty interval); for a selection of all values of the
    /// observable, the trigger efficiency of the events in `[c0, c1]`.
    fn mean_probability_in(&self, sel: &Selection, c0: f64, c1: f64) -> f64 {
        if c1 > c0 {
            self.class_probability_in(sel, c0, c1) / (c1 - c0)
        } else {
            sel.probability(&self.model.kernel_at(self.p, c0.clamp(0., 1.)))
        }
    }

    /// `∫ P(recorded in the class | c_b) dc_b` over `[c0, c1]`.
    fn class_probability_in(&self, sel: &Selection, c0: f64, c1: f64) -> f64 {
        if !(c1 > c0) {
            return 0.;
        }
        let q = Quadrature::on_interval(c0, c1, 16);
        q.nodes
            .iter()
            .zip(&q.weights)
            .map(|(&cb, &w)| w * sel.probability(&self.model.kernel_at(self.p, cb)))
            .sum()
    }
}

/// The values of the observable selected by a class, `[x_lo, x_hi)`, as
/// intervals `[bounds[j], bounds[j+1])` with the trigger efficiency
/// `weights[j]` (adjacent intervals with the same efficiency are merged).
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub bounds: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Selection {
    /// The class `[x_lo, x_hi)` for the efficiency `efficiency[i]` of the
    /// data bins (the first bin's below the histogram, 1 above it).
    pub fn new(edges: &[f64], efficiency: &[f64], x_lo: f64, x_hi: f64) -> Self {
        let mut pieces: Vec<(f64, f64, f64)> = Vec::new();
        let n_bins = edges.len() - 1;
        if x_lo < edges[0] {
            pieces.push((x_lo, edges[0].min(x_hi), efficiency[0]));
        }
        for i in 0..n_bins {
            let (lo, hi) = (edges[i].max(x_lo), edges[i + 1].min(x_hi));
            if lo < hi {
                pieces.push((lo, hi, efficiency[i]));
            }
        }
        if x_hi > edges[n_bins] {
            pieces.push((edges[n_bins].max(x_lo), x_hi, 1.));
        }
        let mut bounds = Vec::with_capacity(pieces.len() + 1);
        let mut weights: Vec<f64> = Vec::with_capacity(pieces.len());
        for (lo, hi, w) in pieces {
            if bounds.is_empty() {
                bounds.push(lo);
            }
            if weights.last() == Some(&w) {
                *bounds.last_mut().expect("bound") = hi;
            } else {
                weights.push(w);
                bounds.push(hi);
            }
        }
        if bounds.is_empty() {
            bounds = vec![x_lo, x_hi];
            weights = vec![0.];
        }
        Selection { bounds, weights }
    }

    /// `Σ_j weights[j] P(bounds[j] <= n < bounds[j+1])` for a kernel.
    fn probability(&self, k: &NodeKernel) -> f64 {
        let survival = |x: f64| {
            if x == f64::NEG_INFINITY {
                1.
            } else {
                k.survival(x)
            }
        };
        let mut prev = survival(self.bounds[0]);
        let mut sum = 0.;
        for (j, w) in self.weights.iter().enumerate() {
            let next = survival(self.bounds[j + 1]);
            sum += w * (prev - next).max(0.);
            prev = next;
        }
        sum
    }
}

/// Trigger efficiency of each data bin: 1 from the fit range on, where the
/// trigger must be fully efficient; below it, data over fit, made
/// non-decreasing with `n` (weighted pool-adjacent-violators, with the
/// expected counts as weights, which also smooths the statistical
/// fluctuations) and limited to `[0, 1]`.
pub fn trigger_efficiency(
    data: &crate::data::DataHistogram,
    fit_counts: &[f64],
    fit_min: f64,
) -> Vec<f64> {
    let n_bins = data.n_bins();
    let first_fit_bin = data.edges[..n_bins].partition_point(|&e| e < fit_min);
    /* blocks of (sum of data, sum of fit, number of bins) */
    let mut blocks: Vec<(f64, f64, usize)> = Vec::new();
    for (&d, &m) in data.counts.iter().zip(fit_counts).take(first_fit_bin) {
        blocks.push((d, m, 1));
        while blocks.len() >= 2 {
            let (d1, m1, n1) = blocks[blocks.len() - 1];
            let (d0, m0, n0) = blocks[blocks.len() - 2];
            /* merge while the efficiency decreases (an empty fit merges) */
            if m1 > 0. && m0 > 0. && d0 / m0 <= d1 / m1 {
                break;
            }
            blocks.pop();
            *blocks.last_mut().expect("block") = (d0 + d1, m0 + m1, n0 + n1);
        }
    }
    let mut efficiency = Vec::with_capacity(n_bins);
    for (d, m, n) in blocks {
        let e = if m > 0. { (d / m).clamp(0., 1.) } else { 1. };
        efficiency.extend(std::iter::repeat_n(e, n));
    }
    efficiency.resize(n_bins, 1.);
    efficiency
}

/// Reads the fit file, defines the classes, writes the output file and the
/// tables, and returns the classes.
pub fn run(config: &CentralityConfig) -> Result<CentralityResult> {
    crate::with_threads(config.n_threads, || {
        config.validate()?;
        let fit = read_fit(&config.fit_file)?;
        if !fit.valid {
            eprintln!(
                "Warning: the fit in {} is not valid",
                config.fit_file.display()
            );
        }
        let impact = config
            .impact_parameter
            .as_ref()
            .map(ImpactProfile::from_config)
            .transpose()?;
        std::fs::create_dir_all(&config.out_dir)?;
        let (result, objects) = compute(config, &fit, impact.as_ref())?;
        objects.write(&config.final_path())?;
        println!(
            "Centrality classes written to {}",
            config.final_path().display()
        );
        for &format in &config.table_formats {
            let path = config.table_path(format);
            std::fs::write(&path, crate::table::table(&result, format))?;
            println!("Table written to {}", path.display());
        }
        if let Some(path) = config.plot_path() {
            crate::plot::write_plot(&path, &fit, &result, config.plot_title.as_deref())?;
            println!("Picture written to {}", path.display());
        }
        Ok(result)
    })
}

/// Objects of the output file.
#[derive(Default)]
pub struct FinalObjects {
    pub tree: Option<Tree>,
    /// The best fit of the fit file (`BestResult`).
    pub best_result: Option<Tree>,
    pub hists: Vec<Hist1D>,
    pub hists_2d: Vec<Hist2D>,
    pub graphs: Vec<Graph>,
}

impl FinalObjects {
    pub fn write(self, path: &std::path::Path) -> Result<()> {
        let mut writer = FileWriter::create(path);
        if let Some(tree) = self.tree {
            writer = writer.put(tree);
        }
        if let Some(tree) = self.best_result {
            writer = writer.put(tree);
        }
        for h in &self.hists {
            writer = writer.add(h);
        }
        for h in &self.hists_2d {
            writer = writer.add(h);
        }
        for g in &self.graphs {
            writer = writer.add(g);
        }
        writer.write(Compression::default())?;
        Ok(())
    }
}

/// Computes the classes and the output objects.
pub fn compute(
    config: &CentralityConfig,
    fit: &StoredFit,
    impact: Option<&ImpactProfile>,
) -> Result<(CentralityResult, FinalObjects)> {
    let model = &fit.model;
    let p = &fit.values;
    let quad = Quadrature::new(config.cb_points);
    let calc = Calc::new(model.clone(), p, &quad);
    let data = &fit.data;
    let edges = &data.edges;
    let percents = config.class_edges();

    /* class borders: exact values, or snapped to the nearest data bin edge in
     * centrality; ±∞ at 0% and 100% */
    let exact: Vec<f64> = percents
        .par_iter()
        .map(|&e| calc.border(e / 100.))
        .collect();
    let mut borders: Vec<f64> = match config.borders {
        BorderMode::Exact => exact.clone(),
        BorderMode::BinEdges => exact
            .iter()
            .zip(&percents)
            .map(|(&x, &e)| {
                if !x.is_finite() {
                    return x;
                }
                let i = edges.partition_point(|&v| v < x);
                let candidates = [i.saturating_sub(1), i.min(edges.len() - 1)];
                candidates
                    .iter()
                    .map(|&j| edges[j])
                    .min_by(|a, b| {
                        let da = (calc.centrality(*a) - e / 100.).abs();
                        let db = (calc.centrality(*b) - e / 100.).abs();
                        da.total_cmp(&db)
                    })
                    .expect("two candidates")
            })
            .collect(),
    };
    /* expected recorded events: single collisions and pile-up */
    let (single, pile_up) = model.bin_probability_parts(p, &quad, edges, fit.bin_model);
    let single_counts: Vec<f64> = single.iter().map(|v| v * fit.norm).collect();
    let pile_up_counts: Option<Vec<f64>> =
        pile_up.map(|d| d.iter().map(|v| v * fit.norm).collect());

    /* the classes are defined for single collisions; the tail where pile-up
     * dominates (from the first bin above the knee where it exceeds the single
     * collisions) is not assigned to any class */
    let n_knee = model.n_knee(p);
    let pile_up_tail = pile_up_counts.as_ref().and_then(|d| {
        (0..data.n_bins())
            .find(|&i| data.center(i) > n_knee && d[i] > single_counts[i])
            .map(|i| edges[i])
    });
    if let Some(tail) = pile_up_tail {
        println!(
            "Pile-up dominates above {tail}; these events ({:.4}% of the fit) are not classified",
            100. * (single_counts
                .iter()
                .zip(pile_up_counts.as_ref().expect("pile-up"))
                .zip(edges)
                .filter(|(_, e)| **e >= tail)
                .map(|((s, d), _)| s + d)
                .sum::<f64>())
                / fit.norm
        );
        for b in &mut borders {
            *b = b.min(tail);
        }
    }

    let model_counts: Vec<f64> = match &pile_up_counts {
        Some(d) => single_counts.iter().zip(d).map(|(s, d)| s + d).collect(),
        None => single_counts.clone(),
    };
    /* the Bayes results describe the recorded events: each bin weighted with
     * the trigger efficiency (1 in and above the fit range) */
    let efficiency = trigger_efficiency(data, &model_counts, fit.fit_range.0);
    let n_classes = percents.len() - 1;
    let selections: Vec<Selection> = (0..n_classes)
        .map(|k| Selection::new(edges, &efficiency, borders[k + 1], borders[k]))
        .collect();

    let display = |x: f64| {
        if x == f64::INFINITY {
            edges[edges.len() - 1]
        } else if x == f64::NEG_INFINITY {
            edges[0]
        } else {
            x
        }
    };

    let mut classes: Vec<CentralityClass> = (0..n_classes)
        .into_par_iter()
        .map(|k| {
            let (x_hi, x_lo) = (borders[k], borders[k + 1]);
            let c_hi = calc.centrality(x_hi);
            let c_lo = calc.centrality(x_lo);
            let (total, (cb_mean, cb_rms), b) = calc.class_moments(&selections[k], impact);
            let cb = Stats {
                mean: cb_mean,
                rms: cb_rms,
                min: c_hi,
                max: c_lo,
            };
            let b = b.map(|(mean, rms)| {
                let ip = impact.expect("impact profile");
                Stats {
                    mean,
                    rms,
                    min: ip.b_of_cb(c_hi),
                    max: ip.b_of_cb(c_lo),
                }
            });
            if !(total > 0.) {
                eprintln!(
                    "Warning: class {} is empty (borders {} and {} coincide)",
                    class_label(percents[k], percents[k + 1]),
                    display(x_lo),
                    display(x_hi)
                );
            }
            CentralityClass {
                number: k,
                min_percent: percents[k],
                max_percent: percents[k + 1],
                min_border: display(x_lo),
                max_border: display(x_hi),
                fit_min_percent: 100. * c_hi,
                fit_max_percent: 100. * c_lo,
                data_percent: 100. * data_in_class(fit, x_lo, x_hi, config.borders) / fit.norm,
                efficiency: {
                    let fit_in_class: f64 = (0..data.n_bins())
                        .filter(|&i| bin_in_class(fit, i, x_lo, x_hi, config.borders))
                        .map(|i| single_counts[i] + pile_up_counts.as_ref().map_or(0., |d| d[i]))
                        .sum();
                    data_in_class(fit, x_lo, x_hi, config.borders) / fit_in_class
                },
                pile_up_percent: pile_up_counts.as_ref().map_or(0., |d| {
                    let (mut sum_d, mut sum_all) = (0., 0.);
                    for i in (0..data.n_bins())
                        .filter(|&i| bin_in_class(fit, i, x_lo, x_hi, config.borders))
                    {
                        sum_d += d[i];
                        sum_all += d[i] + single_counts[i];
                    }
                    if sum_all > 0. {
                        100. * sum_d / sum_all
                    } else {
                        0.
                    }
                }),
                cb,
                b,
                errors: None,
            }
        })
        .collect();

    let mut n_samples = 0;
    if config.n_error_samples > 0 {
        let (errors, used) = propagate_errors(config, fit, impact, &quad, &percents, &selections)?;
        n_samples = used;
        for (c, e) in classes.iter_mut().zip(errors) {
            c.errors = Some(e);
        }
    }

    let result = CentralityResult {
        classes,
        source: config.fit_file.clone(),
        impact: impact.map(ImpactProfile::description),
        b_range: impact.map(|ip| (ip.b_max(), config.b_max.unwrap_or(ip.b_max()))),
        n_knee,
        sigma_knee: model.sigma_knee(p),
        c_knee: calc.centrality(n_knee),
        n_error_samples: n_samples,
        pile_up_probability: model.pile_up_probability(p),
        pile_up_tail,
        efficiency: data.total / fit.norm,
        normalization: fit.normalization,
        model_counts: model_counts.clone(),
    };
    let bins = ClassBins {
        borders: &borders,
        selections: &selections,
        model_counts: &model_counts,
        efficiency: &efficiency,
    };
    let objects = final_objects(config, fit, impact, &calc, &result, &bins)?;
    Ok((result, objects))
}

/// Entries of the data histogram in `[x_lo, x_hi)`: whole bins between the
/// borders (bin centers for exact borders), plus under/overflow at the ends.
fn data_in_class(fit: &StoredFit, x_lo: f64, x_hi: f64, mode: BorderMode) -> f64 {
    let data = &fit.data;
    let mut sum: f64 = (0..data.n_bins())
        .filter(|&i| bin_in_class(fit, i, x_lo, x_hi, mode))
        .map(|i| data.counts[i])
        .sum();
    let contents = &data.hist.contents;
    if x_hi == f64::INFINITY {
        sum += contents[contents.len() - 1];
    }
    if x_lo == f64::NEG_INFINITY {
        sum += contents[0];
    }
    sum
}

/// Whether data bin `i` belongs to `[x_lo, x_hi)`: the whole bin for borders
/// on bin edges, its center for exact borders.
fn bin_in_class(fit: &StoredFit, i: usize, x_lo: f64, x_hi: f64, mode: BorderMode) -> bool {
    let data = &fit.data;
    match mode {
        BorderMode::BinEdges => data.edges[i] >= x_lo && data.edges[i + 1] <= x_hi,
        BorderMode::Exact => (x_lo..x_hi).contains(&data.center(i)),
    }
}

/// Standard deviations of the class results over parameter sets drawn from
/// the fit covariance (the borders of the classes are kept fixed, except for
/// the error of the exact borders themselves).
fn propagate_errors(
    config: &CentralityConfig,
    fit: &StoredFit,
    impact: Option<&ImpactProfile>,
    quad: &Quadrature,
    percents: &[f64],
    selections: &[Selection],
) -> Result<(Vec<ClassErrors>, usize)> {
    let free: Vec<usize> = (0..fit.values.len()).filter(|&i| !fit.fixed[i]).collect();
    let n = fit.values.len();
    let m = free.len();
    let sub: Vec<f64> = free
        .iter()
        .flat_map(|&i| free.iter().map(move |&j| fit.covariance[i * n + j]))
        .collect();
    let l = cholesky(&sub, m).ok_or_else(|| {
        Error::Input("the fit covariance is not positive definite; cannot propagate errors".into())
    })?;
    let seed = config.seed.unwrap_or(0);
    let n_classes = percents.len() - 1;
    println!(
        "Propagating the fit uncertainties with {} parameter sets",
        config.n_error_samples
    );

    /* per sample: exact borders, then (cb mean, cb rms, b mean, b rms) per class */
    type Sample = Option<(Vec<f64>, Vec<[f64; 4]>)>;
    let samples: Vec<Sample> = (0..config.n_error_samples)
        .into_par_iter()
        .map(|s| {
            let mut rng = SmallRng::seed_from_u64(splitmix64(seed ^ splitmix64(s as u64 + 1)));
            let z: Vec<f64> = (0..m).map(|_| StandardNormal.sample(&mut rng)).collect();
            let mut p = fit.values.clone();
            for (a, &i) in free.iter().enumerate() {
                p[i] += (0..=a).map(|b| l[a * m + b] * z[b]).sum::<f64>();
            }
            if !fit.model.is_valid(&p) {
                return None;
            }
            let calc = Calc::new(fit.model.clone(), &p, quad);
            let exact: Vec<f64> = percents.iter().map(|&e| calc.border(e / 100.)).collect();
            let per_class = (0..n_classes)
                .map(|k| {
                    let (_, (cm, cr), b) = calc.class_moments(&selections[k], impact);
                    let (bm, br) = b.unwrap_or((f64::NAN, f64::NAN));
                    [cm, cr, bm, br]
                })
                .collect();
            Some((exact, per_class))
        })
        .collect();
    let valid: Vec<_> = samples.into_iter().flatten().collect();
    if valid.len() < 2 {
        return Err(Error::Input(
            "too few valid parameter sets for the error propagation".into(),
        ));
    }
    if valid.len() < config.n_error_samples {
        eprintln!(
            "Warning: {} of {} parameter sets were outside the valid range and skipped",
            config.n_error_samples - valid.len(),
            config.n_error_samples
        );
    }
    let std = |values: Vec<f64>| -> f64 {
        let v: Vec<f64> = values.into_iter().filter(|x| x.is_finite()).collect();
        if v.len() < 2 {
            return f64::NAN;
        }
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
    };
    let errors = (0..n_classes)
        .map(|k| ClassErrors {
            min_border: std(valid.iter().map(|(e, _)| e[k + 1]).collect()),
            max_border: std(valid.iter().map(|(e, _)| e[k]).collect()),
            cb_mean: std(valid.iter().map(|(_, c)| c[k][0]).collect()),
            cb_rms: std(valid.iter().map(|(_, c)| c[k][1]).collect()),
            b_mean: std(valid.iter().map(|(_, c)| c[k][2]).collect()),
            b_rms: std(valid.iter().map(|(_, c)| c[k][3]).collect()),
        })
        .collect();
    Ok((errors, valid.len()))
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

pub fn class_label(min_percent: f64, max_percent: f64) -> String {
    format!("{}%-{}%", percent(min_percent), percent(max_percent))
}

/// A percent without trailing zeros.
pub fn percent(p: f64) -> String {
    let s = format!("{p:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Whether all borders are integers (e.g. multiplicities).
pub fn integer_borders(classes: &[CentralityClass]) -> bool {
    classes.iter().all(|c| {
        [c.min_border, c.max_border]
            .iter()
            .all(|v| v.is_finite() && v.fract() == 0. && v.abs() < 2e9)
    })
}

/// The tree `Result`, the class histograms and graphs.
/// The classes in terms of the data bins.
struct ClassBins<'a> {
    /// Class borders, from the most central class on (±∞ at the ends).
    borders: &'a [f64],
    selections: &'a [Selection],
    /// Expected recorded events in each data bin.
    model_counts: &'a [f64],
    /// Trigger efficiency of each data bin.
    efficiency: &'a [f64],
}

fn final_objects(
    config: &CentralityConfig,
    fit: &StoredFit,
    impact: Option<&ImpactProfile>,
    calc: &Calc,
    result: &CentralityResult,
    bins: &ClassBins,
) -> Result<FinalObjects> {
    let ClassBins {
        borders,
        selections,
        model_counts,
        efficiency,
    } = *bins;
    let classes = &result.classes;
    let data = &fit.data;
    let mut objects = FinalObjects::default();

    /* tree */
    let col = |get: &dyn Fn(&CentralityClass) -> f64| classes.iter().map(get).collect::<Vec<f64>>();
    let mut branches = vec![
        Branch::i32("Ncc", classes.iter().map(|c| c.number as i32).collect()),
        Branch::f32(
            "MinPercent",
            classes.iter().map(|c| c.min_percent as f32).collect(),
        ),
        Branch::f32(
            "MaxPercent",
            classes.iter().map(|c| c.max_percent as f32).collect(),
        ),
    ];
    if integer_borders(classes) {
        branches.push(Branch::i32(
            "MinBorder",
            classes.iter().map(|c| c.min_border as i32).collect(),
        ));
        branches.push(Branch::i32(
            "MaxBorder",
            classes.iter().map(|c| c.max_border as i32).collect(),
        ));
    } else {
        branches.push(Branch::f64("MinBorder", col(&|c| c.min_border)));
        branches.push(Branch::f64("MaxBorder", col(&|c| c.max_border)));
    }
    branches.extend([
        Branch::f64("FitMinPercent", col(&|c| c.fit_min_percent)),
        Branch::f64("FitMaxPercent", col(&|c| c.fit_max_percent)),
        Branch::f64("DataPercent", col(&|c| c.data_percent)),
        Branch::f64("Efficiency", col(&|c| c.efficiency)),
        Branch::f64("PileUpPercent", col(&|c| c.pile_up_percent)),
        Branch::f64("CbAverage", col(&|c| c.cb.mean)),
        Branch::f64("CbWidth", col(&|c| c.cb.rms)),
        Branch::f64("CbMin", col(&|c| c.cb.min)),
        Branch::f64("CbMax", col(&|c| c.cb.max)),
    ]);
    if impact.is_some() {
        let b = |f: fn(&Stats) -> f64| col(&|c| c.b.as_ref().map_or(f64::NAN, f));
        branches.extend([
            Branch::f64("BAverage", b(|s| s.mean)),
            Branch::f64("BWidth", b(|s| s.rms)),
            Branch::f64("BMin", b(|s| s.min)),
            Branch::f64("BMax", b(|s| s.max)),
        ]);
    }
    if classes.iter().all(|c| c.errors.is_some()) {
        let e = |f: fn(&ClassErrors) -> f64| col(&|c| c.errors.as_ref().map_or(f64::NAN, f));
        branches.extend([
            Branch::f64("MinBorderErr", e(|e| e.min_border)),
            Branch::f64("MaxBorderErr", e(|e| e.max_border)),
            Branch::f64("CbAverageErr", e(|e| e.cb_mean)),
            Branch::f64("CbWidthErr", e(|e| e.cb_rms)),
        ]);
        if impact.is_some() {
            branches.extend([
                Branch::f64("BAverageErr", e(|e| e.b_mean)),
                Branch::f64("BWidthErr", e(|e| e.b_rms)),
            ]);
        }
    }
    objects.tree = Some(Tree::new("Result", branches));
    objects.best_result = Some(best_result_tree(
        &fit.names,
        &fit.values,
        &fit.errors,
        fit.statistic,
        fit.ndf,
    ));

    /* averages versus centrality */
    let mut percent_edges: Vec<f64> = classes.iter().map(|c| c.min_percent).collect();
    percent_edges.push(classes[classes.len() - 1].max_percent);
    let contiguous = classes
        .windows(2)
        .all(|w| w[0].max_percent == w[1].min_percent);
    let middle: Vec<f64> = classes
        .iter()
        .map(|c| 0.5 * (c.min_percent + c.max_percent))
        .collect();
    if contiguous {
        objects.hists.push(histogram_from_edges(
            "Cb_average_VS_Centrality",
            "<c_{b}> (errors: RMS);centrality, %;<c_{b}>",
            &percent_edges,
            &col(&|c| c.cb.mean),
            Some(&col(&|c| c.cb.rms)),
        ));
    }
    if impact.is_some() {
        let mean = col(&|c| c.b.map_or(f64::NAN, |s| s.mean));
        let rms = col(&|c| c.b.map_or(f64::NAN, |s| s.rms));
        if contiguous {
            objects.hists.push(histogram_from_edges(
                "B_average_VS_Centrality",
                "<b> (errors: RMS);centrality, %;<b>, fm",
                &percent_edges,
                &mean,
                Some(&rms),
            ));
        }
        /* read by printFinal.C of the C++ GammaFit */
        objects.graphs.push(graph(
            "Fit_B_Mean",
            "B vs Centrality;Centrality, %;<b>, fm",
            middle,
            mean,
            Some(rms),
        )?);
    }

    /* distributions of c_b and b in each class */
    let cb_edges: Vec<f64> = (0..=config.cb_bins)
        .map(|i| i as f64 / config.cb_bins as f64)
        .collect();
    let b_max = config
        .b_max
        .or(impact.map(ImpactProfile::b_max))
        .unwrap_or(1.);
    if let (Some(ip), Some(user)) = (impact, config.b_max) {
        let beyond = 1. - ip.cb_of_b(user);
        if beyond > 1e-4 {
            eprintln!(
                "Warning: b_max = {user} fm is below the largest impact parameter of P(b), \
                 {:.3} fm: the histograms of b miss {:.3}% of the inelastic events \
                 (the means and widths of b are not affected)",
                ip.b_max(),
                100. * beyond
            );
        }
    }
    let b_edges: Vec<f64> = (0..=config.b_bins)
        .map(|i| b_max * i as f64 / config.b_bins as f64)
        .collect();
    /* the impact parameter distribution of all inelastic events */
    if let Some(ip) = impact {
        let density: Vec<f64> = b_edges
            .windows(2)
            .map(|w| (ip.cb_of_b(w[1]) - ip.cb_of_b(w[0])) / (w[1] - w[0]))
            .collect();
        objects.hists.push(uniform_histogram(
            "B_Distribution",
            &format!(
                "P(b) of inelastic events: {};b, fm;P(b), 1/fm",
                ip.description()
            ),
            0.,
            b_max,
            &density,
            None,
        ));
    }
    let per_class: Vec<(Hist1D, Option<Hist1D>)> = classes
        .par_iter()
        .map(|c| {
            let sel = &selections[c.number];
            /* recorded fraction of all events in the class */
            let fraction: f64 = calc.class_weights(sel).iter().sum();
            let density = |c0: f64, c1: f64, width: f64| {
                if fraction > 0. {
                    calc.class_probability_in(sel, c0, c1) / fraction / width
                } else {
                    0.
                }
            };
            let cb_density: Vec<f64> = cb_edges
                .windows(2)
                .map(|w| density(w[0], w[1], w[1] - w[0]))
                .collect();
            let cb_hist = uniform_histogram(
                &format!("Cb_VS_CentralityClass {}", c.label()),
                &format!("P(c_{{b}}) in {};c_{{b}};P(c_{{b}})", c.label()),
                0.,
                1.,
                &cb_density,
                None,
            );
            let b_hist = impact.map(|ip| {
                let b_density: Vec<f64> = b_edges
                    .windows(2)
                    .map(|w| density(ip.cb_of_b(w[0]), ip.cb_of_b(w[1]), w[1] - w[0]))
                    .collect();
                uniform_histogram(
                    &format!("B_VS_CentralityClass {}", c.label()),
                    &format!("P(b) in {};b, fm;P(b), 1/fm", c.label()),
                    0.,
                    b_max,
                    &b_density,
                    None,
                )
            });
            (cb_hist, b_hist)
        })
        .collect();
    for (cb_hist, b_hist) in per_class {
        objects.hists.push(cb_hist);
        objects.hists.extend(b_hist);
    }

    /* data and fit in each class, and the class of each bin */
    let in_class =
        |i: usize, k: usize| bin_in_class(fit, i, borders[k + 1], borders[k], config.borders);
    let mut class_of_bin = vec![0.; data.n_bins()];
    for c in classes {
        let select = |v: &[f64]| -> Vec<f64> {
            (0..data.n_bins())
                .map(|i| if in_class(i, c.number) { v[i] } else { 0. })
                .collect()
        };
        objects.hists.push(histogram_like(
            &data.hist,
            &format!("CentralityClass {}", c.label()),
            &format!("data in {}", c.label()),
            &select(&data.counts),
            Some(&select(&data.errors)),
        ));
        objects.hists.push(histogram_like(
            &data.hist,
            &format!("CentralityClass_Fit {}", c.label()),
            &format!("fit in {}", c.label()),
            &select(model_counts),
            None,
        ));
        for (i, v) in class_of_bin.iter_mut().enumerate() {
            if in_class(i, c.number) {
                *v = c.max_percent;
            }
        }
    }
    /* trigger efficiency versus c_b and b: ε(c_b) = Σ_i ε_i P(n ∈ bin i | c_b)
     * of single collisions, averaged over the events in each bin */
    let all = Selection::new(&data.edges, efficiency, f64::NEG_INFINITY, f64::INFINITY);
    let cb_efficiency: Vec<f64> = cb_edges
        .par_windows(2)
        .map(|w| calc.mean_probability_in(&all, w[0], w[1]))
        .collect();
    objects.hists.push(uniform_histogram(
        "Trigger_Efficiency_VS_Cb",
        "trigger efficiency of the events in each bin of c_{b};c_{b};efficiency",
        0.,
        1.,
        &cb_efficiency,
        None,
    ));
    if let Some(ip) = impact {
        let b_efficiency: Vec<f64> = b_edges
            .par_windows(2)
            .map(|w| calc.mean_probability_in(&all, ip.cb_of_b(w[0]), ip.cb_of_b(w[1])))
            .collect();
        objects.hists.push(uniform_histogram(
            "Trigger_Efficiency_VS_B",
            "trigger efficiency of the events in each bin of b;b, fm;efficiency",
            0.,
            b_max,
            &b_efficiency,
            None,
        ));
    }
    objects.hists.push(histogram_like(
        &data.hist,
        "Trigger_Efficiency",
        "trigger efficiency used for the classes: data / fit below the fit range, made non-decreasing;n;efficiency",
        efficiency,
        None,
    ));
    objects.hists.push(histogram_like(
        &data.hist,
        "Centrality_vs_Multiplicity",
        "upper percent of the class of each bin;n;centrality, %",
        &class_of_bin,
        None,
    ));
    objects.hists.push(fit_histogram(&data.hist, model_counts));
    objects.hists.push(data.hist.clone());

    /* joint distribution of the observable and b */
    if let Some(ip) = impact {
        let cb_of_b: Vec<f64> = b_edges.iter().map(|&b| ip.cb_of_b(b)).collect();
        /* single collisions only */
        let joint = joint_counts(
            &calc.model,
            calc.p,
            fit.norm * (1. - result.pile_up_probability),
            &data.edges,
            fit.bin_model,
            &cb_of_b,
        );
        let mut h = histogram_2d(
            &data.hist,
            "B_VS_Multiplicity",
            "expected single-collision events;n;b, fm",
            config.b_bins,
            0.,
            b_max,
        );
        for (iy, row) in joint.iter().enumerate() {
            for (ix, v) in row.iter().enumerate() {
                set_2d(&mut h, ix + 1, iy + 1, *v);
            }
        }
        h.entries = joint.iter().flatten().sum();
        objects.hists_2d.push(h);
    }
    Ok(objects)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_merges_intervals_of_equal_efficiency() {
        let edges = [0., 1., 2., 3., 4.];
        let eff = [0.2, 0.5, 1., 1.];
        let s = Selection::new(&edges, &eff, 0.5, 3.5);
        assert_eq!(s.bounds, vec![0.5, 1., 2., 3.5]);
        assert_eq!(s.weights, vec![0.2, 0.5, 1.]);
        /* open ends: below the histogram with the first bin, above with 1 */
        let s = Selection::new(&edges, &eff, f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(s.bounds, vec![f64::NEG_INFINITY, 1., 2., f64::INFINITY]);
        assert_eq!(s.weights, vec![0.2, 0.5, 1.]);
        /* fully efficient: one interval, the probability of the class */
        let s = Selection::new(&edges, &[1.; 4], 1., 3.);
        assert_eq!((s.bounds.len(), s.weights.len()), (2, 1));
        let k = NodeKernel::new(crate::model::Kernel::Gamma, 2., 1.);
        let p = s.probability(&k);
        assert!((p - (k.survival(1.) - k.survival(3.))).abs() < 1e-15);
    }

    #[test]
    fn efficiency_curve_is_monotonic_and_bounded() {
        let mut hist = oxiroot::prelude::Hist::reg(8, 0., 8.).double();
        let counts = [10., 30., 25., 70., 95., 103., 98., 100.];
        for (i, c) in counts.iter().enumerate() {
            hist.fill_weight(i as f64 + 0.5, *c);
        }
        let data = crate::data::DataHistogram::new(hist).unwrap();
        let fit = [100.; 8];
        /* fit range from n = 5: efficiency 1 there */
        let e = trigger_efficiency(&data, &fit, 5.);
        assert_eq!(&e[5..], &[1., 1., 1.]);
        /* 0.3 and 0.25 are pooled to 0.275 */
        assert!((e[0] - 0.1).abs() < 1e-12);
        assert!((e[1] - 0.275).abs() < 1e-12 && (e[2] - 0.275).abs() < 1e-12);
        assert!((e[3] - 0.7).abs() < 1e-12 && (e[4] - 0.95).abs() < 1e-12);
        assert!(e.windows(2).all(|w| w[0] <= w[1]));
    }
}
