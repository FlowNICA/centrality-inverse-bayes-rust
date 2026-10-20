//! Configuration of the two steps, read from the `fit` and `centrality`
//! sections of a RON file (see `config.ron`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::config_file::{expand_home, read_section};
use crate::error::{Error, Result};
use crate::model::{BinModel, Kernel, MeanModel, Model, VarianceModel};
use crate::profile::{Axis, HistogramSpec};

/// Statistic minimized by the fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum FitMethod {
    /// Poisson likelihood ratio `2 Σ [m - d + d ln(d/m)]` (Baker-Cousins),
    /// for histograms of raw counts. Unbiased also in the sparse tail.
    Likelihood,
    /// Neyman χ² with the statistical errors of the data; empty data bins are
    /// skipped. Works for weighted or scaled histograms, but bins with few
    /// counts bias the fit (e.g. `θ` low by a few percent).
    Chi2,
}

impl FitMethod {
    pub fn statistic_name(self) -> &'static str {
        match self {
            FitMethod::Chi2 => "chi2/ndf",
            FitMethod::Likelihood => "-2ln(L/Lsat)/ndf",
        }
    }
}

/// Normalization of the model to the data: the number `N` of all inelastic
/// events, to which the centrality percents refer. The trigger efficiency
/// is the number of recorded events over `N`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub enum Normalization {
    /// `N` is the number of entries of the data histogram (including under-
    /// and overflow), as in Das et al.: `P(n)` is normalized to all events, so
    /// the fraction of events outside the fit range constrains the fit too.
    /// The histogram must contain all inelastic events (efficiency 1): with
    /// a trigger inefficiency, the fit distorts `n̄(c_b)` to absorb the lost
    /// events, without any sign in its quality, and all percents are wrong.
    Data,
    /// `N` = entries / `ε`, for a trigger efficiency `ε` known from elsewhere
    /// (e.g. a simulation of the trigger). Choose the fit range where the
    /// trigger is fully efficient.
    Efficiency(f64),
    /// `N` is fitted together with the shape, from the bins in the fit range
    /// only (profiled analytically), and the model extrapolates below it. Use
    /// it for a trigger inefficiency below the fit range when `ε` is not
    /// known. `N` is then only as good as the extrapolation of `n̄(c_b)`: on
    /// toys the `Glauber` mean gives `N` within 1% if the data follow it,
    /// while `ExpPoly` can be off by 5-30% with an equally good fit. Check the
    /// stability of `N` against `fit_min`.
    Free,
}

impl Normalization {
    /// The number of all events for a data histogram with `entries`, if it
    /// does not depend on the fit.
    pub fn fixed_total(self, entries: f64) -> Option<f64> {
        match self {
            Normalization::Data => Some(entries),
            Normalization::Efficiency(e) => Some(entries / e),
            Normalization::Free => None,
        }
    }
}

impl std::fmt::Display for Normalization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Normalization::Data => f.write_str("Data"),
            Normalization::Efficiency(e) => write!(f, "Efficiency({e:?})"),
            Normalization::Free => f.write_str("Free"),
        }
    }
}

impl std::str::FromStr for Normalization {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        ron::from_str(s.trim())
            .map_err(|e| Error::Input(format!("unknown normalization '{s}': {e}")))
    }
}

/// Initial value, step, limits and fixing of a fit parameter.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamConfig {
    pub value: Option<f64>,
    pub step: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    #[serde(default)]
    pub fixed: bool,
}

/// Configuration of `bin/fit`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitConfig {
    /// ROOT file with the data histogram.
    pub data_file: PathBuf,
    /// Name of the data histogram (TH1 of the centrality observable).
    pub data_hist: String,
    #[serde(default = "default_out_dir")]
    pub out_dir: PathBuf,
    /// Fit range in the units of the data: bins whose centers lie in
    /// `[fit_min, fit_max]`; default: the whole histogram.
    #[serde(default)]
    pub fit_min: Option<f64>,
    #[serde(default)]
    pub fit_max: Option<f64>,
    #[serde(default = "default_fit_method")]
    pub fit_method: FitMethod,
    #[serde(default = "default_normalization")]
    pub normalization: Normalization,
    #[serde(default = "default_kernel")]
    pub kernel: Kernel,
    #[serde(default = "default_mean")]
    pub mean: MeanModel,
    #[serde(default = "default_variance")]
    pub variance: VarianceModel,
    /// Fit pile-up: with probability `p` (parameter `p`), a recorded event is
    /// the sum of two independent collisions.
    #[serde(default)]
    pub pile_up: bool,
    #[serde(default = "default_bin_model")]
    pub bin_model: BinModel,
    /// Number of quadrature points for the integral over `c_b`.
    #[serde(default = "default_fit_cb_points")]
    pub cb_points: usize,
    /// Initial values, limits and fixed parameters, by parameter name.
    #[serde(default)]
    pub params: BTreeMap<String, ParamConfig>,
    /// Minuit strategy (0, 1 or 2).
    #[serde(default = "default_strategy")]
    pub strategy: u32,
    /// Maximum number of function calls of Migrad.
    #[serde(default)]
    pub max_calls: Option<usize>,
    /// Migrad tolerance (EDM goal is 0.002 · tolerance · up).
    #[serde(default)]
    pub tolerance: Option<f64>,
    /// Number of worker threads; default: all cores.
    #[serde(default)]
    pub n_threads: Option<usize>,
}

/// Impact parameter distribution `P(b)` of inelastic events, used to turn
/// `c_b` into `b`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ImpactParameter {
    /// `P(b) = 2πb/σ_inel` for `b < √(σ_inel/π)`: `c_b = πb²/σ_inel`.
    /// `sigma_inel` in fm² (1 b = 100 fm²).
    Geometric { sigma_inel: f64 },
    /// `P(b) ∝ 2πb / (1 + exp((b - b0)/width))` (Frankland et al.), given
    /// either `b0` or the total cross section `sigma_inel` (fm²), from which
    /// `b0` is computed.
    Fermi {
        #[serde(default)]
        sigma_inel: Option<f64>,
        #[serde(default)]
        b0: Option<f64>,
        width: f64,
    },
    /// Histogram of `b` (fm) of inelastic events, e.g. from a model.
    Histogram { file: PathBuf, hist: String },
    /// Projection on `b` (fm) of a 2-D histogram of `n` and `b` of all
    /// inelastic events, e.g. the one of the `Histogram` mean or variance of
    /// the fit; `x_axis` and `y_axis` say which axis holds `"n"` and which
    /// `"b"`.
    Histogram2D {
        file: PathBuf,
        hist: String,
        x_axis: Axis,
        y_axis: Axis,
    },
    /// Branch with `b` (fm) of inelastic events, e.g. the `B` branch of an
    /// MC-Glauber tree.
    Tree {
        file: PathBuf,
        tree: String,
        branch: String,
    },
    /// Optical Glauber model from the density profiles of the nuclei
    /// `nucleus_a` and `nucleus_b` (names of `data/nuclei.ron`, the nuclei of
    /// oxy-glauber, or of `nuclei_file`): `P(b) ∝ 2πb [1 - (1 - σ_NN
    /// T_AB(b))^{AB}]`. `σ_NN` is `sigma_nn` (mb) or computed from `sqrt_snn`
    /// (GeV, at least 10 GeV); with `sigma_np`, `sigma_nn` is the pp
    /// cross section and the np one is `sigma_np`.
    Glauber {
        nucleus_a: String,
        nucleus_b: String,
        #[serde(default)]
        sigma_nn: Option<f64>,
        #[serde(default)]
        sigma_np: Option<f64>,
        #[serde(default)]
        sqrt_snn: Option<f64>,
        #[serde(default)]
        nuclei_file: Option<PathBuf>,
    },
}

/// How the borders of the centrality classes are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum BorderMode {
    /// The data bin edge nearest to the exact border, so that each class is a
    /// set of whole bins (e.g. integer multiplicities). The class fractions
    /// then differ slightly from the nominal percents; the actual ones are
    /// written too.
    BinEdges,
    /// The exact value of the observable with the nominal fraction above it.
    Exact,
}

/// Table files written in addition to the table on stdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum TableFormat {
    Tex,
    Csv,
    Cpp,
}

impl TableFormat {
    pub fn extension(self) -> &'static str {
        match self {
            TableFormat::Tex => "tex",
            TableFormat::Csv => "csv",
            TableFormat::Cpp => "C",
        }
    }
}

/// Configuration of `bin/define-centrality`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CentralityConfig {
    /// File written by `fit`.
    pub fit_file: PathBuf,
    #[serde(default = "default_out_dir")]
    pub out_dir: PathBuf,
    #[serde(default = "default_final_file")]
    pub final_file: String,
    /// Class edges in percent, e.g. `[0, 5, 10, 20, ..., 100]`; default:
    /// `n_classes` classes of equal width.
    #[serde(default)]
    pub classes: Option<Vec<f64>>,
    #[serde(default = "default_n_classes")]
    pub n_classes: usize,
    #[serde(default = "default_border_mode")]
    pub borders: BorderMode,
    /// Impact parameter distribution; without it only `c_b` is reconstructed.
    #[serde(default)]
    pub impact_parameter: Option<ImpactParameter>,
    /// Bins of the `b` histograms; their range is `[0, b_max]`.
    #[serde(default = "default_b_bins")]
    pub b_bins: usize,
    /// Upper edge of the `b` histograms (fm); default: where `c_b` reaches 1.
    #[serde(default)]
    pub b_max: Option<f64>,
    /// Bins of the `c_b` histograms on `[0, 1]`.
    #[serde(default = "default_cb_bins")]
    pub cb_bins: usize,
    /// Quadrature points for the integrals over `c_b`.
    #[serde(default = "default_centrality_cb_points")]
    pub cb_points: usize,
    /// Number of parameter sets drawn from the fit covariance to propagate the
    /// fit uncertainties to the results; 0 disables it.
    #[serde(default)]
    pub n_error_samples: usize,
    /// Seed of the parameter sampling; default: fixed seed 0.
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "default_table_formats")]
    pub table_formats: Vec<TableFormat>,
    #[serde(default = "default_table_name")]
    pub table_name: String,
    /// PNG picture of the data, the fit, the ratio and the class borders,
    /// written to `out_dir`; `None` disables it.
    #[serde(default = "default_mult_plot")]
    pub mult_plot: Option<MultPlot>,
    #[serde(default)]
    pub n_threads: Option<usize>,
}

/// The picture of the distribution of the observable, the fit and the
/// classes written by `define-centrality`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultPlot {
    /// File name in `out_dir`.
    #[serde(default = "default_plot_file")]
    pub file: String,
    /// Beginning of the header, e.g. the experiment and the colliding system.
    #[serde(default)]
    pub title: Option<String>,
    /// Title of the x axis; default: that of the data histogram, or its name.
    #[serde(default)]
    pub x_title: Option<String>,
    /// Title of the y axis of the distribution.
    #[serde(default = "default_y_title")]
    pub y_title: String,
    /// Title of the y axis of the ratio data / fit.
    #[serde(default = "default_ratio_title")]
    pub ratio_title: String,
    /// Draw the class borders (vertical lines) and the class names.
    #[serde(default = "default_true")]
    pub class_borders: bool,
    /// Range of the x axis; default: the histogram up to the last bin with
    /// data or fit.
    #[serde(default)]
    pub x_range: Option<(f64, f64)>,
    /// Range of the (logarithmic) y axis; default: from 0.4, with room for
    /// the header above the data.
    #[serde(default)]
    pub y_range: Option<(f64, f64)>,
    /// Range of the y axis of the ratio; default: 0.45 to 1.55, lower if the
    /// ratio is.
    #[serde(default)]
    pub ratio_range: Option<(f64, f64)>,
}

impl Default for MultPlot {
    fn default() -> Self {
        MultPlot {
            file: default_plot_file(),
            title: None,
            x_title: None,
            y_title: default_y_title(),
            ratio_title: default_ratio_title(),
            class_borders: true,
            x_range: None,
            y_range: None,
            ratio_range: None,
        }
    }
}

impl MultPlot {
    pub fn validate(&self) -> Result<()> {
        let err = |m: String| Err(Error::Config(format!("mult_plot: {m}")));
        if self.file.trim().is_empty() {
            return err("file must not be empty".into());
        }
        for (name, range) in [
            ("x_range", self.x_range),
            ("y_range", self.y_range),
            ("ratio_range", self.ratio_range),
        ] {
            if let Some((lo, hi)) = range
                && !(lo < hi && lo.is_finite() && hi.is_finite())
            {
                return err(format!("{name}: ({lo}, {hi}) must be increasing"));
            }
        }
        if let Some((lo, _)) = self.y_range
            && !(lo > 0.)
        {
            return err(format!(
                "y_range must start above 0 (logarithmic axis), got {lo}"
            ));
        }
        Ok(())
    }
}

fn default_out_dir() -> PathBuf {
    PathBuf::from(".")
}
fn default_fit_method() -> FitMethod {
    FitMethod::Likelihood
}
fn default_normalization() -> Normalization {
    Normalization::Data
}
fn default_kernel() -> Kernel {
    Kernel::Gamma
}
fn default_mean() -> MeanModel {
    MeanModel::ExpPoly(3)
}
fn default_variance() -> VarianceModel {
    VarianceModel::Proportional
}
fn default_bin_model() -> BinModel {
    BinModel::Integral
}
fn default_fit_cb_points() -> usize {
    1024
}
fn default_strategy() -> u32 {
    1
}
fn default_final_file() -> String {
    "FINAL.root".into()
}
fn default_n_classes() -> usize {
    10
}
fn default_border_mode() -> BorderMode {
    BorderMode::BinEdges
}
fn default_b_bins() -> usize {
    200
}
fn default_cb_bins() -> usize {
    500
}
fn default_centrality_cb_points() -> usize {
    4096
}
fn default_table_formats() -> Vec<TableFormat> {
    vec![TableFormat::Csv]
}
fn default_table_name() -> String {
    "centrality_table".into()
}
fn default_plot_file() -> String {
    "centrality_result.png".into()
}
fn default_y_title() -> String {
    "dN".into()
}
fn default_ratio_title() -> String {
    "data/fit".into()
}
fn default_true() -> bool {
    true
}
fn default_mult_plot() -> Option<MultPlot> {
    Some(MultPlot::default())
}

impl FitConfig {
    /// Configuration with all defaults for the given data histogram.
    pub fn new(data_file: impl Into<PathBuf>, data_hist: impl Into<String>) -> Self {
        FitConfig {
            data_file: data_file.into(),
            data_hist: data_hist.into(),
            out_dir: default_out_dir(),
            fit_min: None,
            fit_max: None,
            fit_method: default_fit_method(),
            normalization: default_normalization(),
            kernel: default_kernel(),
            mean: default_mean(),
            variance: default_variance(),
            pile_up: false,
            bin_model: default_bin_model(),
            cb_points: default_fit_cb_points(),
            params: BTreeMap::new(),
            strategy: default_strategy(),
            max_calls: None,
            tolerance: None,
            n_threads: None,
        }
    }

    /// Reads and validates the `fit` section of a RON file.
    pub fn from_ron_file(path: &Path) -> Result<Self> {
        let mut config: FitConfig = read_section(path, "fit")?;
        config.data_file = expand_home(&config.data_file);
        config.out_dir = expand_home(&config.out_dir);
        if let MeanModel::Glauber {
            nuclei_file: Some(file),
            ..
        } = &mut config.mean
        {
            *file = expand_home(file);
        }
        if let MeanModel::Histogram { file, .. }
        | MeanModel::Tree { file, .. }
        | MeanModel::Idrisov { file, .. } = &mut config.mean
        {
            *file = expand_home(file);
        }
        if let VarianceModel::Histogram { file, .. }
        | VarianceModel::Tree { file, .. }
        | VarianceModel::Idrisov { file, .. } = &mut config.variance
        {
            *file = expand_home(file);
        }
        config.validate()?;
        Ok(config)
    }

    /// The model; for the `Glauber` mean this computes the optical Glauber
    /// table, for the `Histogram` mean and variance it reads the histograms
    /// (see [`Model::new`]).
    pub fn model(&self) -> Result<Model> {
        Model::new(
            self.kernel,
            self.mean.clone(),
            self.variance.clone(),
            self.pile_up,
        )
    }

    pub fn validate(&self) -> Result<()> {
        let err = |m: String| Err(Error::Config(m));
        if let MeanModel::ExpPoly(n) = self.mean
            && !(1..=10).contains(&n)
        {
            return err(format!("ExpPoly needs 1 to 10 coefficients, got {n}"));
        }
        if let Some(spec) = self.mean.glauber_spec() {
            spec.validate()
                .map_err(|e| Error::Config(format!("mean: {e}")))?;
        }
        if let Some(spec) = self.mean.profile_spec() {
            spec.validate()
                .map_err(|e| Error::Config(format!("mean: {e}")))?;
        }
        if let Some(spec) = self.variance.profile_spec() {
            spec.validate()
                .map_err(|e| Error::Config(format!("variance: {e}")))?;
        }
        if self.mean.function_spec().is_some() != self.variance.function_spec().is_some() {
            return err(format!(
                "the Idrisov mean and variance are used together (the variance depends on m1 \
                 of the mean), got mean {} and variance {}",
                self.mean, self.variance
            ));
        }
        if let (Some(lo), Some(hi)) = (self.fit_min, self.fit_max)
            && !(lo < hi)
        {
            return err(format!("fit_min ({lo}) must be below fit_max ({hi})"));
        }
        if self.cb_points < 8 {
            return err(format!(
                "cb_points must be at least 8, got {}",
                self.cb_points
            ));
        }
        if let Normalization::Efficiency(e) = self.normalization
            && !(e > 0. && e <= 1.)
        {
            return err(format!("the trigger efficiency must be in (0, 1], got {e}"));
        }
        if self.strategy > 2 {
            return err(format!("strategy must be 0, 1 or 2, got {}", self.strategy));
        }
        if self.n_threads == Some(0) {
            return err("n_threads must be positive".into());
        }
        let names = Model::param_names_for(&self.mean, &self.variance, self.pile_up);
        for (name, p) in &self.params {
            if !names.contains(name) {
                return err(format!(
                    "unknown parameter '{name}' for mean model {} and variance model {}, \
                     expected one of: {}",
                    self.mean,
                    self.variance,
                    names.join(", ")
                ));
            }
            if p.fixed && p.value.is_none() {
                return err(format!("fixed parameter '{name}' needs a value"));
            }
            if let (Some(lo), Some(hi)) = (p.min, p.max)
                && !(lo < hi)
            {
                return err(format!(
                    "parameter '{name}': min ({lo}) must be below max ({hi})"
                ));
            }
            if let (Some(v), Some(lo), Some(hi)) = (p.value, p.min, p.max)
                && !(lo..=hi).contains(&v)
            {
                return err(format!(
                    "parameter '{name}': value {v} outside [{lo}, {hi}]"
                ));
            }
            if p.step.is_some_and(|s| !(s > 0.)) {
                return err(format!("parameter '{name}': step must be positive"));
            }
        }
        Ok(())
    }
}

impl CentralityConfig {
    /// Configuration with all defaults for the given fit file.
    pub fn new(fit_file: impl Into<PathBuf>) -> Self {
        CentralityConfig {
            fit_file: fit_file.into(),
            out_dir: default_out_dir(),
            final_file: default_final_file(),
            classes: None,
            n_classes: default_n_classes(),
            borders: default_border_mode(),
            impact_parameter: None,
            b_bins: default_b_bins(),
            b_max: None,
            cb_bins: default_cb_bins(),
            cb_points: default_centrality_cb_points(),
            n_error_samples: 0,
            seed: None,
            table_formats: default_table_formats(),
            table_name: default_table_name(),
            mult_plot: default_mult_plot(),
            n_threads: None,
        }
    }

    /// Reads and validates the `centrality` section of a RON file.
    pub fn from_ron_file(path: &Path) -> Result<Self> {
        let mut config: CentralityConfig = read_section(path, "centrality")?;
        config.fit_file = expand_home(&config.fit_file);
        config.out_dir = expand_home(&config.out_dir);
        if let Some(ip) = &mut config.impact_parameter {
            ip.expand_home();
        }
        config.validate()?;
        Ok(config)
    }

    /// Class edges in percent, ascending from 0 to 100.
    pub fn class_edges(&self) -> Vec<f64> {
        match &self.classes {
            Some(edges) => edges.clone(),
            None => (0..=self.n_classes)
                .map(|i| 100. * i as f64 / self.n_classes as f64)
                .collect(),
        }
    }

    pub fn final_path(&self) -> PathBuf {
        self.out_dir.join(&self.final_file)
    }

    /// Path of the picture, if one is written.
    pub fn plot_path(&self) -> Option<PathBuf> {
        self.mult_plot.as_ref().map(|p| self.out_dir.join(&p.file))
    }

    pub fn table_path(&self, format: TableFormat) -> PathBuf {
        self.out_dir
            .join(format!("{}.{}", self.table_name, format.extension()))
    }

    pub fn validate(&self) -> Result<()> {
        let err = |m: String| Err(Error::Config(m));
        let edges = self.class_edges();
        if edges.len() < 2 {
            return err("at least one centrality class is needed".into());
        }
        if edges.windows(2).any(|w| !(w[0] < w[1])) {
            return err(format!("class edges must increase: {edges:?}"));
        }
        if !(edges[0] >= 0. && edges[edges.len() - 1] <= 100.) {
            return err(format!("class edges must lie within [0, 100]: {edges:?}"));
        }
        if self.b_bins == 0 || self.cb_bins == 0 {
            return err("b_bins and cb_bins must be positive".into());
        }
        if self.cb_points < 8 {
            return err(format!(
                "cb_points must be at least 8, got {}",
                self.cb_points
            ));
        }
        if self.b_max.is_some_and(|b| !(b > 0.)) {
            return err("b_max must be positive".into());
        }
        if let Some(plot) = &self.mult_plot {
            plot.validate()?;
        }
        if self.n_threads == Some(0) {
            return err("n_threads must be positive".into());
        }
        match &self.impact_parameter {
            Some(ip) => ip.validate(),
            None => Ok(()),
        }
    }
}

impl ImpactParameter {
    /// Expands a leading `~/` in the file names.
    pub fn expand_home(&mut self) {
        match self {
            ImpactParameter::Histogram { file, .. }
            | ImpactParameter::Histogram2D { file, .. }
            | ImpactParameter::Tree { file, .. } => {
                *file = expand_home(file);
            }
            ImpactParameter::Glauber {
                nuclei_file: Some(file),
                ..
            } => *file = expand_home(file),
            _ => {}
        }
    }

    /// The 2-D histogram of `Histogram2D`.
    pub fn histogram_spec(&self) -> Option<HistogramSpec> {
        match self {
            ImpactParameter::Histogram2D {
                file,
                hist,
                x_axis,
                y_axis,
            } => Some(HistogramSpec {
                file: file.clone(),
                hist: hist.clone(),
                x_axis: *x_axis,
                y_axis: *y_axis,
            }),
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let err = |m: String| Err(Error::Config(m));
        match self {
            ImpactParameter::Histogram2D { .. } => self
                .histogram_spec()
                .expect("Histogram2D")
                .validate()
                .map_err(|e| Error::Config(format!("impact_parameter: {e}"))),
            ImpactParameter::Geometric { sigma_inel } if !(*sigma_inel > 0.) => {
                err(format!("sigma_inel must be positive, got {sigma_inel}"))
            }
            ImpactParameter::Fermi {
                sigma_inel,
                b0,
                width,
            } => match (sigma_inel, b0) {
                _ if !(*width > 0.) => err(format!("Fermi width must be positive, got {width}")),
                (Some(_), Some(_)) | (None, None) => {
                    err("Fermi impact parameter needs exactly one of sigma_inel and b0".into())
                }
                (Some(s), None) if !(*s > 0.) => {
                    err(format!("sigma_inel must be positive, got {s}"))
                }
                (None, Some(b)) if !(*b > 0.) => err(format!("b0 must be positive, got {b}")),
                _ => Ok(()),
            },
            ImpactParameter::Glauber {
                nucleus_a,
                nucleus_b,
                sigma_nn,
                sigma_np,
                sqrt_snn,
                nuclei_file,
            } => crate::optical::GlauberSpec {
                nucleus_a: nucleus_a.clone(),
                nucleus_b: nucleus_b.clone(),
                sigma_nn: *sigma_nn,
                sigma_np: *sigma_np,
                sqrt_snn: *sqrt_snn,
                nuclei_file: nuclei_file.clone(),
            }
            .validate()
            .map_err(|e| Error::Config(format!("impact_parameter: {e}"))),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_file::parse_section;

    #[test]
    fn fit_defaults_and_params() {
        let text = r#"(
            fit: (
                data_file: "data.root",
                data_hist: "hRefMult",
                fit_min: 10,
                mean: ExpPoly(4),
                params: { "theta": (value: 1.0, min: 0.1), "a4": (value: 0, fixed: true) },
            ),
        )"#;
        let c: FitConfig = parse_section(text, "fit").unwrap();
        assert_eq!(c.fit_min, Some(10.));
        assert_eq!(c.kernel, Kernel::Gamma);
        assert_eq!(c.mean, MeanModel::ExpPoly(4));
        assert!(c.params["a4"].fixed);
        c.validate().unwrap();

        let mut bad = c.clone();
        bad.params.insert("a5".into(), ParamConfig::default());
        assert!(bad.validate().is_err());
        let mut bad = c.clone();
        bad.params.insert(
            "theta".into(),
            ParamConfig {
                fixed: true,
                ..Default::default()
            },
        );
        assert!(bad.validate().is_err());
    }

    #[test]
    fn histogram_models() {
        let dir = std::env::temp_dir().join(format!("cib-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.ron");
        let write = |fit: &str| {
            std::fs::write(
                &path,
                format!(r#"(fit: (data_file: "d.root", data_hist: "h", {fit}))"#),
            )
            .unwrap();
            FitConfig::from_ron_file(&path)
        };
        let c = write(
            r#"mean: Histogram(file: "~/m.root", hist: "h2", x_axis: "b", y_axis: "n"),
               variance: Histogram(file: "m.root", hist: "h2", x_axis: "b", y_axis: "n")"#,
        )
        .unwrap();
        assert!(!c.mean.histogram_spec().unwrap().file.starts_with("~"));
        assert_eq!(
            Model::param_names_for(&c.mean, &c.variance, false),
            ["variance_scale", "mean_scale"]
        );
        /* same axis twice, unknown axis, P(b) is not set separately */
        let bad = [
            r#"variance: Histogram(file: "m.root", hist: "h2", x_axis: "n", y_axis: "n")"#,
            r#"mean: Histogram(file: "m.root", hist: "h2", x_axis: "x", y_axis: "n")"#,
            r#"mean: Histogram(file: "m.root", hist: "h2", x_axis: "b", y_axis: "n",
                   impact_parameter: Geometric(sigma_inel: 685.0))"#,
        ];
        for text in bad {
            assert!(write(text).is_err(), "{text}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mult_plot_options() {
        let parse = |plot: &str| {
            let text = format!(r#"(centrality: (fit_file: "f.root", {plot}))"#);
            parse_section::<CentralityConfig>(&text, "centrality").map(|c| c.validate().map(|_| c))
        };
        /* default: a picture with the default options */
        let c = parse("").unwrap().unwrap();
        assert_eq!(c.mult_plot, Some(MultPlot::default()));
        assert_eq!(
            c.plot_path(),
            Some(PathBuf::from("./centrality_result.png"))
        );
        assert_eq!(parse("mult_plot: None").unwrap().unwrap().plot_path(), None);
        let c = parse(
            r#"mult_plot: (file: "p.png", title: "Toy", x_title: "N_{ch}", y_title: "counts",
                ratio_title: "ratio", class_borders: false, x_range: (0, 300),
                y_range: (0.5, 1e6), ratio_range: (0.8, 1.2))"#,
        )
        .unwrap()
        .unwrap();
        let p = c.mult_plot.unwrap();
        assert_eq!(p.file, "p.png");
        assert_eq!(p.x_title.as_deref(), Some("N_{ch}"));
        assert!(!p.class_borders);
        assert_eq!(p.x_range, Some((0., 300.)));
        assert_eq!(p.y_range, Some((0.5, 1e6)));
        /* invalid ranges, unknown fields, the former options */
        assert!(parse("mult_plot: (x_range: (300, 0))").unwrap().is_err());
        assert!(parse("mult_plot: (y_range: (0, 1e6))").unwrap().is_err());
        assert!(parse("mult_plot: (colour: 1)").is_err());
        assert!(parse(r#"plot_file: "p.png""#).is_err());
    }

    #[test]
    fn normalizations() {
        let parse = |n: &str| {
            let text =
                format!(r#"(fit: (data_file: "d.root", data_hist: "h", normalization: {n}))"#);
            parse_section::<FitConfig>(&text, "fit")
                .unwrap()
                .validate()
                .map(|_| ())
        };
        assert!(
            parse("Data").is_ok() && parse("Free").is_ok() && parse("Efficiency(0.92)").is_ok()
        );
        assert!(parse("Efficiency(0)").is_err() && parse("Efficiency(1.2)").is_err());
        for n in [
            Normalization::Data,
            Normalization::Free,
            Normalization::Efficiency(0.875),
        ] {
            assert_eq!(n.to_string().parse::<Normalization>().unwrap(), n);
        }
        assert_eq!(Normalization::Efficiency(0.5).fixed_total(100.), Some(200.));
        assert_eq!(Normalization::Free.fixed_total(100.), None);
    }

    #[test]
    fn glauber_mean_settings_are_checked_when_read() {
        let check = |glauber: &str| {
            let text = format!(
                r#"(fit: (data_file: "d.root", data_hist: "h", mean: Glauber({glauber})))"#
            );
            parse_section::<FitConfig>(&text, "fit").unwrap().validate()
        };
        for ok in [
            r#"nucleus_a: "Au", nucleus_b: "Au", sqrt_snn: 2.4"#,
            r#"nucleus_a: "Au", nucleus_b: "Au", sqrt_snn: 3.5"#,
            r#"nucleus_a: "Pb", nucleus_b: "Pb", sqrt_snn: 200"#,
            r#"nucleus_a: "Au", nucleus_b: "Au", sigma_nn: 20, sigma_np: 25"#,
        ] {
            check(ok).unwrap();
        }
        for (bad, message) in [
            (
                r#"nucleus_a: "Au", nucleus_b: "Au", sqrt_snn: 2.0"#,
                "2.0223",
            ),
            (r#"nucleus_a: "Au", nucleus_b: "Au""#, "exactly one"),
            (
                r#"nucleus_a: "Au", nucleus_b: "Au", sigma_nn: 30, sqrt_snn: 200"#,
                "exactly one",
            ),
            (
                r#"nucleus_a: "Au", nucleus_b: "Au", sqrt_snn: 2.4, sigma_np: 20"#,
                "sigma_np",
            ),
            (
                r#"nucleus_a: "Gold", nucleus_b: "Au", sqrt_snn: 2.4"#,
                "unknown nucleus",
            ),
            (
                r#"nucleus_a: "O", nucleus_b: "O", sqrt_snn: 200"#,
                "analytic density",
            ),
        ] {
            let e = check(bad).unwrap_err().to_string();
            assert!(e.contains("mean") && e.contains(message), "{bad}: {e}");
        }
    }

    #[test]
    fn centrality_impact_parameter_variants() {
        let parse = |ip: &str| {
            let text = format!(r#"(centrality: (fit_file: "f.root", impact_parameter: {ip}))"#);
            parse_section::<CentralityConfig>(&text, "centrality").map(|c| c.validate().map(|_| c))
        };
        let c = parse("Geometric(sigma_inel: 685.0)").unwrap().unwrap();
        assert_eq!(
            c.impact_parameter,
            Some(ImpactParameter::Geometric { sigma_inel: 685. })
        );
        assert_eq!(c.class_edges().len(), 11);
        assert!(
            parse("Fermi(sigma_inel: 685.0, width: 0.4)")
                .unwrap()
                .is_ok()
        );
        assert!(parse("Fermi(b0: 13.0, width: 0.4)").unwrap().is_ok());
        assert!(parse("Fermi(width: 0.4)").unwrap().is_err());
        assert!(
            parse(r#"Tree(file: "g.root", tree: "nt", branch: "B")"#)
                .unwrap()
                .is_ok()
        );
        assert!(parse("Geometric(sigma_inel: -1)").unwrap().is_err());
        let c = parse(r#"Histogram2D(file: "m.root", hist: "h2", x_axis: "b", y_axis: "n")"#)
            .unwrap()
            .unwrap();
        assert_eq!(
            c.impact_parameter.unwrap().histogram_spec().unwrap().x_axis,
            Axis::B
        );
        assert!(
            parse(r#"Histogram2D(file: "m.root", hist: "h2", x_axis: "b", y_axis: "b")"#)
                .unwrap()
                .is_err()
        );
        let c = parse(r#"Glauber(nucleus_a: "Pb", nucleus_b: "Pb", sqrt_snn: 5020)"#)
            .unwrap()
            .unwrap();
        assert!(matches!(
            c.impact_parameter,
            Some(ImpactParameter::Glauber { .. })
        ));
        assert!(
            parse(r#"Glauber(nucleus_a: "Au", nucleus_b: "Au", sigma_nn: 30, sigma_np: 35)"#)
                .unwrap()
                .is_ok()
        );
        /* exactly one of sigma_nn and sqrt_snn; sqrt_snn from 2.0223 GeV */
        assert!(
            parse(r#"Glauber(nucleus_a: "Au", nucleus_b: "Au")"#)
                .unwrap()
                .is_err()
        );
        let sqrt_snn = |e: f64| {
            parse(&format!(
                r#"Glauber(nucleus_a: "Au", nucleus_b: "Au", sqrt_snn: {e})"#
            ))
            .unwrap()
        };
        assert!(sqrt_snn(2.2).is_ok() && sqrt_snn(3.).is_ok() && sqrt_snn(200.).is_ok());
        assert!(sqrt_snn(5.).is_ok() && sqrt_snn(2.).is_err());
        assert!(
            parse(r#"Glauber(nucleus_a: "Au", nucleus_b: "Au", sqrt_snn: 2.4, sigma_np: 20)"#)
                .unwrap()
                .is_err()
        );
        assert!(
            parse(r#"Glauber(nucleus_a: "Au", nucleus_b: "Au", sigma_nn: 30, sqrt_snn: 200)"#)
                .unwrap()
                .is_err()
        );
        assert!(parse("Nope(x: 1)").is_err());
    }

    #[test]
    fn class_edges() {
        let mut c = CentralityConfig::new("f.root");
        c.classes = Some(vec![0., 5., 10., 30., 100.]);
        c.validate().unwrap();
        c.classes = Some(vec![0., 10., 5.]);
        assert!(c.validate().is_err());
    }
}

#[cfg(test)]
mod example_config {
    use super::*;
    use crate::config_file::parse_section;

    /// The documented example configuration parses and validates.
    #[test]
    fn example_config_ron_is_valid() {
        let text = include_str!("../config.ron");
        let fit: FitConfig = parse_section(text, "fit").unwrap();
        fit.validate().unwrap();
        let centrality: CentralityConfig = parse_section(text, "centrality").unwrap();
        centrality.validate().unwrap();
    }
}
