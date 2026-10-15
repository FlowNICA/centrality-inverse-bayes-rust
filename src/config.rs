//! Configuration of the two steps, read from the `fit` and `centrality`
//! sections of a RON file (see `config.ron`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::config_file::{expand_home, read_section};
use crate::error::{Error, Result};
use crate::model::{BinModel, Kernel, MeanModel, Model, VarianceModel};

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

/// Normalization of the model to the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Normalization {
    /// `N` is the number of entries of the data histogram (including under-
    /// and overflow), as in Das et al.: `P(n)` is normalized to all events, so
    /// the fraction of events outside the fit range constrains the fit too.
    /// The histogram must contain all (triggered) events.
    Data,
    /// The number of events `N` of the model is a free parameter, fitted
    /// together with the shape (it is profiled analytically). Use it when
    /// events are lost below the fit range (e.g. by trigger inefficiency): the
    /// model then extrapolates to all events, but the extrapolation is only
    /// weakly constrained, so `N` and the centrality scale can be biased.
    Free,
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
    #[serde(default)]
    pub n_threads: Option<usize>,
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
        config.validate()?;
        Ok(config)
    }

    pub fn model(&self) -> Model {
        Model {
            kernel: self.kernel,
            mean: self.mean,
            variance: self.variance,
            pile_up: self.pile_up,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let err = |m: String| Err(Error::Config(m));
        if let MeanModel::ExpPoly(n) = self.mean
            && !(1..=10).contains(&n)
        {
            return err(format!("ExpPoly needs 1 to 10 coefficients, got {n}"));
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
        if self.strategy > 2 {
            return err(format!("strategy must be 0, 1 or 2, got {}", self.strategy));
        }
        if self.n_threads == Some(0) {
            return err("n_threads must be positive".into());
        }
        let names = self.model().param_names();
        for (name, p) in &self.params {
            if !names.contains(name) {
                return err(format!(
                    "unknown parameter '{name}' for mean model {}, expected one of: {}",
                    self.mean,
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
            n_threads: None,
        }
    }

    /// Reads and validates the `centrality` section of a RON file.
    pub fn from_ron_file(path: &Path) -> Result<Self> {
        let mut config: CentralityConfig = read_section(path, "centrality")?;
        config.fit_file = expand_home(&config.fit_file);
        config.out_dir = expand_home(&config.out_dir);
        if let Some(ImpactParameter::Histogram { file, .. } | ImpactParameter::Tree { file, .. }) =
            &mut config.impact_parameter
        {
            *file = expand_home(file);
        }
        if let Some(ImpactParameter::Glauber {
            nuclei_file: Some(file),
            ..
        }) = &mut config.impact_parameter
        {
            *file = expand_home(file);
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
        if self.n_threads == Some(0) {
            return err("n_threads must be positive".into());
        }
        match &self.impact_parameter {
            Some(ImpactParameter::Geometric { sigma_inel }) if !(*sigma_inel > 0.) => {
                err(format!("sigma_inel must be positive, got {sigma_inel}"))
            }
            Some(ImpactParameter::Fermi {
                sigma_inel,
                b0,
                width,
            }) => match (sigma_inel, b0) {
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
            Some(ImpactParameter::Glauber {
                sigma_nn,
                sigma_np,
                sqrt_snn,
                ..
            }) => match (sigma_nn, sqrt_snn) {
                (Some(_), Some(_)) | (None, None) => err(
                    "Glauber impact parameter needs exactly one of sigma_nn and sqrt_snn".into(),
                ),
                (Some(s), None) if !(*s > 0.) => err(format!("sigma_nn must be positive, got {s}")),
                (None, Some(_)) if sigma_np.is_some() => {
                    err("sigma_np can only be used with sigma_nn".into())
                }
                (None, Some(e)) => crate::optical::nn_cross_sections(*e).map(|_| ()),
                _ if sigma_np.is_some_and(|s| !(s > 0.)) => err("sigma_np must be positive".into()),
                _ => Ok(()),
            },
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
        /* exactly one of sigma_nn and sqrt_snn; sqrt_snn from 2.02 to 3.38 GeV
         * (Bystricky tables) or from 10 GeV */
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
        assert!(sqrt_snn(2.).is_err() && sqrt_snn(5.).is_err());
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
