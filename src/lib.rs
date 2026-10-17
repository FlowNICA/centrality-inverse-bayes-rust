//! Centrality determination with the inverse Bayes (Γ-fit) method.
//!
//! The measured distribution of a centrality observable `n` (multiplicity,
//! energy, ...) is fitted with
//!
//! ```text
//! P(n) = ∫₀¹ P(n|c_b) dc_b,
//! ```
//!
//! where `P(n|c_b)` is a Gamma (or Gaussian) fluctuation kernel whose mean
//! `n̄(c_b)` is a smooth function of the b-centrality `c_b` and whose variance
//! is `θ n̄(c_b)`. No Glauber model or participant nucleons are needed. Bayes'
//! theorem then gives the distribution of `c_b`, and with `P(b)` of the impact
//! parameter `b`, in any centrality class:
//!
//! ```text
//! P(c_b | n₁ ≤ n < n₂) = ∫_{n₁}^{n₂} P(n|c_b) dn / ∫_{n₁}^{n₂} P(n) dn.
//! ```
//!
//! References: R. Rogly, G. Giacalone, J.-Y. Ollitrault, PRC 98 (2018) 024902
//! (arXiv:1804.03031); S. J. Das, G. Giacalone, P.-A. Monard,
//! J.-Y. Ollitrault, PRC 97 (2018) 014905 (arXiv:1708.00081);
//! J. D. Frankland et al. (INDRA), PRC 104 (2021) 034609 (arXiv:2011.04496).

// `!(x > 0.)` is used on purpose: it also rejects NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod centrality;
pub mod config;
pub mod config_file;
pub mod data;
pub mod error;
pub mod fitter;
pub mod impact;
pub mod model;
pub mod numeric;
pub mod optical;
pub mod output;
pub mod plot;
pub mod profile;
pub mod table;

pub use centrality::CentralityResult;
pub use config::{
    BorderMode, CentralityConfig, FitConfig, FitMethod, ImpactParameter, Normalization,
    ParamConfig, TableFormat,
};
pub use error::{Error, Result};
pub use fitter::{FitProgress, FitResult, Fitter};
pub use model::{BinModel, Kernel, MeanModel, Model, Quadrature, VarianceModel};

/// Runs the whole fit: reads the data, fits, and writes the fit file into
/// `config.out_dir`. The progress is printed to stdout.
pub fn run(config: FitConfig) -> Result<FitResult> {
    run_with_progress(config, &FitProgress::print)
}

/// Same as [`run`], reporting the progress to `progress`.
pub fn run_with_progress(
    config: FitConfig,
    progress: &(dyn Fn(FitProgress) + Sync),
) -> Result<FitResult> {
    with_threads(config.n_threads, || {
        std::fs::create_dir_all(&config.out_dir)?;
        let fitter = Fitter::new(config)?;
        let result = fitter.fit_with_progress(progress)?;
        let path = fitter.config().out_dir.join(output::FIT_FILE_NAME);
        output::write_fit(&path, fitter.config(), fitter.data(), &result)?;
        println!("Fit results written to {}", path.display());
        Ok(result)
    })
}

/// Runs `f` in a thread pool with `n_threads` threads (all cores if `None`).
pub fn with_threads<T: Send>(
    n_threads: Option<usize>,
    f: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(n) = n_threads {
        builder = builder.num_threads(n);
    }
    builder.build()?.install(f)
}
