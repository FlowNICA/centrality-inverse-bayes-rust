use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use centrality_inverse_bayes_rust::{FitConfig, FitProgress};
use clap::Parser;

/// Fit the distribution of a centrality observable with the inverse Bayes
/// model P(n) = ∫ P(n|c_b) dc_b (Gamma or Gauss fluctuation kernel).
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// RON configuration file; the `fit` section is used (see config.ron)
    config: PathBuf,
}

fn main() -> ExitCode {
    let start = Instant::now();
    let args = Args::parse();
    println!("fit: using configuration {}", args.config.display());
    let result = FitConfig::from_ron_file(&args.config).and_then(|config| {
        let method = config.fit_method;
        centrality_inverse_bayes_rust::run_with_progress(config, &FitProgress::print)
            .map(|r| (r, method))
    });
    match result {
        Ok((r, method)) => {
            println!();
            println!(
                "Results of the fit ({}), {}:",
                r.model.describe(),
                if r.valid {
                    "valid minimum"
                } else {
                    "INVALID minimum"
                }
            );
            for ((name, v), e) in r.names.iter().zip(&r.values).zip(&r.errors) {
                println!("  {name:>8} = {v:>14.6} +- {e:.6}");
            }
            println!(
                "  {} = {:.2} / {} = {:.4}",
                method.statistic_name(),
                r.statistic,
                r.ndf,
                r.reduced_statistic()
            );
            println!(
                "  N = {:.1} (data entries {:.0}), n_knee = {:.4}, sigma(n_knee) = {:.4}, c_knee = {:.4}%",
                r.norm,
                r.data_total,
                r.n_knee,
                r.sigma_knee,
                100. * r.c_knee
            );
            println!("Total time: {:.1?}", start.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
