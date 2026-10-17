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
            centrality_inverse_bayes_rust::print_fit_summary(&r, method);
            println!("Total time: {:.1?}", start.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
