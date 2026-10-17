use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use centrality_inverse_bayes_rust::output::FIT_FILE_NAME;
use centrality_inverse_bayes_rust::{CentralityConfig, FitConfig, FitProgress, Result};
use clap::Parser;

/// Run `fit` and then `define-centrality` with one configuration file that
/// has both the `fit` and the `centrality` sections. The centrality step uses
/// the fit file just written.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// RON configuration file with `fit` and `centrality` sections (see config.ron)
    config: PathBuf,
}

/// Whether two paths name the same file, comparing them as absolute paths
/// without `.` components (the fit file need not exist yet).
fn same_path(a: &Path, b: &Path) -> bool {
    let normal = |p: &Path| -> PathBuf {
        p.components()
            .filter(|c| !matches!(c, Component::CurDir))
            .collect()
    };
    match (std::path::absolute(a), std::path::absolute(b)) {
        (Ok(a), Ok(b)) => normal(&a) == normal(&b),
        _ => normal(a) == normal(b),
    }
}

fn run(args: &Args, start: Instant) -> Result<()> {
    /* both sections are read and checked before the fit starts */
    let fit = FitConfig::from_ron_file(&args.config)?;
    let mut centrality = CentralityConfig::from_ron_file(&args.config)?;
    let fit_file = fit.out_dir.join(FIT_FILE_NAME);
    if !same_path(&centrality.fit_file, &fit_file) {
        eprintln!(
            "Warning: fit_file of the centrality section ({}) is not the file written by the \
             fit; using {}",
            centrality.fit_file.display(),
            fit_file.display()
        );
        centrality.fit_file = fit_file;
    }

    println!("== fit");
    let method = fit.fit_method;
    let result = centrality_inverse_bayes_rust::run_with_progress(fit, &FitProgress::print)?;
    centrality_inverse_bayes_rust::print_fit_summary(&result, method);
    println!("Fit time: {:.1?}", start.elapsed());

    println!();
    println!("== define-centrality");
    let classes = centrality_inverse_bayes_rust::centrality::run(&centrality)?;
    println!();
    print!(
        "{}",
        centrality_inverse_bayes_rust::table::plain_table(&classes)
    );
    Ok(())
}

fn main() -> ExitCode {
    let start = Instant::now();
    let args = Args::parse();
    println!("run-all: using configuration {}", args.config.display());
    match run(&args, start) {
        Ok(()) => {
            println!("Total time: {:.1?}", start.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
