use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use centrality_inverse_bayes_rust::CentralityConfig;
use clap::Parser;

/// Define centrality classes from the result of `fit`: class borders of the
/// observable, and the distributions of the b-centrality c_b and of the impact
/// parameter b in each class (Bayes' theorem).
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// RON configuration file; the `centrality` section is used (see config.ron)
    config: PathBuf,
}

fn main() -> ExitCode {
    let start = Instant::now();
    let args = Args::parse();
    println!(
        "define-centrality: using configuration {}",
        args.config.display()
    );
    let result = CentralityConfig::from_ron_file(&args.config)
        .and_then(|config| centrality_inverse_bayes_rust::centrality::run(&config));
    match result {
        Ok(r) => {
            println!();
            print!("{}", centrality_inverse_bayes_rust::table::plain_table(&r));
            println!("Total time: {:.1?}", start.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
