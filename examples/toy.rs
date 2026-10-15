//! Generates toy data for the inverse Bayes fit: impact parameters from a
//! geometric distribution, and a multiplicity from a Gamma kernel with a
//! known mean n̄(c_b), truncated to integers.
//!
//! ```sh
//! cargo run --release --example toy -- toy.root [events] [seed] [--large] [--pile-up=P]
//! ```
//!
//! With `--pile-up=P`, an event is with probability `P` the sum of two
//! independent collisions. The file contains the histogram `hMult` (unit
//! bins), the histogram `hB` of the impact parameter and the tree `events`
//! with the branches `B` (of the first collision), `Mult` and `PileUp`.

use std::f64::consts::PI;

use centrality_inverse_bayes_rust::{Kernel, MeanModel, Model, VarianceModel};
use oxiroot::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma};

/// θ, n_knee, a1, a2, a3 of the ExpPoly(3) mean.
const TRUTH: [f64; 5] = [1.1, 220., 4.0, -1.0, 1.5];
/// θ, n_knee, a1, a2, a3 for the `--large` toy (ALICE V0 amplitude like).
const TRUTH_LARGE: [f64; 5] = [26., 20406., 4.11, -1.9, 4.4];
/// Inelastic cross section, fm².
const SIGMA_INEL: f64 = 685.;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let large = std::env::args().any(|a| a == "--large");
    let pile_up: f64 = std::env::args()
        .find_map(|a| a.strip_prefix("--pile-up=").map(str::to_string))
        .map_or(Ok(0.), |s| s.parse())?;
    let truth = if large { TRUTH_LARGE } else { TRUTH };
    let mut args = std::env::args().skip(1).filter(|a| !a.starts_with("--"));
    let path = args.next().unwrap_or_else(|| "toy.root".into());
    let n_events: usize = args.next().map_or(Ok(1_000_000), |s| s.parse())?;
    let seed: u64 = args.next().map_or(Ok(1), |s| s.parse())?;

    let model = Model::new(
        Kernel::Gamma,
        MeanModel::ExpPoly(3),
        VarianceModel::Proportional,
        false,
    )?;
    let b_max = (SIGMA_INEL / PI).sqrt();
    let mut rng = SmallRng::seed_from_u64(seed);
    let (nbins, max) = if large { (2500, 25000.) } else { (500, 500.) };
    let mut h_mult = Hist::reg(nbins, 0., max)
        .name("hMult")
        .title("multiplicity;N_{ch};events")
        .double();
    let mut h_b = Hist::reg(200, 0., 20.)
        .name("hB")
        .title("impact parameter;b, fm;events")
        .double();
    let collision = |rng: &mut SmallRng| -> Result<(f64, f64), rand_distr::GammaError> {
        let b = b_max * rng.random::<f64>().sqrt();
        let mean = model.mean(&truth, PI * b * b / SIGMA_INEL);
        Ok((b, Gamma::new(mean / truth[0], truth[0])?.sample(rng)))
    };
    let (mut bs, mut mults, mut piled) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n_events {
        let (b, mut n) = collision(&mut rng)?;
        /* pile-up: a second, independent collision in the same event */
        let is_pile_up = rng.random::<f64>() < pile_up;
        if is_pile_up {
            n += collision(&mut rng)?.1;
        }
        let n = n.floor();
        h_mult.fill(n);
        h_b.fill(b);
        bs.push(b as f32);
        mults.push(n as i32);
        piled.push(i32::from(is_pile_up));
    }
    let tree = Tree::new(
        "events",
        vec![
            Branch::f32("B", bs),
            Branch::i32("Mult", mults),
            Branch::i32("PileUp", piled),
        ],
    );
    FileWriter::create(&path)
        .add(&h_mult)
        .add(&h_b)
        .put(tree)
        .write(Compression::default())?;
    println!("{n_events} events written to {path}");
    Ok(())
}
