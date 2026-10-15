//! End-to-end tests: toy data generated with known parameters and impact
//! parameters are fitted, the classes are defined, and the reconstructed
//! impact parameters are compared with the true ones.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use centrality_inverse_bayes_rust::output::{FIT_FILE_NAME, numbers, read_fit};
use centrality_inverse_bayes_rust::{
    BorderMode, CentralityConfig, FitConfig, FitMethod, ImpactParameter, Kernel, MeanModel, Model,
    Normalization, ParamConfig, TableFormat, VarianceModel, centrality,
};
use oxiroot::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma};

const SIGMA_INEL: f64 = 685.;
/// θ, n_knee, a1, a2, a3.
const TRUTH: [f64; 5] = [1.1, 220., 4.0, -1.0, 1.5];

fn exp_poly() -> Model {
    Model {
        kernel: Kernel::Gamma,
        mean: MeanModel::ExpPoly(3),
        variance: VarianceModel::Proportional,
        pile_up: false,
    }
}

fn test_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Toy events: (b, n) with geometric b and a Gamma kernel; `n` is floored
/// to an integer when `unit` is `None`, otherwise kept continuous and
/// multiplied by `unit`.
fn toy_events(n_events: usize, seed: u64, unit: Option<f64>) -> Vec<(f64, f64)> {
    toy_events_of(&exp_poly(), &TRUTH, n_events, seed, unit)
}

fn toy_events_of(
    model: &Model,
    truth: &[f64],
    n_events: usize,
    seed: u64,
    unit: Option<f64>,
) -> Vec<(f64, f64)> {
    let b_max = (SIGMA_INEL / PI).sqrt();
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n_events)
        .map(|_| {
            let b = b_max * rng.random::<f64>().sqrt();
            let mean = model.mean(truth, PI * b * b / SIGMA_INEL);
            let n = Gamma::new(mean / truth[0], truth[0])
                .unwrap()
                .sample(&mut rng);
            (b, unit.map_or(n.floor(), |u| n * u))
        })
        .collect()
}

/// Writes `hMult` (`nbins` on `[0, max)`) and the tree `events` (`B`, `Mult`).
fn write_toy(path: &Path, events: &[(f64, f64)], nbins: i32, max: f64) {
    let mut h = Hist::reg(nbins, 0., max).name("hMult").double();
    for &(_, n) in events {
        h.fill(n);
    }
    let tree = Tree::new(
        "events",
        vec![
            Branch::f64("B", events.iter().map(|e| e.0).collect()),
            Branch::f64("Mult", events.iter().map(|e| e.1).collect()),
        ],
    );
    FileWriter::create(path)
        .add(&h)
        .put(tree)
        .write(Compression::default())
        .unwrap();
}

/// True mean b of the events in `[lo, hi)` (open above for the first class).
fn true_mean_b(events: &[(f64, f64)], lo: f64, hi: f64, open: bool) -> f64 {
    let sel: Vec<f64> = events
        .iter()
        .filter(|(_, n)| *n >= lo && (open || *n < hi))
        .map(|(b, _)| *b)
        .collect();
    sel.iter().sum::<f64>() / sel.len() as f64
}

fn fit_config(dir: &Path, data: &Path) -> FitConfig {
    let mut c = FitConfig::new(data, "hMult");
    c.out_dir = dir.to_path_buf();
    c.fit_min = Some(10.);
    c
}

#[test]
fn recovers_parameters_and_impact_parameter() {
    let dir = test_dir("recovers");
    let events = toy_events(400_000, 11, None);
    let data = dir.join("toy.root");
    write_toy(&data, &events, 500, 500.);

    let result = centrality_inverse_bayes_rust::run(fit_config(&dir, &data)).unwrap();
    assert!(result.valid);
    assert_eq!(result.names, ["theta", "n_knee", "a1", "a2", "a3"]);
    for (((name, value), error), truth) in result
        .names
        .iter()
        .zip(&result.values)
        .zip(&result.errors)
        .zip(TRUTH)
    {
        let pull = (value - truth) / error;
        assert!(pull.abs() < 5., "{name}: {value} vs {truth} (pull {pull})");
    }
    assert!(result.reduced_statistic() < 1.5);
    assert_eq!(result.norm, events.len() as f64);

    /* the fit file reads back */
    let stored = read_fit(&dir.join(FIT_FILE_NAME)).unwrap();
    assert_eq!(stored.values, result.values);
    assert_eq!(stored.model, exp_poly());
    assert!(stored.covariance[0] > 0.);

    let mut config = CentralityConfig::new(dir.join(FIT_FILE_NAME));
    config.out_dir = dir.clone();
    config.classes = Some(vec![
        0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.,
    ]);
    config.impact_parameter = Some(ImpactParameter::Geometric {
        sigma_inel: SIGMA_INEL,
    });
    config.n_error_samples = 40;
    config.table_formats = vec![TableFormat::Csv, TableFormat::Tex, TableFormat::Cpp];
    let r = centrality::run(&config).unwrap();
    assert_eq!(r.classes.len(), 11);
    assert_eq!(r.n_error_samples, 40);
    for (k, c) in r.classes.iter().enumerate() {
        /* contiguous integer borders; the fit fraction is close to nominal */
        assert_eq!(c.min_border.fract(), 0.);
        if k > 0 {
            assert_eq!(c.max_border, r.classes[k - 1].min_border);
        }
        let width = c.max_percent - c.min_percent;
        assert!(((c.fit_max_percent - c.fit_min_percent) - width).abs() < 0.25 * width);
        assert!((c.data_percent - (c.fit_max_percent - c.fit_min_percent)).abs() < 0.3);
        /* reconstructed versus true mean impact parameter */
        let b = c.b.unwrap();
        let truth = true_mean_b(&events, c.min_border, c.max_border, k == 0);
        assert!(
            (b.mean - truth).abs() < 0.05,
            "class {}: {} vs {truth}",
            c.label(),
            b.mean
        );
        assert!(b.min <= b.mean && b.mean <= b.max + 2. * b.rms);
        let e = c.errors.unwrap();
        assert!(e.b_mean > 0. && e.b_mean < 0.1);
    }

    /* output file and tables */
    let fin = FileReader::open(config.final_path()).unwrap();
    let tree = TreeReader::open(&fin, "Result").unwrap();
    let b_avg = numbers(&tree, &fin, "BAverage").unwrap();
    assert_eq!(b_avg.len(), 11);
    assert_eq!(numbers(&tree, &fin, "BAverageErr").unwrap().len(), 11);
    let p_b = Hist1D::read_root(&fin, "B_VS_CentralityClass 0%-5%").unwrap();
    let width = (p_b.xaxis.xmax - p_b.xaxis.xmin) / p_b.xaxis.nbins as f64;
    let integral: f64 = p_b.contents.iter().sum::<f64>() * width;
    assert!((integral - 1.).abs() < 1e-3, "{integral}");
    Hist2D::read_root(&fin, "B_VS_Multiplicity").unwrap();
    let csv = std::fs::read_to_string(config.table_path(TableFormat::Csv)).unwrap();
    assert_eq!(csv.lines().count(), 13);
    let cpp = std::fs::read_to_string(config.table_path(TableFormat::Cpp)).unwrap();
    assert!(cpp.contains("Float_t GetCentMult(Int_t mult)"));
    assert!(
        std::fs::read_to_string(config.table_path(TableFormat::Tex))
            .unwrap()
            .contains("\\begin{tabular}")
    );
}

#[test]
fn model_variants_and_impact_inputs() {
    let dir = test_dir("variants");
    let events = toy_events(200_000, 5, None);
    let data = dir.join("toy.root");
    write_toy(&data, &events, 500, 500.);

    /* the INDRA parametrization, on data generated with it */
    let indra = Model {
        mean: MeanModel::Indra,
        ..exp_poly()
    };
    let indra_truth = [0.8, 150., 4., 1.2, 1.5];
    let indra_data = dir.join("indra.root");
    write_toy(
        &indra_data,
        &toy_events_of(&indra, &indra_truth, 200_000, 6, None),
        500,
        500.,
    );

    for (kernel, mean, method) in [
        (Kernel::Gauss, MeanModel::ExpPoly(3), FitMethod::Chi2),
        (Kernel::Gamma, MeanModel::Indra, FitMethod::Likelihood),
        (Kernel::Gamma, MeanModel::ExpPoly(4), FitMethod::Likelihood),
    ] {
        let (input, n_knee) = if mean == MeanModel::Indra {
            (&indra_data, indra_truth[1] + indra_truth[2])
        } else {
            (&data, TRUTH[1])
        };
        let mut c = fit_config(&dir, input);
        c.kernel = kernel;
        c.mean = mean;
        c.fit_method = method;
        if kernel == Kernel::Gauss {
            /* the Gaussian kernel needs a larger cut on peripheral events */
            c.fit_min = Some(40.);
        }
        let fitter = centrality_inverse_bayes_rust::Fitter::new(c).unwrap();
        let r = fitter.fit_with_progress(&|_| {}).unwrap();
        assert!(r.valid, "{kernel} {mean}");
        assert!(
            (r.n_knee - n_knee).abs() < 0.02 * n_knee,
            "{kernel} {mean}: n_knee = {}",
            r.n_knee
        );
        assert!(
            r.reduced_statistic() < 2.,
            "{kernel} {mean}: {}",
            r.reduced_statistic()
        );
    }

    /* the b of the events, read from the tree, give the same classes as the
     * geometric distribution they were drawn from */
    centrality_inverse_bayes_rust::run(fit_config(&dir, &data)).unwrap();
    let run = |ip: ImpactParameter, borders: BorderMode| {
        let mut config = CentralityConfig::new(dir.join(FIT_FILE_NAME));
        config.out_dir = dir.clone();
        config.impact_parameter = Some(ip);
        config.borders = borders;
        config.table_formats = vec![];
        centrality::run(&config).unwrap()
    };
    let geometric = run(
        ImpactParameter::Geometric {
            sigma_inel: SIGMA_INEL,
        },
        BorderMode::BinEdges,
    );
    let tree = run(
        ImpactParameter::Tree {
            file: data.clone(),
            tree: "events".into(),
            branch: "B".into(),
        },
        BorderMode::BinEdges,
    );
    for (g, t) in geometric.classes.iter().zip(&tree.classes) {
        assert!((g.b.unwrap().mean - t.b.unwrap().mean).abs() < 0.03);
    }
    /* exact borders reproduce the nominal percents */
    let exact = run(
        ImpactParameter::Fermi {
            sigma_inel: Some(SIGMA_INEL),
            b0: None,
            width: 0.01,
        },
        BorderMode::Exact,
    );
    for (c, g) in exact.classes.iter().zip(&geometric.classes) {
        assert!((c.fit_min_percent - c.min_percent).abs() < 1e-6);
        assert!((c.fit_max_percent - c.max_percent).abs() < 1e-6);
        assert!((c.b.unwrap().mean - g.b.unwrap().mean).abs() < 0.1);
    }
}

#[test]
fn continuous_observable_does_not_depend_on_units() {
    let dir = test_dir("units");
    let mut results = Vec::new();
    for (unit, name) in [(1e-3, "gev"), (1., "mev")] {
        let events = toy_events(200_000, 7, Some(unit));
        let data = dir.join(format!("{name}.root"));
        write_toy(&data, &events, 400, 400. * unit);
        let out = dir.join(name);
        let mut c = fit_config(&out, &data);
        c.fit_min = Some(10. * unit);
        let fit = centrality_inverse_bayes_rust::run(c).unwrap();
        let mut config = CentralityConfig::new(out.join(FIT_FILE_NAME));
        config.out_dir = out.clone();
        config.borders = BorderMode::Exact;
        config.table_formats = vec![];
        results.push((unit, fit, centrality::run(&config).unwrap()));
    }
    let (gev, mev) = (&results[0], &results[1]);
    assert!((gev.1.values[0] / gev.0 - mev.1.values[0]).abs() < 1e-3 * mev.1.values[0]);
    assert!((gev.1.n_knee / gev.0 - mev.1.n_knee).abs() < 1e-4 * mev.1.n_knee);
    for (a, b) in gev.2.classes.iter().zip(&mev.2.classes) {
        assert!((a.min_border / gev.0 - b.min_border).abs() < 1e-3 * b.min_border.max(1.));
        assert!((a.cb.mean - b.cb.mean).abs() < 1e-4);
    }
}

#[test]
fn fixed_parameters_and_free_normalization() {
    let dir = test_dir("fixed");
    let events = toy_events(100_000, 9, None);
    let data = dir.join("toy.root");
    write_toy(&data, &events, 500, 500.);

    let mut c = fit_config(&dir, &data);
    c.params.insert(
        "theta".into(),
        ParamConfig {
            value: Some(1.1),
            fixed: true,
            ..Default::default()
        },
    );
    c.params.insert(
        "n_knee".into(),
        ParamConfig {
            value: Some(210.),
            min: Some(150.),
            max: Some(300.),
            ..Default::default()
        },
    );
    c.normalization = Normalization::Free;
    let r = centrality_inverse_bayes_rust::run(c).unwrap();
    assert_eq!(r.values[0], 1.1);
    assert_eq!(r.errors[0], 0.);
    assert!(r.fixed[0] && !r.fixed[1]);
    assert!((r.n_knee - TRUTH[1]).abs() < 3.);
    /* the free normalization extrapolates below the fit range */
    assert!(
        (r.norm / events.len() as f64 - 1.).abs() < 0.2,
        "{}",
        r.norm
    );
    let stored = read_fit(&dir.join(FIT_FILE_NAME)).unwrap();
    assert_eq!(stored.covariance[0], 0.);
    assert!(stored.fixed[0]);
}

#[test]
fn ron_configuration_end_to_end() {
    let dir = test_dir("ron");
    let events = toy_events(50_000, 13, None);
    let data = dir.join("toy.root");
    write_toy(&data, &events, 500, 500.);
    let config = dir.join("config.ron");
    std::fs::write(
        &config,
        format!(
            r#"(
                fit: (
                    data_file: "{data}",
                    data_hist: "hMult",
                    out_dir: "{dir}",
                    fit_min: 10,
                    mean: ExpPoly(3),
                    cb_points: 512,
                    n_threads: 2,
                ),
                centrality: (
                    fit_file: "{dir}/bayes_fit.root",
                    out_dir: "{dir}",
                    n_classes: 5,
                    impact_parameter: Histogram(file: "{data}", hist: "hMultMissing"),
                ),
            )"#,
            data = data.display(),
            dir = dir.display()
        ),
    )
    .unwrap();
    let fit = FitConfig::from_ron_file(&config).unwrap();
    assert_eq!(fit.n_threads, Some(2));
    centrality_inverse_bayes_rust::run(fit).unwrap();
    /* a missing impact parameter histogram is reported */
    let c = CentralityConfig::from_ron_file(&config).unwrap();
    let err = centrality::run(&c).unwrap_err().to_string();
    assert!(err.contains("hMultMissing"), "{err}");
}

#[test]
fn pile_up_is_fitted_and_excluded_from_classes() {
    let dir = test_dir("pile_up");
    let probability = 0.02;
    let model = exp_poly();
    let b_max = (SIGMA_INEL / PI).sqrt();
    let mut rng = SmallRng::seed_from_u64(17);
    let mut collision = || {
        let b = b_max * rng.random::<f64>().sqrt();
        let mean = model.mean(&TRUTH, PI * b * b / SIGMA_INEL);
        let n = Gamma::new(mean / TRUTH[0], TRUTH[0])
            .unwrap()
            .sample(&mut rng);
        (b, n)
    };
    /* (b of the first collision, n, pile-up) */
    let mut events = Vec::new();
    let mut uniform = SmallRng::seed_from_u64(18);
    for _ in 0..400_000 {
        let (b, mut n) = collision();
        let piled = uniform.random::<f64>() < probability;
        if piled {
            n += collision().1;
        }
        events.push((b, n.floor(), piled));
    }
    let data = dir.join("toy.root");
    let pairs: Vec<(f64, f64)> = events.iter().map(|e| (e.0, e.1)).collect();
    write_toy(&data, &pairs, 600, 600.);

    /* without pile-up in the model, the tail is described by a wider kernel */
    let without = centrality_inverse_bayes_rust::Fitter::new(fit_config(&dir, &data))
        .unwrap()
        .fit_with_progress(&|_| {})
        .unwrap();
    let mut config = fit_config(&dir, &data);
    config.pile_up = true;
    let r = centrality_inverse_bayes_rust::run(config).unwrap();
    assert!(r.valid);
    assert_eq!(r.names.last().unwrap(), "p");
    let p = *r.values.last().unwrap();
    let p_err = *r.errors.last().unwrap();
    assert!((p - probability).abs() < 5. * p_err, "p = {p} +- {p_err}");
    assert!(
        (r.values[0] - TRUTH[0]).abs() < 5. * r.errors[0],
        "theta = {}",
        r.values[0]
    );
    assert!(r.reduced_statistic() < 1.5);
    assert!(without.statistic > 2. * r.statistic);
    assert!(without.values[0] > 1.5 * TRUTH[0]);
    let single: f64 = r.single_counts.iter().sum();
    let pile_up: f64 = r.pile_up_counts.as_ref().unwrap().iter().sum();
    assert!((pile_up / (single + pile_up) - p).abs() < 1e-3);

    let stored = read_fit(&dir.join(FIT_FILE_NAME)).unwrap();
    assert!(stored.model.pile_up);
    let file = FileReader::open(dir.join(FIT_FILE_NAME)).unwrap();
    Hist1D::read_root(&file, "fit_plp_histo").unwrap();
    Hist1D::read_root(&file, "fit_sng_histo").unwrap();

    let mut config = CentralityConfig::new(dir.join(FIT_FILE_NAME));
    config.out_dir = dir.clone();
    config.impact_parameter = Some(ImpactParameter::Geometric {
        sigma_inel: SIGMA_INEL,
    });
    config.table_formats = vec![];
    let c = centrality::run(&config).unwrap();
    let tail = c.pile_up_tail.expect("a pile-up dominated tail");
    assert!(tail > r.n_knee);
    let classes = &c.classes;
    assert_eq!(classes[0].max_border, tail);
    assert!(classes[0].pile_up_percent > classes[9].pile_up_percent);
    for (k, class) in classes.iter().enumerate() {
        /* single collisions in the class, against the true b */
        let sel: Vec<f64> = events
            .iter()
            .filter(|e| !e.2 && e.1 >= class.min_border && e.1 < class.max_border)
            .map(|e| e.0)
            .collect();
        let truth = sel.iter().sum::<f64>() / sel.len() as f64;
        let b = class.b.unwrap().mean;
        assert!((b - truth).abs() < 0.05, "class {k}: {b} vs {truth}");
        /* the pile-up fraction of the class */
        let all = events
            .iter()
            .filter(|e| e.1 >= class.min_border && e.1 < class.max_border)
            .count();
        let piled = all - sel.len();
        let true_percent = 100. * piled as f64 / all as f64;
        assert!(
            (class.pile_up_percent - true_percent).abs() < 0.25 * true_percent + 0.05,
            "class {k}: {} vs {true_percent}",
            class.pile_up_percent
        );
    }
    let fin = FileReader::open(config.final_path()).unwrap();
    let tree = TreeReader::open(&fin, "Result").unwrap();
    assert_eq!(numbers(&tree, &fin, "PileUpPercent").unwrap().len(), 10);
}

#[test]
fn impact_parameter_from_nuclear_density_profiles() {
    let dir = test_dir("optical");
    let events = toy_events(100_000, 23, None);
    let data = dir.join("toy.root");
    write_toy(&data, &events, 500, 500.);
    centrality_inverse_bayes_rust::run(fit_config(&dir, &data)).unwrap();

    let run = |ip: ImpactParameter| {
        let mut config = CentralityConfig::new(dir.join(FIT_FILE_NAME));
        config.out_dir = dir.clone();
        config.impact_parameter = Some(ip);
        config.table_formats = vec![];
        (centrality::run(&config).unwrap(), config.final_path())
    };
    let (glauber, final_path) = run(ImpactParameter::Glauber {
        nucleus_a: "Au".into(),
        nucleus_b: "Au".into(),
        sigma_nn: Some(30.),
        sigma_np: None,
        sqrt_snn: None,
        nuclei_file: None,
    });
    let description = glauber.impact.clone().unwrap();
    assert!(description.contains("optical Glauber"), "{description}");

    /* the same total cross section with a sharp geometric edge */
    let sigma_inel = centrality_inverse_bayes_rust::optical::OpticalGlauber::new(
        "Au",
        "Au",
        Some(30.),
        None,
        None,
        None,
    )
    .unwrap()
    .profile()
    .sigma_inel()
    .unwrap();
    assert!((600. ..760.).contains(&sigma_inel), "{sigma_inel}");
    let (geometric, _) = run(ImpactParameter::Geometric { sigma_inel });
    let (g, o) = (&geometric.classes, &glauber.classes);
    /* central collisions: P_inel = 1, the same b */
    assert!((g[0].b.unwrap().mean - o[0].b.unwrap().mean).abs() < 0.05);
    /* peripheral: the diffuse edge of the nuclei reaches larger b */
    assert!(o[9].b.unwrap().mean > g[9].b.unwrap().mean + 0.3);
    /* c_b does not depend on P(b) */
    for (a, b) in g.iter().zip(o) {
        assert_eq!(a.cb, b.cb);
    }

    let fin = FileReader::open(final_path).unwrap();
    let h = Hist1D::read_root(&fin, "B_Distribution").unwrap();
    let width = (h.xaxis.xmax - h.xaxis.xmin) / h.xaxis.nbins as f64;
    let integral: f64 = h.contents.iter().sum::<f64>() * width;
    assert!((integral - 1.).abs() < 1e-3, "{integral}");
}
