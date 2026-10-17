//! ROOT output of the fit, and reading it back for the centrality step.

use std::path::Path;

use oxiroot::prelude::*;
use rayon::prelude::*;

use crate::config::{FitConfig, FitMethod, Normalization};
use crate::data::DataHistogram;
use crate::error::{Error, Result};
use crate::fitter::FitResult;
use crate::model::{BinModel, Model, Quadrature};

/// Name of the file written by the fit.
pub const FIT_FILE_NAME: &str = "bayes_fit.root";

/// Number of points of the graphs versus `c_b`.
const CB_GRAPH_POINTS: usize = 501;
/// Bins in `c_b` of `Cb_VS_Multiplicity`.
const CB_2D_BINS: usize = 100;

/// Name of the tree with the best fit, one entry with a branch per parameter.
pub const BEST_RESULT_TREE: &str = "BestResult";

/// The best fit as in centrality-glauber-rust: one entry, with a branch per
/// parameter (its value), `<name>_error` (its error, 0 if fixed), `chi2`
/// (the fit statistic per degree of freedom, `chi2/ndf` or
/// `-2ln(L/Lsat)/ndf`) and `ndf`.
pub fn best_result_tree(
    names: &[String],
    values: &[f64],
    errors: &[f64],
    statistic: f64,
    ndf: usize,
) -> Tree {
    let mut branches: Vec<Branch> = names
        .iter()
        .zip(values)
        .map(|(name, v)| Branch::f64(name.as_str(), vec![*v]))
        .collect();
    branches.extend(
        names
            .iter()
            .zip(errors)
            .map(|(name, e)| Branch::f64(format!("{name}_error"), vec![*e])),
    );
    branches.push(Branch::f64("chi2", vec![statistic / ndf.max(1) as f64]));
    branches.push(Branch::i32("ndf", vec![ndf as i32]));
    Tree::new(BEST_RESULT_TREE, branches)
}

/// Makes ROOT recompute the statistics from the bin contents
/// (`TH1::GetStats` does so if `fTsumw == 0` and `fEntries > 0`).
pub fn reset_stats(h: &mut Hist1D) {
    h.entries = h.contents.iter().sum();
    h.tsumw = 0.;
    h.tsumw2 = 0.;
    h.tsumwx = 0.;
    h.tsumwx2 = 0.;
}

/// A double-precision histogram with the binning of `like` and the given bin
/// contents (bins `1..=nbins`) and errors (default: zero).
pub fn histogram_like(
    like: &Hist1D,
    name: &str,
    title: &str,
    contents: &[f64],
    errors: Option<&[f64]>,
) -> Hist1D {
    let mut h = like
        .clone()
        .with_bin_content_type(BinContentType::F64)
        .named(name)
        .titled(title);
    let n = h.contents.len();
    h.contents = vec![0.; n];
    h.sumw2 = vec![0.; n];
    for (i, v) in contents.iter().enumerate() {
        h.contents[i + 1] = *v;
        if let Some(e) = errors {
            h.sumw2[i + 1] = e[i] * e[i];
        }
    }
    reset_stats(&mut h);
    h
}

/// `fit_histo`: the expected counts of the fit in the data bins, with the
/// bins below one count left empty, so that, like the data histogram of
/// counts, it starts at 1 on a logarithmic y axis instead of following the
/// far tails of the model down to tiny fractions of an event.
pub fn fit_histogram(like: &Hist1D, model_counts: &[f64]) -> Hist1D {
    let shown: Vec<f64> = model_counts
        .iter()
        .map(|&m| if m >= 1. { m } else { 0. })
        .collect();
    histogram_like(like, "fit_histo", "inverse Bayes fit", &shown, None)
}

/// A histogram with the given bin edges, contents and errors.
pub fn histogram_from_edges(
    name: &str,
    title: &str,
    edges: &[f64],
    contents: &[f64],
    errors: Option<&[f64]>,
) -> Hist1D {
    let like = Hist::var(edges).double();
    histogram_like(&like, name, title, contents, errors)
}

/// A uniform histogram on `[lo, hi]`.
pub fn uniform_histogram(
    name: &str,
    title: &str,
    lo: f64,
    hi: f64,
    contents: &[f64],
    errors: Option<&[f64]>,
) -> Hist1D {
    let like = Hist::reg(contents.len() as i32, lo, hi).double();
    histogram_like(&like, name, title, contents, errors)
}

/// A graph, with y errors if given.
pub fn graph(
    name: &str,
    title: &str,
    x: Vec<f64>,
    y: Vec<f64>,
    ey: Option<Vec<f64>>,
) -> Result<Graph> {
    let g = match ey {
        Some(ey) => {
            let ex = vec![0.; x.len()];
            Graph::with_errors(x, y, ex, ey)?
        }
        None => Graph::new(x, y)?,
    };
    Ok(g.named(name).titled(title))
}

/// An empty 2-D histogram with the x binning of the data and `ny` uniform
/// bins on `[ylo, yhi]`.
pub fn histogram_2d(
    data: &Hist1D,
    name: &str,
    title: &str,
    ny: usize,
    ylo: f64,
    yhi: f64,
) -> Hist2D {
    let x = &data.xaxis;
    let builder = if x.xbins.is_empty() {
        Hist::reg(x.nbins, x.xmin, x.xmax)
    } else {
        Hist::var(&x.xbins)
    };
    builder
        .reg(ny as i32, ylo, yhi)
        .name(name)
        .title(title)
        .double()
}

/// Sets bin `(ix, iy)` (1-based) of a 2-D histogram.
pub fn set_2d(h: &mut Hist2D, ix: usize, iy: usize, value: f64) {
    let nx = h.nx();
    h.contents[ix + (nx + 2) * iy] = value;
}

/// Expected counts `N ∫ P(x bin | c_b) dc_b` in bins of `c_b` (rows) and of
/// the data (columns); each row is integrated with its own quadrature rule.
pub fn joint_counts(
    model: &Model,
    p: &[f64],
    norm: f64,
    edges: &[f64],
    bin_model: BinModel,
    cb_edges: &[f64],
) -> Vec<Vec<f64>> {
    cb_edges
        .par_windows(2)
        .map(|w| {
            let quad = Quadrature::on_interval(w[0], w[1], 16);
            let mut row = vec![0.; edges.len() - 1];
            for (&cb, &wt) in quad.nodes.iter().zip(&quad.weights) {
                model
                    .kernel_at(p, cb)
                    .add_bin_probabilities(edges, wt * norm, bin_model, &mut row);
            }
            row
        })
        .collect()
}

/// Writes the fit result: the data, the fitted distribution and its ratio to
/// the data, `n̄(c_b)`, `σ(c_b)`, the centrality `c(n)`, the joint distribution
/// of `n` and `c_b`, and the trees `FitResult`, `Parameters` and `Covariance`.
pub fn write_fit(
    path: &Path,
    config: &FitConfig,
    data: &DataHistogram,
    result: &FitResult,
) -> Result<()> {
    let model = &result.model;
    let p = &result.values;
    let quad = Quadrature::new(config.cb_points);

    let fit = fit_histogram(&data.hist, &result.model_counts);
    let (ratio, ratio_err): (Vec<f64>, Vec<f64>) = data
        .counts
        .iter()
        .zip(&data.errors)
        .zip(&result.model_counts)
        .map(|((&d, &e), &m)| if m > 0. { (d / m, e / m) } else { (0., 0.) })
        .unzip();
    let ratio = histogram_like(
        &data.hist,
        "fit_ratio",
        "data / fit",
        &ratio,
        Some(&ratio_err),
    );

    let cb: Vec<f64> = (0..CB_GRAPH_POINTS)
        .map(|i| i as f64 / (CB_GRAPH_POINTS - 1) as f64)
        .collect();
    let mean: Vec<f64> = cb.iter().map(|&c| model.mean(p, c)).collect();
    let sigma: Vec<f64> = cb
        .iter()
        .zip(&mean)
        .map(|(&c, &m)| model.variance(p, c, m).sqrt())
        .collect();
    let mean_graph = graph(
        "Mean_VS_Cb",
        "mean of n vs c_b (errors: sigma);c_b;n",
        cb.clone(),
        mean,
        Some(sigma.clone()),
    )?;
    let sigma_graph = graph(
        "Sigma_VS_Cb",
        "sigma of n vs c_b;c_b;#sigma_{n}",
        cb,
        sigma,
        None,
    )?;

    let centrality: Vec<f64> = data
        .edges
        .par_iter()
        .map(|&x| 100. * model.centrality(p, &quad, x))
        .collect();
    let centrality_graph = graph(
        "Centrality_VS_Multiplicity",
        "centrality of the fit, fraction of single-collision events above n;n;centrality, %",
        data.edges.clone(),
        centrality,
        None,
    )?;

    let cb_edges: Vec<f64> = (0..=CB_2D_BINS)
        .map(|i| i as f64 / CB_2D_BINS as f64)
        .collect();
    /* single collisions only */
    let single_norm = result.norm * (1. - model.pile_up_probability(p));
    let joint = joint_counts(
        model,
        p,
        single_norm,
        &data.edges,
        config.bin_model,
        &cb_edges,
    );
    let mut cb_vs_mult = histogram_2d(
        &data.hist,
        "Cb_VS_Multiplicity",
        "expected single-collision events;n;c_{b}",
        CB_2D_BINS,
        0.,
        1.,
    );
    for (iy, row) in joint.iter().enumerate() {
        for (ix, v) in row.iter().enumerate() {
            set_2d(&mut cb_vs_mult, ix + 1, iy + 1, *v);
        }
    }
    cb_vs_mult.entries = joint.iter().flatten().sum();

    let s = |name: &str, v: String| Branch::strings(name, vec![v]);
    let f = |name: &str, v: f64| Branch::f64(name, vec![v]);
    let fit_result = Tree::new(
        "FitResult",
        vec![
            s("data_file", config.data_file.display().to_string()),
            s("data_hist", config.data_hist.clone()),
            s("kernel", model.kernel.to_string()),
            s("mean", model.mean.to_string()),
            s("variance", model.variance.to_string()),
            s("bin_model", config.bin_model.to_string()),
            s("fit_method", format!("{:?}", config.fit_method)),
            s("normalization", config.normalization.to_string()),
            Branch::i32("pile_up", vec![i32::from(model.pile_up)]),
            f("pile_up_probability", model.pile_up_probability(p)),
            Branch::i32("cb_points", vec![config.cb_points as i32]),
            f("statistic", result.statistic),
            Branch::i32("ndf", vec![result.ndf as i32]),
            f("chi2_ndf", result.reduced_statistic()),
            f("norm", result.norm),
            f("efficiency", result.efficiency),
            f(
                "efficiency_below_fit",
                result.efficiency_below_fit.unwrap_or(f64::NAN),
            ),
            f("data_total", result.data_total),
            f("fit_min", result.fit_range.0),
            f("fit_max", result.fit_range.1),
            f("n_knee", result.n_knee),
            f("sigma_knee", result.sigma_knee),
            f("c_knee", result.c_knee),
            Branch::i32("valid", vec![i32::from(result.valid)]),
            f("edm", result.edm),
            Branch::i32("n_calls", vec![result.n_calls as i32]),
        ],
    );
    let best_result = best_result_tree(
        &result.names,
        &result.values,
        &result.errors,
        result.statistic,
        result.ndf,
    );
    let parameters = Tree::new(
        "Parameters",
        vec![
            Branch::strings("name", result.names.clone()),
            Branch::f64("value", result.values.clone()),
            Branch::f64("error", result.errors.clone()),
            Branch::i32(
                "fixed",
                result.fixed.iter().map(|&f| i32::from(f)).collect(),
            ),
        ],
    );
    let n = result.values.len();
    let covariance = Tree::new(
        "Covariance",
        vec![
            Branch::i32("row", (0..n * n).map(|k| (k / n) as i32).collect()),
            Branch::i32("col", (0..n * n).map(|k| (k % n) as i32).collect()),
            Branch::f64("value", result.covariance.clone()),
        ],
    );

    let mut writer = FileWriter::create(path).add(&data.hist).add(&fit);
    if let Some(pile_up) = &result.pile_up_counts {
        let single = histogram_like(
            &data.hist,
            "fit_sng_histo",
            "single collisions",
            &result.single_counts,
            None,
        );
        let pile_up = histogram_like(&data.hist, "fit_plp_histo", "pile-up", pile_up, None);
        writer = writer.add(&single).add(&pile_up);
    }
    writer
        .add(&ratio)
        .add(&mean_graph)
        .add(&sigma_graph)
        .add(&centrality_graph)
        .add(&cb_vs_mult)
        .put(fit_result)
        .put(best_result)
        .put(parameters)
        .put(covariance)
        .write(Compression::default())?;
    Ok(())
}

/// A fit result read back from the fit file.
#[derive(Debug, Clone)]
pub struct StoredFit {
    pub model: Model,
    pub bin_model: BinModel,
    pub fit_method: FitMethod,
    pub normalization: Normalization,
    pub names: Vec<String>,
    pub values: Vec<f64>,
    pub errors: Vec<f64>,
    pub fixed: Vec<bool>,
    pub covariance: Vec<f64>,
    pub norm: f64,
    pub statistic: f64,
    pub ndf: usize,
    pub fit_range: (f64, f64),
    pub valid: bool,
    pub data: DataHistogram,
}

/// Reads a file written by [`write_fit`].
pub fn read_fit(path: &Path) -> Result<StoredFit> {
    let file = FileReader::open(path)
        .map_err(|e| Error::Input(format!("cannot open fit file {}: {e}", path.display())))?;
    let tree = |name: &str| {
        TreeReader::open(&file, name).map_err(|e| {
            Error::Input(format!(
                "{}: cannot read tree '{name}': {e}",
                path.display()
            ))
        })
    };
    let fit_result = tree("FitResult")?;
    let string = |name: &str| -> Result<String> {
        match fit_result.read_branch(&file, name)? {
            BranchValues::Str(v) if !v.is_empty() => Ok(v[0].clone()),
            _ => Err(Error::Input(format!(
                "FitResult branch '{name}' is not a string"
            ))),
        }
    };
    let number = |t: &TreeReader, name: &str| -> Result<Vec<f64>> { numbers(t, &file, name) };
    let scalar = |name: &str| -> Result<f64> {
        number(&fit_result, name)?
            .first()
            .copied()
            .ok_or_else(|| Error::Input(format!("FitResult branch '{name}' is empty")))
    };

    let model = Model::new(
        string("kernel")?.parse()?,
        string("mean")?.parse()?,
        string("variance")?.parse()?,
        /* absent in files without pile-up support */
        scalar("pile_up").is_ok_and(|v| v != 0.),
    )?;
    let fit_method = match string("fit_method")?.as_str() {
        "Chi2" => FitMethod::Chi2,
        _ => FitMethod::Likelihood,
    };
    let normalization: Normalization = string("normalization")?.parse()?;

    let parameters = tree("Parameters")?;
    let names = match parameters.read_branch(&file, "name")? {
        BranchValues::Str(v) => v,
        _ => {
            return Err(Error::Input(
                "Parameters branch 'name' is not a string".into(),
            ));
        }
    };
    let values = number(&parameters, "value")?;
    let errors = number(&parameters, "error")?;
    let fixed = number(&parameters, "fixed")?
        .iter()
        .map(|&f| f != 0.)
        .collect();
    if names != model.param_names() || values.len() != model.n_params() {
        return Err(Error::Input(format!(
            "{}: parameters {names:?} do not match the model {}",
            path.display(),
            model.describe()
        )));
    }
    let covariance = number(&tree("Covariance")?, "value")?;
    if covariance.len() != values.len() * values.len() {
        return Err(Error::Input("covariance has the wrong size".into()));
    }

    let data_hist = string("data_hist")?;
    let data = DataHistogram::new(Hist1D::read_root(&file, &data_hist)?)?;
    Ok(StoredFit {
        model,
        bin_model: string("bin_model")?.parse()?,
        fit_method,
        normalization,
        names,
        values,
        errors,
        fixed,
        covariance,
        norm: scalar("norm")?,
        statistic: scalar("statistic")?,
        ndf: scalar("ndf")? as usize,
        fit_range: (scalar("fit_min")?, scalar("fit_max")?),
        valid: scalar("valid")? != 0.,
        data,
    })
}

/// Reads a numeric scalar branch as `f64`.
pub fn numbers(tree: &TreeReader, file: &FileReader, name: &str) -> Result<Vec<f64>> {
    Ok(match tree.read_branch(file, name)? {
        BranchValues::F64(v) => v,
        BranchValues::F32(v) => v.into_iter().map(f64::from).collect(),
        BranchValues::I32(v) => v.into_iter().map(f64::from).collect(),
        BranchValues::U32(v) => v.into_iter().map(f64::from).collect(),
        BranchValues::I16(v) => v.into_iter().map(f64::from).collect(),
        BranchValues::U16(v) => v.into_iter().map(f64::from).collect(),
        BranchValues::I64(v) => v.into_iter().map(|x| x as f64).collect(),
        BranchValues::U64(v) => v.into_iter().map(|x| x as f64).collect(),
        BranchValues::I8(v) => v.into_iter().map(f64::from).collect(),
        BranchValues::U8(v) => v.into_iter().map(f64::from).collect(),
        _ => {
            return Err(Error::Input(format!(
                "branch '{name}' is not a numeric scalar branch"
            )));
        }
    })
}
