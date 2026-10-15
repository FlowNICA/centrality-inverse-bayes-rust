# centrality-inverse-bayes-rust

Centrality determination in heavy-ion collisions with the **inverse Bayes**
(Γ-fit) method, written in Rust. It is a companion of
[centrality-glauber-rust](https://github.com/FlowNICA/centrality-glauber-rust)
with the same structure (a `fit` step and a `define-centrality` step, one
shared RON configuration, ROOT input and output), but instead of an
MC-Glauber model it reconstructs the relation between the centrality
observable and the impact parameter directly from the data, following

- S. J. Das, G. Giacalone, P.-A. Monard, J.-Y. Ollitrault, PRC 97 (2018) 014905,
  [arXiv:1708.00081](https://arxiv.org/abs/1708.00081) (Gaussian kernel);
- R. Rogly, G. Giacalone, J.-Y. Ollitrault, PRC 98 (2018) 024902,
  [arXiv:1804.03031](https://arxiv.org/abs/1804.03031) (Gamma kernel);
- J. D. Frankland et al. (INDRA), PRC 104 (2021) 034609,
  [arXiv:2011.04496](https://arxiv.org/abs/2011.04496) (selections, `P(b)`,
  alternative mean parametrization);
- the C++ macro `GammaFit2.C` of the
  [CentralityFramework](https://github.com/FlowNICA/CentralityFramework/tree/master/Framework/GammaFit).

**NOTE**: This is a very early version, developed with AI-assisted tools. It may not yet be ready for use in a formal analysis.

ROOT files are read and written with [oxiroot](https://github.com/mathieuouillon/oxiroot)
(pure Rust, no ROOT installation needed), and the fit uses Minuit (Migrad + Hesse)
from the pure Rust [minuit2](https://crates.io/crates/minuit2) crate.

## Method

Let `n` be the centrality observable (multiplicity, energy, ...) and
`c_b = ∫₀ᵇ P(b') db'` the *b-centrality*, the cumulative distribution of the
impact parameter. `c_b` is uniform on `[0, 1]` by construction, so

```
P(n) = ∫₀¹ P(n|c_b) dc_b
```

for any `P(b)`. The fluctuations of `n` at fixed impact parameter are
described by a kernel `P(n|c_b)` with mean `n̄(c_b)` and variance `σ²(c_b)`:

| `kernel` | `P(n|c_b)` |
|----------|------------|
| `Gamma` (default) | Gamma distribution, shape `n̄²/σ²`, scale `σ²/n̄` (Rogly et al.) |
| `Gauss` | Gaussian truncated to `n > 0` and renormalized (Das et al.) |

| `mean` | `n̄(c_b)` | parameters |
|--------|----------|------------|
| `ExpPoly(J)` (default `J = 3`) | `n_knee · exp(-Σ_{j=1}^{J} a_j c_b^j)` | `n_knee, a1..aJ` |
| `Indra` | `x_max · (1 - c_b^α)^γ + x_min` | `x_max, x_min, alpha, gamma` |

| `variance` | `σ²(c_b)` |
|------------|-----------|
| `Proportional` (default) | `θ · n̄(c_b)` (constant reduced variance, scenario A of Das et al.) |
| `Constant` | `θ · n̄(0)` (constant width, scenario B) |

The parameter vector is `theta` followed by the mean parameters. `n_knee = n̄(0)`
is the knee of the distribution, `σ(0) = √(θ n_knee)` the width of the
fluctuations in the most central collisions, and `c_knee = P(n > n_knee)` the
fraction of events above the knee, a measure of the centrality resolution.
As pointed out by Das et al., `P(n)` only constrains `n̄(c_b)` and the width of
the most central collisions, not how the width changes with `c_b`; the
`variance` choice is therefore a systematic variation.

Once fitted, Bayes' theorem gives the distribution of `c_b` in a class
`n₁ ≤ n < n₂`, and with `P(b)` that of the impact parameter:

```
P(c_b | n₁ ≤ n < n₂) = [P(n ≥ n₁ | c_b) - P(n ≥ n₂ | c_b)] / [c(n₁) - c(n₂)]
P(b | class)        = P(b) · P(c_b(b) | class),       c(n) = P(n' ≥ n)
```

The class borders are the values of `n` with the requested fraction `c(n)` of
the fitted distribution above them.

### Pile-up

With `pile_up: true`, a recorded event is with probability `p` (a fit
parameter, appended to the parameter vector) the sum of two independent
collisions, as in centrality-glauber-rust:

```
P_rec(n) = (1 - p) P(n) + p ∫₀¹∫₀¹ P(n | c₁, c₂) dc₁ dc₂
```

For the Gamma kernel with `Proportional` variance (the default) the
distribution of the sum is computed exactly: two collisions share the scale
`θ`, so their sum is a Gamma variable with mean `M = n̄(c₁) + n̄(c₂)` and scale
`θ`. The weights of all pairs of `c_b` nodes are collected on a grid of 1024
points in `M` (each shared linearly between its two nearest points, which
keeps the mean exactly and the fit statistic smooth), so the double integral
costs about as much as the single one. For the other models the single
collision distribution is convolved with itself numerically, on 8192 uniform
cells up to where the remaining fraction of events is below 1e-10 (the sum of
two cells is distributed triangularly and integrated over the bins); this is
slower (about 2 s for the toy fit below with the Gauss kernel).

Pile-up only changes how the recorded distribution is fitted: `n̄(c_b)`, `θ`,
the centrality `c(n)` and the classes refer to single collisions. In
`define-centrality`, the tail where pile-up dominates (from the first bin above
the knee where the expected pile-up exceeds the single collisions) is not
assigned to any class, and each class reports the fraction of its recorded
events that are pile-up. On a toy with `p = 0.01` the fit gives
`p = 0.0094 ± 0.0005` and the right `θ`, with ⟨b⟩ of the single collisions in
each class within 0.016 fm of the truth; without pile-up in the model, the
same data give `θ` 2.4 times too large and χ²/ndf ≈ 9.

### Numerics

- The integral over `c_b` is a composite Gauss-Legendre rule in `u = √c_b`
  (`cb_points` nodes), which puts more points at small `c_b`, where `P(n|c_b)`
  varies fastest (it is uniform in `b` for a geometric `P(b)`).
- With `bin_model: Integral` (default) the model in a bin is the integral of
  the density over the bin, via the regularized incomplete gamma function (or
  the normal distribution), so any bin width and continuous observables in
  any units work. For each `c_b` node the bin edges are walked outwards from
  the mean until the remaining tail probability is below 1e-22, so each edge
  is evaluated once and far tails cost nothing.
- Start values: the mean is pre-fitted to the data in the limit of vanishing
  fluctuations, where `n̄(c)` is the inverse of the measured centrality `c(n)`
  (linear least squares for `ExpPoly`); then `θ` (relative width
  `σ(0)/n_knee` from 0.4% to 60%) and the overall scale of the mean are scanned
  on a grid of 80 points, then (with pile-up) `p` from 1e-5 to 0.1, and Migrad
  starts from the best point. If Migrad does
  not converge, Simplex + Migrad are run; Hesse computes the covariance.
  User values in `params` replace the automatic ones.

### Multithreading

All heavy loops run on a [rayon](https://crates.io/crates/rayon) thread pool
(`n_threads`, default: all cores):

- each evaluation of the fit statistic splits the `c_b` nodes into chunks
  processed in parallel (summed in a fixed order, so results are
  reproducible and independent of the number of threads);
- the pile-up distribution collects the pairs of nodes on its grid in parallel;
- the start point scan evaluates its candidates in parallel;
- the class borders, the classes, the histograms of the joint distributions
  and the error propagation samples are computed in parallel.

On a 24-core machine, a fit of 490 multiplicity bins takes about 0.35 s
(4.7 s on one thread), and of 2500 bins of an ALICE V0-like amplitude
(`n_knee ≈ 20000`) about 2 s; `define-centrality` with 200 error samples takes
about 0.35 s. With pile-up, the fit of 490 bins takes about 1.3 s.

## Requirements

- Rust 1.95 or newer (edition 2024)

## Building

```sh
cargo build --release
```

## Usage

Each step is an executable in `src/bin/`; both read one configuration file,
[`config.ron`](config.ron), with a section per executable:

```sh
cargo run --release --bin fit -- config.ron
cargo run --release --bin define-centrality -- config.ron
# or
target/release/fit config.ron
target/release/define-centrality config.ron
```

- Only the input file(s) and histogram name are required; every other field
  defaults to the value shown in `config.ron`.
- A misspelled field or an invalid value stops the program with its line and
  column. A leading `~/` in a path is expanded to `$HOME`. Options are written
  without `Some(...)`.

### `fit`

```ron
fit: (
    data_file: "~/data.root",     // input file with the histogram of the observable
    data_hist: "hRefMult",
    out_dir: "./out/",
    fit_min: 10,                  // fit range: bins with centers in [fit_min, fit_max]
    // fit_max: 400,
    kernel: Gamma,                // Gamma or Gauss
    mean: ExpPoly(3),             // ExpPoly(J) or Indra
    variance: Proportional,       // Proportional or Constant
    pile_up: false,               // fit pile-up with probability p
    fit_method: Likelihood,       // Likelihood or Chi2
    normalization: Data,          // Data or Free
    bin_model: Integral,          // Integral or Center
    cb_points: 1024,
    // params: { "theta": (value: 1.0, min: 0.1, max: 10), "a3": (value: 0, fixed: true) },
    strategy: 1,
    // n_threads: 8,
),
```

- `fit_method: Likelihood` (default) minimizes the Poisson likelihood ratio
  `2 Σ [m - d + d ln(d/m)]` (Baker-Cousins) over all bins in the range,
  including empty ones. It needs raw counts. `Chi2` is the Neyman χ² with the
  data errors (empty bins skipped); it works for weighted or scaled
  histograms, but the sparse tail biases it: on the toy data below it gives
  `θ` about 2% low, the likelihood fit is unbiased. Both are `-2 ln L` up to a
  constant (Minuit `up = 1`); the printed value is divided by the number of
  degrees of freedom.
- `normalization: Data` (default, as Das et al. and `GammaFit2.C`): the model
  is normalized to all entries of the histogram, including those outside the
  fit range and in under/overflow, so `P(n)` integrates to one over all events
  and the fraction of events below the fit range constrains the fit. The
  histogram must contain all (triggered) events. With `Free` the number of
  events is fitted (profiled analytically) and the model extrapolates below
  the fit range, e.g. for data lost by trigger inefficiency. This
  extrapolation is weakly constrained: on the toy data with `fit_min: 10` the
  fitted total differs from the true one by several percent.
- `params`: start value (`value`), step, limits (`min`, `max`) and `fixed` for
  any parameter, by name (`theta`, `n_knee`, `a1`, ... or `x_max`, `x_min`,
  `alpha`, `gamma`). By default `theta`, `n_knee`, `x_max`, `x_min`, `alpha`
  and `gamma` are limited to positive values.
- `pile_up`: adds the parameter `p` (default limits `[0, 0.5]`, it can be
  fixed in `params` like any other); see [Pile-up](#pile-up).
- `cb_points`: more points are only needed for very narrow kernels; the toy
  fits below give the same parameters with 1024 and 8192 points.
- The `Gauss` kernel has a left tail below `n = 0` for peripheral collisions;
  use a larger `fit_min` with it (Das et al. cut at 60-80% centrality).

The fit writes `<out_dir>/bayes_fit.root` with

- the data histogram (under its own name), `fit_histo` (fitted expected
  counts in the data bins) and `fit_ratio` (data / fit);
- with pile-up, `fit_sng_histo` and `fit_plp_histo`: the single collision and
  pile-up parts of `fit_histo`;
- `Mean_VS_Cb` (TGraphErrors, `n̄(c_b)` with `σ(c_b)` as errors) and
  `Sigma_VS_Cb`;
- `Centrality_VS_Multiplicity` (TGraph): the centrality `c(n)` in percent of the
  fitted distribution at the bin edges;
- `Cb_VS_Multiplicity` (TH2D): expected single-collision events versus `n`
  and `c_b`;
- tree `FitResult`: model (`kernel`, `mean`, `variance`, `bin_model`,
  `fit_method`, `normalization` as strings), `statistic`, `ndf`, `chi2_ndf`,
  `pile_up`, `pile_up_probability`, `norm`, `data_total`, `fit_min`, `fit_max`, `n_knee`, `sigma_knee`, `c_knee`,
  `valid`, `edm`, `n_calls`, the data file and histogram name;
- tree `Parameters`: `name`, `value`, `error`, `fixed`, one entry per parameter;
- tree `Covariance`: `row`, `col`, `value` of the full covariance matrix.

### `define-centrality`

```ron
centrality: (
    fit_file: "./out/bayes_fit.root",
    out_dir: "./out/",
    final_file: "FINAL.root",
    classes: [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],   // or n_classes: 10
    borders: BinEdges,                                          // or Exact
    impact_parameter: Geometric(sigma_inel: 685.0),             // optional
    b_bins: 200,
    cb_bins: 500,
    cb_points: 4096,
    n_error_samples: 200,
    table_formats: [Csv, Cpp],                                  // any of Tex, Csv, Cpp
    table_name: "centrality_table",
),
```

- `borders: BinEdges` (default) snaps each border to the data bin edge closest
  in centrality, so each class is a set of whole bins (integer multiplicities),
  and the classes are contiguous. The actual fractions of the fitted
  distribution (`FitMinPercent`, `FitMaxPercent`) then differ slightly from the
  nominal ones. With `Exact` the borders are the exact values with the nominal
  fractions above them, e.g. for a continuous observable. The most central
  class is open above, the most peripheral one extends to `n = 0`.
- `impact_parameter` sets `P(b)` of inelastic events, to turn `c_b` into `b`
  (without it only `c_b` is reconstructed, which is model independent):
  - `Geometric(sigma_inel: 685.0)`: `c_b = πb²/σ_inel`, `σ_inel` in fm²
    (1 b = 100 fm²);
  - `Fermi(sigma_inel: 685.0, width: 0.4)` or `Fermi(b0: 14.5, width: 0.4)`:
    `P(b) ∝ 2πb / (1 + exp((b - b0)/width))` (Frankland et al.), with `b0` given
    or computed from `σ_inel`;
  - `Histogram(file: "...", hist: "hB")`: a histogram of `b` (fm) of
    inelastic events, e.g. from a model;
  - `Tree(file: "...", tree: "nt_Au_Au", branch: "B")`: the `b` of each event,
    e.g. the `B` branch of the MC-Glauber tree used by centrality-glauber-rust;
  - `Glauber(nucleus_a: "Au", nucleus_b: "Au", sigma_nn: 30.0)`: the optical
    Glauber model from the nuclear density profiles, see
    [Impact parameter from the nuclear density](#impact-parameter-from-the-nuclear-density).
- `n_error_samples`: the fit uncertainties are propagated by drawing parameter
  sets from a multivariate normal distribution with the fit covariance (fixed
  parameters stay fixed) and recomputing everything; the standard deviations
  are written as `*Err` columns. The borders stay fixed for `<c_b>` and `<b>`;
  `MinBorderErr`/`MaxBorderErr` are the spreads of the exact borders.

`FINAL.root` in `out_dir` contains:

- tree `Result`, one entry per class: `Ncc`, `MinPercent`, `MaxPercent`,
  `MinBorder`, `MaxBorder` (`Int_t` if all are integers, as `printFinal.C`
  expects, otherwise `Double_t`), `FitMinPercent`, `FitMaxPercent`,
  `DataPercent` (entries of the data in the class), `PileUpPercent` (percent
  of the recorded events of the fit in the class that are pile-up),
  `CbAverage`, `CbWidth`
  (RMS), `CbMin`, `CbMax`, with `impact_parameter` also `BAverage`, `BWidth`,
  `BMin`, `BMax`, and with error propagation `MinBorderErr`, `MaxBorderErr`,
  `CbAverageErr`, `CbWidthErr`, `BAverageErr`, `BWidthErr`. `CbMin`/`CbMax`
  and `BMin`/`BMax` are the range a sharp cut on `c_b` (on `b`) at the class
  fractions would select; the mean can lie outside it, as `n` fluctuates at
  fixed `b`;
- `Cb_VS_CentralityClass <min>%-<max>%` and `B_VS_CentralityClass <min>%-<max>%`:
  the probability densities `P(c_b|class)` and `P(b|class)` (1/fm);
- `B_Distribution`: `P(b)` of all inelastic events (1/fm), with the source of
  `P(b)` in the title;
- `Cb_average_VS_Centrality`, `B_average_VS_Centrality`: the means versus
  centrality with the RMS as errors, and `Fit_B_Mean` (TGraphErrors, as written
  by `GammaFit2.C` and read by its `printFinal.C`);
- `CentralityClass <min>%-<max>%` and `CentralityClass_Fit <min>%-<max>%`: the
  data and the fit in each class;
- `Centrality_vs_Multiplicity`: the upper percent of the class of each bin;
- `fit_histo`, the data histogram, and `B_VS_Multiplicity` (TH2D, expected
  single-collision events versus `n` and `b`).

The table is printed to stdout and written in the requested formats; the
`.C` macro has arrays of the borders and results and `GetCentMult(mult)`,
which returns the middle of the class of `mult` in percent (the most central
class is open above).

### Impact parameter from the nuclear density

`impact_parameter: Glauber(...)` computes `P(b)` from the density profiles of
the colliding nuclei in the optical Glauber model (Miller et al., Ann. Rev.
Nucl. Part. Sci. 57 (2007) 205), with no Monte Carlo:

```
T_A(s)    = ∫ ρ_A(√(s² + z²)) dz              thickness, ∫ T_A d²s = 1
T_AB(b)   = ∫ T_A(s) T_B(|s - b|) d²s          overlap,   ∫ T_AB d²b = 1
P_inel(b) = 1 - (1 - σ_NN T_AB(b))^{AB}
P(b)      = 2πb P_inel(b) / σ_inel,            σ_inel = ∫ 2πb P_inel(b) db
```

```ron
impact_parameter: Glauber(
    nucleus_a: "Au",
    nucleus_b: "Au",
    sigma_nn: 30.0,          // mb; or sqrt_snn: 2.2 (GeV, from 2.0223)
    // sigma_np: 35.0,       // then sigma_nn is the pp cross section
    // nuclei_file: "~/my_nuclei.ron",
),
```

- The nuclei are those of [oxy-glauber](https://github.com/FlowNICA/oxy-glauber):
  [`data/nuclei.ron`](data/nuclei.ron) is its `data/nuclei.ron` (GPL-3.0),
  embedded in the binary at compile time; `nuclei_file` reads another file in
  the same format. Every analytic profile is supported: Woods-Saxon (3pF,
  3pG), proton-neutron 3pF, harmonic oscillator, the 1970 oxygen profile, hard
  sphere, the proton profiles (for p+A) and the Hulthén deuteron (the
  nucleon density at half the pair separation). `FromFile` and `FromGraph`
  (e.g. `O`, `C`, `Osat`) have no analytic density; use e.g. `Opar` or `Oho`.
- Deformed nuclei (`Ellipsoid`, `DeformedBox`, `DeformedTF2`, e.g. `U`, `Xe2`)
  use the density averaged over the orientations, and the reweighted profiles
  (`*rw`) their Woods-Saxon density: the reweighting polynomial only
  compensates the minimum distance between nucleons sampled in the Monte Carlo.
  Densities are cut at `max_r`, where oxy-glauber stops sampling.
- `σ_NN` is `sigma_nn` (with `sigma_np`, `sigma_nn` is the pp and nn cross
  section, and the effective one is averaged over the pp, nn and np pairs of
  the two nuclei, weighted by their numbers of protons and neutrons), or it is
  computed from `sqrt_snn`:
  - from 10 GeV, with the parametrization of Loizides et al.
    (arXiv:2011.14909), as in oxy-glauber, for all pairs;
  - from 2.0223 to 3.378 GeV, from the measured inelastic pp and np cross
    sections of Bystricky et al., J. Physique 48 (1987) 199, Table VI (the
    tables of TGlauberMC and oxy-glauber), linearly interpolated in the
    kinetic energy of the beam on a fixed target,
    `T = (s - 2m²)/(2m) - m` with the mean nucleon mass `m = 0.93892 GeV`,
    and averaged over the pairs as above. The range is that of the np table
    (`T` = 0.3-4.2 GeV); below it is the pion production threshold
    (2.016 GeV), where the inelastic cross section vanishes: there give
    `sigma_nn` and `sigma_np`;
  - from 3.378 to 10 GeV, where the np table has ended, from the measured
    `σ_pp` of the same table (it extends to `T` = 425 GeV) and
    `σ_np = r σ_pp`, where the ratio `r` goes linearly in `√s_NN` from its
    measured value at the end of the np table (1.059) to 1 at 10 GeV. This is
    an assumption, not data: inelastic np and pp cross sections become equal
    at high energies, but how fast is not measured in the table. `σ_np` is
    continuous at 3.378 GeV; at 10 GeV the switch to the parametrization is a
    step of about 2% in `σ_NN` (29.9 to 30.6 mb), 0.2% in `σ_inel` for
    Au+Au. The description of `P(b)` says which source was used.

  | `√s_NN`, GeV | `σ_pp`, mb | `σ_np`, mb | Au+Au `σ_NN`, mb | Au+Au `σ_inel`, b |
  |------|-------|-------|-------|-------|
  | 3.5  | 28.3  | 29.9 (r = 1.058) | 29.1 | 6.72 |
  | 4.0  | 29.2  | 30.7 (r = 1.053) | 29.9 | 6.73 |
  | 7.7  | 29.8  | 30.4 (r = 1.021) | 30.1 | 6.74 |
  | 9.2  | 29.8  | 30.0 (r = 1.007) | 29.9 | 6.73 |
  | 10.0 | 30.6 (parametrization) | = `σ_pp` | 30.6 | 6.75 |

  Close to the threshold the inelastic cross section, and with it `P(b)` and
  `σ_inel`, changes fast with energy. For Au+Au:

  | `√s_NN`, GeV | `T`, GeV | `σ_pp`, mb | `σ_np`, mb | effective `σ_NN`, mb | `σ_inel`, b |
  |-------|-------|-------|-------|-------|-------|
  | 2.023 | 0.302 | 0.09  | 0.006 | 0.05  | 1.35  |
  | 2.1   | 0.471 | 4.28  | 2.38  | 3.37  | 5.34  |
  | 2.2   | 0.700 | 17.0  | 9.76  | 13.5  | 6.25  |
  | 2.416 | 1.23  | 26.4  | 21.1  | 23.8  | 6.60  |
  | 3.0   | 2.91  | 27.5  | 27.9  | 27.7  | 6.69  |

  Some analyses at these energies use the total (elastic + inelastic) NN
  cross section instead; it can be given with `sigma_nn`/`sigma_np`. The
  cross sections used are printed with the table and in the title of
  `B_Distribution`.
- The computed `σ_inel` is printed with the table and in the title of
  `B_Distribution`: 6.93 b for Au+Au at `σ_NN = 42 mb`, 7.63 b for Pb+Pb at
  64 mb and 2.01 b for p+Pb at 70 mb, close to Monte Carlo Glauber values. The
  optical model has no fluctuations of the nucleon positions, so `P(b)`
  differs slightly from the Monte Carlo one in the most peripheral events;
  for that, use `Tree` with the `B` branch of an MC-Glauber tree.
- The `c_b` results do not depend on `P(b)`; compared with `Geometric` with
  the same `σ_inel`, central classes have the same `<b>`, while in
  peripheral classes the diffuse nuclear edge gives larger `b`.

## Validation

`examples/toy.rs` generates toy data with a known model: geometric `b` with
`σ_inel = 685 fm²`, `n` from a Gamma kernel with `θ = 1.1` and
`n̄ = 220 exp(-4c_b + c_b² - 1.5c_b³)`, floored to integers, plus the tree of
the true `b` and `n`:

```sh
cargo run --release --example toy -- toy.root 1000000 1          # add --large for n_knee ≈ 20000
```

Fitting 10⁶ such events (`fit_min: 10`) and defining the classes gives a mean
impact parameter within 0.006 fm of the true one (the mean `b` of the toy
events in the class) in every class from 0-5% to 90-100%, with RMS widths
within 0.012 fm, and over several toys the fitted parameters scatter around
the true values within their errors. The integration tests
(`tests/end_to_end.rs`) check this, the other kernels and parametrizations,
the impact parameter inputs, and that a continuous observable fitted in GeV
and in MeV gives the same classes.

## Library use

```rust
use centrality_inverse_bayes_rust::{CentralityConfig, FitConfig, ImpactParameter, centrality};

let mut fit = FitConfig::new("data.root", "hRefMult");
fit.fit_min = Some(10.);
fit.out_dir = "out".into();
let result = centrality_inverse_bayes_rust::run(fit)?;      // fits and writes out/bayes_fit.root
println!("n_knee = {}, c_knee = {}", result.n_knee, result.c_knee);

let mut config = CentralityConfig::new("out/bayes_fit.root");
config.impact_parameter = Some(ImpactParameter::Geometric { sigma_inel: 685. });
let classes = centrality::run(&config)?;                   // writes FINAL.root and the tables
```

For finer control use `Fitter` (e.g. `Fitter::objective` for the fit statistic
as a function of the parameters), `Model` (kernel, `n̄(c_b)`, bin
probabilities, `c(n)`), `output::read_fit` and `centrality::compute`.

## Differences from `GammaFit2.C`

- The sign convention of the coefficients follows the papers:
  `n̄ = n_knee exp(-a1 c_b - a2 c_b² - a3 c_b³)`, so `a_j` here is `-a_j` of
  `GammaFit2.C`, which uses `exp(a1 c_b + ...)`. `teta` there is `theta` here.
- The model is integrated over each bin instead of being evaluated at the bin
  center (`bin_model: Center` gives the `TH1::Fit` behaviour), and the
  Poisson likelihood is the default instead of a final χ² fit.
- The start values are found automatically instead of being set in the macro.
- Class borders are found on the fitted `c(n)` without integer stepping, and
  `<b>` and its RMS are computed by quadrature instead of from 20000 random
  draws of `P(b)`; the min/max of `b` come from the sharp-cut correspondence
  instead of evaluating a `pol5` fit to the class means.
- `σ_inel` is given in the configuration (or comes from a histogram or tree of
  `b`) instead of being fitted to a model `b`-vs-multiplicity histogram.
- Uncertainties of the fit are propagated to the classes.

## Development

```sh
cargo test --release                                  # unit + end-to-end tests
cargo test --release --test end_to_end recovers       # a single test
cargo clippy --all-targets -- -D warnings
cargo fmt
```
