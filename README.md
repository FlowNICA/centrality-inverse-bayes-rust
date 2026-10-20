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
| `Glauber(...)` | `μ · [f ⟨N_part⟩(c_b) + (1 - f) ⟨N_coll⟩(c_b)]` from the optical Glauber model | `mu, f` |
| `Histogram(...)` | `mean_scale · ⟨n⟩(b(c_b))` from a 2-D histogram of `n` and `b` | `mean_scale` |
| `Idrisov(...)` | `m1 · max(f_m(c_b), 0.1)` with a `TF1` `f_m` of a model | `m1` |

| `variance` | `σ²(c_b)` |
|------------|-----------|
| `Proportional` (default) | `θ · n̄(c_b)` (constant reduced variance, scenario A of Das et al.) |
| `Constant` | `θ · n̄(0)` (constant width, scenario B) |
| `Histogram(...)` | `variance_scale · Var(n)(b(c_b))` from a 2-D histogram of `n` and `b`; `variance_scale` replaces `θ` |
| `Idrisov(...)` | `max(m1² f_D(c_b) + m1 m2 n̄(c_b), 0.1)` with a `TF1` `f_D` of a model; with the `Idrisov` mean only, `m2` replaces `θ` |

The `Glauber` mean is the two-component model of the MC-Glauber fit of
centrality-glauber-rust (`mode: "Default"`), with the mean numbers of
participants and binary collisions of inelastic events at each `c_b` from
the optical Glauber model instead of a Monte Carlo:

```ron
mean: Glauber(nucleus_a: "Au", nucleus_b: "Au", sigma_nn: 30.0),   // or sqrt_snn: 2.4
```

The nuclei and cross section are given as for
[`impact_parameter: Glauber`](#impact-parameter-from-the-nuclear-density).
`N_coll(b) = AB σ_NN T_AB(b)` and
`N_part(b) = A ∫ T_A [1 - (1 - σ_NN T_B)^B] d²s + B ∫ T_B [1 - (1 - σ_NN T_A)^A] d²s`
are divided by `P_inel(b)` (means over inelastic events) and mapped to
`c_b` with the optical `c_b(b)`; for Au+Au at 30 mb, `⟨N_part⟩ = 371` and
`⟨N_coll⟩ = 881` at `b = 0`. In the most peripheral events (about the last 7%
for Au+Au) the optical `N_part / P_inel` falls below 2, as the optical model
treats the nucleons and `P_inel` with different approximations; there they
are set to their minimum for an inelastic collision, 2 and 1. With only two
mean parameters the shape of `n̄(c_b)` is fixed by the geometry, `μ` is the
number of particles per source and `f` (limited to `[0, 1]`) the fraction of
participant-like (soft) production. The start values come from a linear
least squares fit of `μf` and `μ(1 - f)` in the limit of vanishing
fluctuations. On a toy of 10⁶ events generated with this model (Au+Au, 30 mb,
`θ = 1`, `μ = 0.3`, `f = 0.8`) the fit gives `θ = 0.989 ± 0.017`,
`μ = 0.2997 ± 0.0015`, `f = 0.799 ± 0.005` in 0.4 s. Unlike in the Monte
Carlo Glauber fit, the fluctuations come from the kernel (`θ`), not from
event-by-event `N_part` and `N_coll`.

### Mean and variance from a histogram of `n` versus `b`

The `Histogram` mean and variance take the dependence on the impact
parameter from a 2-D histogram of `n` and `b` (fm) of **all inelastic
events**, e.g. of a model transported through the detector simulation,
with one fit parameter each:

```ron
mean: Histogram(file: "~/model.root", hist: "h2_n_b", x_axis: "b", y_axis: "n"),
variance: Histogram(file: "~/model.root", hist: "h2_n_b", x_axis: "b", y_axis: "n"),
```

`x_axis` and `y_axis` say which axis holds `"n"` and which `"b"` (both
orders work). In each bin of `b` the mean and variance of `n` are computed
from the bin centers of `n` (for integer multiplicities in bins `[k, k + 1)`
that is `k + 0.5`, as in the model, where a bin is the integral of a
continuous density); they are interpolated linearly between the bin centers
of `b` and constant beyond the first and last filled bins (empty bins are
skipped).

`c_b` is turned into `b` with `P(b)` from the projection of the same
histogram on `b` (all entries in each bin of `b`, including the under- and
overflow of `n`), so that the moments and `P(b)` come from the same events.
This is why the histogram must contain all inelastic events, without any
trigger, vertex or track selection: `c_b` is the fraction of all inelastic
events with a smaller `b`, to which the centrality percents refer. If, for
example, a selection keeps 79% of the events, losing them only at large
`b`, the projection gives `c_b` 1/0.79 = 1.27 times too large at all smaller
`b`, and a 0-10% class is really about 0-8% (the fit may even look better,
as it then describes the selected sample). Entries outside the axis of `b`
stop the fit, as `P(b)` would miss them. The `impact_parameter` of
`define-centrality` is independent of it and only used for the classes; for
the same `P(b)` there, use `Histogram2D` with the same histogram. Then

```
n̄(c_b)  = mean_scale     · ⟨n⟩(b(c_b))
σ²(c_b) = variance_scale · Var(n)(b(c_b))
```

so that, if the model describes the data, both scales are 1 (e.g. a
different efficiency shows up as `mean_scale`). The two can be combined with
the other choices, e.g. `ExpPoly` with a `Histogram` variance (parameters
`variance_scale, n_knee, a1, ...`) or a `Histogram` mean with `Proportional`
variance (`theta, mean_scale`). As for the `Glauber` mean, the shape of
`n̄(c_b)` is fixed, so `normalization: Free` gives no warning. The start value of
`mean_scale` comes from the measured centrality in the limit of vanishing
fluctuations, and `variance_scale` is scanned from 0.05 to 20.

On toys (10⁶ events with a known `n̄(c_b)` and Gamma kernel filled into the
2-D histogram, 3·10⁵ events fitted with `normalization: Free`), both scales
are 1 within their errors (e.g. `variance_scale = 0.967 ± 0.031`, `mean_scale`
within 0.2%), `N` within 2% and `<b>` of the classes within 0.1 fm of the
truth. The kernel shape stays a Gamma (or Gauss) distribution with these
moments: if the distribution of `n` at fixed `b` of the model is not of this
shape, the fit is poor even with both scales at 1.

### Mean and variance from a tree of `b` and `n`

The `Tree` mean and variance take the same dependence from the values of `b`
(fm) and `n` of each event in a tree of **all inelastic events** (e.g. an
MC-Glauber or transport model tree), with the same parameters `mean_scale`
and `variance_scale` and the same meaning:

```ron
mean: Tree(file: "~/model.root", tree: "events", b_branch: "B", n_branch: "Mult"),
variance: Tree(file: "~/model.root", tree: "events", b_branch: "B", n_branch: "Mult", b_bins: 100),
```

As the events need no binning, the moments are built versus `c_b` directly:

- the `c_b` of an event is the fraction of events of the tree with a smaller
  `b`, and `P(b)` is that of all events of the tree (as for
  `impact_parameter: Tree`, with which `define-centrality` can use the same
  `P(b)`);
- the events are grouped into `b_bins` bins (default 100, at least 10
  events each) with equal numbers of events, i.e. of equal width in `c_b`, so
  all bins have the same statistics;
- the mean of `n` of each bin is computed from the values of `n` (no
  binning, so no `(bin width)²/12` and no lost under/overflow, unlike a
  histogram), placed at the bin center in `c_b`, interpolated linearly, and
  extrapolated linearly to `c_b = 0` and 1, i.e. by half a bin: `n̄` falls
  fastest at `c_b = 0`, where it sets the knee, and holding it constant there
  would underestimate it;
- the variance of a bin is that of `n` around this interpolated mean at the
  `c_b` of each event, so the change of the mean across the bin is not
  counted as a fluctuation.

On a toy (a tree of 10⁶ events with a known `n̄(c_b)` and Gamma kernel,
3·10⁵ events fitted with `normalization: Free`) the scales are 1 within
their errors and stable for `b_bins` from 30 to 1000 (`variance_scale` 1.04,
0.985, 0.983, 0.981 for 30, 100, 300, 1000 bins, `± 0.031`; `mean_scale`
1.0003-1.0016, `± 0.0013`; `N` within 1.2%). With 10 bins the mean is too
coarse (`variance_scale = 1.49`, and a worse fit). `<b>` of the classes is
within 0.1 fm of the truth.

### Mean and variance of Idrisov et al.

The `Idrisov` mean and variance are the parametrization of the Γ-fit of
D. Idrisov et al. (`GammaFit.cpp`): the mean `f_m(c_b)` and the variance
`f_D(c_b)` of the observable versus `c_b` of a model (e.g. DCM-SMM through
the simulation of the detector) are given as functions, `TF1`s fitted in
ROOT, and the data fix two parameters:

```
n̄(c_b)  = m1 · max(f_m(c_b), 0.1)
σ²(c_b) = max(m1² f_D(c_b) + m1 m2 n̄(c_b), 0.1)
```

`m1` scales the observable (e.g. a different efficiency), with which the
variance of the model scales as `m1²`, and `m2` adds a part proportional to
the mean (`m2 = 0`: the variance of the model; it may be negative). The
lower limits 0.1 are those of the original. As the variance depends on `m1`,
the two are used together:

```ron
mean: Idrisov(file: "~/MeanVar.root", func: "F_mM"),       // func: default "F_mM"
variance: Idrisov(file: "~/MeanVar.root", func: "F_DM"),   // func: default "F_DM"
```

The parameters are `m2` (in place of `theta`) and `m1`. The functions are
read with their formula and parameters and tabulated on 10⁴ intervals of
`c_b`; formulas that oxiroot cannot parse directly, such as the `polN` of
ROOT (written with `TMath::Sq`), are rewritten (`TMath::Sq(x)` to
`pow(x,2)`, `TMath::Exp` to `exp`, ...). A `TF1` of a compiled C++ function
(e.g. `fPbFitFull`, the `P(b)` of the original files) has no formula and
cannot be read; `P(b)` is the `impact_parameter` of `define-centrality`. The
start value of `m1` comes from the measured centrality in the limit of
vanishing fluctuations, and `m2` is scanned from -0.5 to 6. As for the
other means from a model, `normalization: Free` gives no warning.

On a toy with `f_m` the `ExpPoly(3)` toy mean of the
[Validation](#validation), `f_D = 1.1 f_m`, and `m1 = 0.8`, `m2 = 0.5`
(4·10⁵ events), the fit gives `m1 = 0.8004 ± 0.0012`, `m2 = 0.43 ± 0.09`,
and `<b>` and its RMS in every class within 0.015 fm of the truth. For the
`F_mM` (exponential of a polynomial) and `F_DM` (`pol14`) of the original
files the tabulated functions agree with ROOT.

The parameter vector is `theta` (or `variance_scale`, or `m2`) followed by
the mean parameters. `n_knee = n̄(0)`
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
# or both steps in one go
target/release/run-all config.ron
```

`run-all` runs `fit` and then `define-centrality` with the same file, which
needs both sections. Both are read and checked before the fit starts, and the
centrality step uses the fit file just written (`<out_dir>/bayes_fit.root` of
the `fit` section; a different `fit_file` is replaced, with a warning).

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
    mean: ExpPoly(3),             // ExpPoly(J), Indra, Glauber(...) or Histogram(...)
    variance: Proportional,       // Proportional, Constant or Histogram(...)
    pile_up: false,               // fit pile-up with probability p
    fit_method: Likelihood,       // Likelihood or Chi2
    normalization: Data,          // Data, Efficiency(eps) or Free
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
- `normalization` sets the number `N` of all inelastic events, to which the
  centrality percents refer (the total cross section); see
  [Trigger efficiency](#trigger-efficiency):
  - `Data` (default, as Das et al. and `GammaFit2.C`): `N` is the number of
    entries of the histogram, including those outside the fit range and in
    under/overflow, so `P(n)` integrates to one over the recorded events and
    the fraction of events below the fit range constrains the fit. The
    histogram must contain all inelastic events (trigger efficiency 1).
  - `Efficiency(0.92)`: `N` is the number of entries divided by a trigger
    efficiency known from elsewhere, e.g. a simulation of the trigger.
  - `Free`: `N` is fitted from the bins in the fit range (profiled
    analytically) and the model extrapolates below it.
- `params`: start value (`value`), step, limits (`min`, `max`) and `fixed` for
  any parameter, by name (`theta`, `variance_scale` or `m2`, `n_knee`, `a1`,
  ..., or `x_max`, `x_min`, `alpha`, `gamma`, or `mu`, `f`, or `mean_scale`,
  or `m1`). By default `theta`, `variance_scale`, `mean_scale`, `m1`,
  `n_knee`, `x_max`, `x_min`, `alpha`, `gamma` and `mu` are limited to
  positive values, and `f` to `[0, 1]`; `m2` is free.
- `pile_up`: adds the parameter `p` (default limits `[0, 0.5]`, it can be
  fixed in `params` like any other); see [Pile-up](#pile-up).
- `cb_points`: more points are only needed for very narrow kernels; the toy
  fits below give the same parameters with 1024 and 8192 points.
- The `Gauss` kernel has a left tail below `n = 0` for peripheral collisions;
  use a larger `fit_min` with it (Das et al. cut at 60-80% centrality).

The fit writes `<out_dir>/bayes_fit.root` with

- the data histogram (under its own name), `fit_histo` (fitted expected
  counts in the data bins; bins below one count are left empty, so that, like
  the data, it starts at 1 on a logarithmic y axis) and `fit_ratio` (data /
  fit, from the full expected counts);
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
  `pile_up`, `pile_up_probability`, `norm` (all events `N`), `data_total`,
  `efficiency` (`data_total / norm`), `efficiency_below_fit` (data over fit
  below the fit range), `fit_min`, `fit_max`, `n_knee`, `sigma_knee`, `c_knee`,
  `valid`, `edm`, `n_calls`, the data file and histogram name;
- tree `BestResult`, the best fit as in centrality-glauber-rust: one entry
  with a branch per parameter (e.g. `theta`, `n_knee`, `a1`, ...) holding its
  value, `<name>_error` with its error (0 if fixed), `chi2` (the fit statistic
  per degree of freedom, `-2ln(L/Lsat)/ndf` or `chi2/ndf`) and `ndf`;
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
    mult_plot: (file: "centrality_result.png"),                 // None: no picture
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
  - `Histogram2D(file: "...", hist: "h2_n_b", x_axis: "b", y_axis: "n")`: the
    projection on `b` of a 2-D histogram of `n` and `b` of all inelastic
    events (`x_axis`/`y_axis` as for the
    [`Histogram` mean and variance](#mean-and-variance-from-a-histogram-of-n-versus-b)),
    e.g. the same histogram as in the fit, so that the classes use the same
    `P(b)` as the fit;
  - `Tree(file: "...", tree: "nt_Au_Au", branch: "B")`: the `b` of each event,
    e.g. the `B` branch of the MC-Glauber tree used by centrality-glauber-rust;
  - `Glauber(nucleus_a: "Au", nucleus_b: "Au", sigma_nn: 30.0)`: the optical
    Glauber model from the nuclear density profiles, see
    [Impact parameter from the nuclear density](#impact-parameter-from-the-nuclear-density).
- The largest impact parameter `b_max` comes from `P(b)`: `√(σ_inel/π)` for
  `Geometric`; for the tabulated ones (`Fermi`, `Glauber`, `Histogram`,
  `Histogram2D`, `Tree`) the first `b` where `c_b` reaches `1 - 10⁻⁶`, i.e.
  where `P(b)` ends, not the end of the table: for `Histogram`/`Histogram2D`
  the upper edge of the last filled bin of `b` (empty bins up to the end of the
  axis are ignored), for `Tree` the largest `b`, for `Fermi` a few widths above
  `b0`, for `Glauber` the edge of the overlap of the nuclear densities (19.1 fm
  for Au+Au at 30 mb, 19.4 fm for Pb+Pb at 64 mb, 11.9 fm for p+Pb at 70 mb;
  99.9% of the events are below 16.7, 17.4 and 10.0 fm). Its tail of `10⁻⁶`
  depends on the cut of the densities at `max_r` of `nuclei.ron`, where
  oxy-glauber stops sampling too. All `b` are within `[0, b_max]`: the means,
  the sharp-cut ranges (`BMax` of the most peripheral class is `b_max`), and
  the histograms of `b`, whose range is `[0, b_max]` unless `b_max` is given
  in the configuration; a `b_max` below that of `P(b)` cuts the histograms (a
  warning gives the fraction of events lost; the means are not affected).
  The table prints `b_max`.
- `n_error_samples`: the fit uncertainties are propagated by drawing parameter
  sets from a multivariate normal distribution with the fit covariance (fixed
  parameters stay fixed) and recomputing everything; the standard deviations
  are written as `*Err` columns. The borders stay fixed for `<c_b>` and `<b>`;
  `MinBorderErr`/`MaxBorderErr` are the spreads of the exact borders.

`FINAL.root` in `out_dir` contains:

- tree `BestResult`: the best fit of the fit file (see [`fit`](#fit)), so
  that the output carries the fit it was made with;
- tree `Result`, one entry per class: `Ncc`, `MinPercent`, `MaxPercent`,
  `MinBorder`, `MaxBorder` (`Int_t` if all are integers, as `printFinal.C`
  expects, otherwise `Double_t`), `FitMinPercent`, `FitMaxPercent`,
  `DataPercent` (entries of the data in the class over all events `N` of the
  fit), `Efficiency` (data over fit in the class), `PileUpPercent` (percent
  of the recorded events of the fit in the class that are pile-up),
  `CbAverage`, `CbWidth`
  (RMS), `CbMin`, `CbMax`, with `impact_parameter` also `BAverage`, `BWidth`,
  `BMin`, `BMax`, and with error propagation `MinBorderErr`, `MaxBorderErr`,
  `CbAverageErr`, `CbWidthErr`, `BAverageErr`, `BWidthErr`. `CbMin`/`CbMax`
  and `BMin`/`BMax` are the range a sharp cut on `c_b` (on `b`) at the class
  fractions would select; the mean can lie outside it, as `n` fluctuates at
  fixed `b`;
- `Cb_VS_CentralityClass <min>%-<max>%` and `B_VS_CentralityClass <min>%-<max>%`:
  the probability densities `P(c_b|class)` and `P(b|class)` (1/fm) of the
  recorded events;
- `B_Distribution`: `P(b)` of all inelastic events (1/fm), with the source of
  `P(b)` in the title;
- `Cb_average_VS_Centrality`, `B_average_VS_Centrality`: the means versus
  centrality with the RMS as errors, and `Fit_B_Mean` (TGraphErrors, as written
  by `GammaFit2.C` and read by its `printFinal.C`);
- `CentralityClass <min>%-<max>%` and `CentralityClass_Fit <min>%-<max>%`: the
  data and the fit in each class;
- `Centrality_vs_Multiplicity`: the upper percent of the class of each bin;
- `Trigger_Efficiency`: the trigger efficiency of each bin used for the
  classes, and `Trigger_Efficiency_VS_Cb` and, with `impact_parameter`,
  `Trigger_Efficiency_VS_B`: the efficiency of the events in each bin of
  `c_b` (`cb_bins`) and of `b` (`b_bins`) (see
  [Trigger efficiency](#trigger-efficiency));
- `fit_histo`, the data histogram, and `B_VS_Multiplicity` (TH2D, expected
  single-collision events versus `n` and `b`).

The table is printed to stdout and written in the requested formats; the
`.C` macro has arrays of the borders and results and `GetCentMult(mult)`,
which returns the middle of the class of `mult` in percent (the most central
class is open above).

`mult_plot` is a PNG picture of the result in the style of the ROOT
canvases of the Γ-fit framework: on a logarithmic scale the data (blue
points with errors) and the fit (red squares, bins with at least one expected
count), with a vertical line at each class border, up to the fit, and the
class names between them (where there is room); the header holds the
`title`, the kernel, mean and variance as formulas, the fitted parameters
(the pile-up probability `p` in percent) and the fit statistic per degree of
freedom; with a normalization other than `Data` the trigger efficiency is
shown in red. Below, data / fit with lines at 1, 0.9 and 1.1. The picture is
drawn in pure Rust ([plotters](https://crates.io/crates/plotters)) with the
Liberation Sans font embedded in the binary
([`data/fonts`](data/fonts), SIL Open Font License). Its options, all
optional (`mult_plot: None` writes no picture):

```ron
mult_plot: (
    file: "centrality_result.png",   // in out_dir
    title: "BM@N Run8, Xe+Cs(I)",    // beginning of the header
    x_title: "N_ch",                 // default: x title of the data histogram, or its name
    y_title: "dN",
    ratio_title: "data/fit",         // y title of the ratio
    class_borders: true,             // vertical lines and names of the classes
    x_range: (0, 300),               // default: up to the last bin with data or fit
    y_range: (0.4, 1e6),             // logarithmic; default: from 0.4, with room for the header
    ratio_range: (0.45, 1.55),       // default: 0.45-1.55, lower if the ratio is
),
```

The titles are plain text: Greek letters and superscripts such as `χ²`
render (Liberation Sans has no subscript characters), while the ROOT `#chi`
and `_{...}` markup is not interpreted.

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

### Trigger efficiency

An experimental trigger loses events with a low value of the observable, so
the recorded distribution falls below `P(n)` in the most peripheral events
and the histogram holds only a fraction `ε` of all inelastic events. The
centrality percents must refer to all events (the total inelastic cross
section), so `N`, and with it `ε`, has to be known:

- With `normalization: Data`, `N` is the number of recorded events, which
  assumes `ε = 1`. With an inefficient trigger the fit then distorts
  `n̄(c_b)` to absorb the lost events, **without any sign in its quality**:
  on a toy with `ε = 0.838` (efficiency `1/(1 + exp(-(n - 6)/1.5))`, `fit_min:
  20`) the fit has `-2 ln λ/ndf = 0.56`, but `c_knee` is 0.85% instead of
  0.70% and all borders are shifted (e.g. 0-10% starts at 158 instead of 149).
- With `normalization: Efficiency(ε)` and a fit range where the trigger is
  fully efficient, the classes are those of all events: on the same toy with
  the true `ε`, the borders are those of the fully efficient toy, and the
  efficiency of each class (data over fit) agrees with the true one (0.406 vs
  0.410 in 80-90%, 0.075 vs 0.073 in 90-100%).
- With `normalization: Free`, `N` comes from the extrapolation of the fitted
  `P(n)` below the fit range, so it is only as good as the shape of `n̄(c_b)`
  there, which the data in the fit range do not constrain. On toys with this
  efficiency and `fit_min` from 12 to 50:

  | toy | fitted mean | `N` (true 10⁶) | `-2 ln λ/ndf` |
  |---|---|---|---|
  | two-component | `Glauber` | -0.9% to +0.5% | 0.37-0.43 |
  | two-component | `ExpPoly(3)` | -31% to -19% | 0.38-0.46 |
  | `ExpPoly(3)` | `ExpPoly(3)` | -12% to -5% (once +1%: `ε > 1`) | 0.66 |
  | `ExpPoly(3)` | `Glauber` | +25% to +34% | 1.9-9.5 |

  So `Free` needs a mean model that describes the peripheral events, such as
  the `Glauber` mean when the data follow it (a wrong one shows in the fit
  quality), or the `Histogram` mean from a model; with `ExpPoly` or `Indra`
  the fit prints a warning. Check the
  stability of `N` against `fit_min` in any case.

The fit prints the trigger efficiency, data over fit, for all events and
below the fit range, e.g.
`trigger efficiency (data / fit): 0.8377 of all events, 0.5878 below the fit
range (n < 20)`, and writes them to `FitResult`; `fit_ratio` in the fit file is
data over fit in each bin, the efficiency versus `n` below the fit range.
`define-centrality` gives the efficiency of each class (`Efficiency` in
`Result` and the tables), and `DataPercent` is the data in a class over all
events `N`, so in efficient classes it agrees with the class width. It warns if
an extrapolation finds fewer events than recorded (`ε > 1`).

`P(c_b|class)`, `P(b|class)` and their means describe the **recorded**
events of a class. In a partially efficient class these are biased to larger
`n`, hence smaller `b`, so each bin of the class is weighted with its trigger
efficiency `ε_i`:

```
P(c_b | class, recorded) ∝ Σ_{bins i in class} ε_i P(n ∈ bin i | c_b)
```

`ε_i` is 1 from the fit range on (the trigger must be fully efficient
there) and data over fit below it, made non-decreasing with `n` (weighted
pool-adjacent-violators, with the expected counts as weights, which also
smooths the statistical fluctuations) and limited to `[0, 1]`; it is written
as `Trigger_Efficiency` to `FINAL.root`. Fully efficient classes are
unchanged. On the toy above (`Efficiency` with the true `ε`), `<b>` of the
recorded events is reproduced: 12.789 vs 12.782 fm in 70-80%, 13.521 vs
13.513 fm in 80-90%, 14.152 vs 14.140 fm in 90-100% (without the weights,
13.608 and 14.223 fm, those of all events). The estimated efficiency follows
the true turn-on within about 0.02 (e.g. 0.493 vs 0.500 at `n = 6`, 0.790 vs
0.791 at `n = 8`). The class percents and borders still refer to all events
(`FitMinPercent`/`FitMaxPercent`), and the error propagation keeps `ε_i`
fixed.

The same `ε_i` give the trigger efficiency versus centrality and impact
parameter (of single collisions),

```
ε(c_b) = Σ_i ε_i P(n ∈ bin i | c_b)
```

averaged over the events of each bin of `c_b` (`Trigger_Efficiency_VS_Cb`)
and of `b` (`Trigger_Efficiency_VS_B`, over `[c_b(b₀), c_b(b₁)]`, i.e.
weighted with `P(b)`); its average over `c_b` is the efficiency of all
events. On the toy above the efficiency of each bin of `b` agrees with the
recorded fraction of the toy events in it within 0.027 (132 bins of 0.074 fm
with at least 1000 events): it is 1 up to `b ≈ 11.5` fm and falls to 0.71 at
13 fm and 0.13 at 14.5 fm.

#### The `Indra` mean and the trigger

Frankland et al. attribute the offset `x_min` of the `Indra` mean
(their `k_min`) to the trigger: the mean does not vanish for the most
peripheral **recorded** events. In their approach the centrality and `P(b)`
refer to the recorded events (`P(b)` of all detected events), not to the
total inelastic cross section, and the trigger efficiency is not
reconstructed. A toy tests both uses: 10⁶ Au+Au events of TGlauberMC
(√s_NN = 27.5 GeV), `n` from a Gamma kernel around `0.9 N_part + 0.1 N_coll`
and a measured trigger turn-on (efficient from `n ≈ 14`, `ε = 0.807`), with
`b`, `n` and the trigger decision of each event as truth.

- Percents of **all** events (`normalization: Free`, `fit_min: 15`, `P(b)` of
  all events): `Indra` does not recover the lost events.

  | mean | `N` (true 10⁶) | `-2 ln λ/ndf` | true width of the 10% classes | `<b>` fit - true |
  |---|---|---|---|---|
  | `Glauber` (optical) | +4.6% | 0.37 | 10.2-10.6% | -0.08 to -0.35 fm |
  | `Indra` | +12.8% | 0.43 | 10.9-13.9% | -0.20 to -1.07 fm |
  | `ExpPoly(3)` | -19% | 0.37 | 7.6-8.6% | +0.32 to +1.56 fm |

  The fit quality is the same for all three. The `Glauber` mean is the
  best, but not exact here, as the optical geometry differs from the Monte
  Carlo one of the data.
- Percents of the **recorded** events, as Frankland et al.
  (`normalization: Data`, the whole distribution fitted, `P(b)` of the
  recorded events): `Indra` (`-2 ln λ/ndf = 8.2`) reproduces the class widths
  within 9.4-11.1% and `<b>` within 0.09 fm, as does `ExpPoly(3)` (2.3;
  8.9-10.8%, 0.07 fm), while the `Glauber` mean, whose shape is fixed by the
  geometry, cannot follow the turn-on (409; up to 1.4 fm). With `P(b)` of all
  events instead, the same `Indra` fit gives `<b>` 0.3-1.4 fm too large.

So it is this definition of the centrality, together with `P(b)` of the
recorded events (which needs a simulation of the trigger), that handles the
trigger, not the `Indra` form. Such percents are fractions of the recorded
events (the "0-10%" class of the toy is 8.5% of all events). For percents of
the total cross section use `Efficiency(ε)`, or `Free` with a `Glauber`,
`Histogram` or `Tree` mean.

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
