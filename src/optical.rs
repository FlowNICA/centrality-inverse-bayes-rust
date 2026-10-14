//! Impact parameter distribution of inelastic collisions from the nuclear
//! density profiles, in the optical Glauber model (Miller et al., Ann. Rev.
//! Nucl. Part. Sci. 57 (2007) 205):
//!
//! ```text
//! T_A(s)   = ∫ ρ_A(√(s² + z²)) dz                     (∫ T_A d²s = 1)
//! T_AB(b)  = ∫ T_A(s) T_B(|s - b|) d²s                 (∫ T_AB d²b = 1)
//! P_inel(b) = 1 - (1 - σ_NN T_AB(b))^{AB}
//! P(b)      = 2πb P_inel(b) / σ_inel,   σ_inel = ∫ 2πb P_inel(b) db
//! ```
//!
//! The nuclei and their density profiles are those of
//! [oxy-glauber](https://github.com/FlowNICA/oxy-glauber) (`data/nuclei.ron`,
//! embedded at compile time). For deformed nuclei the density is averaged over
//! the orientations; the reweighted (`*rw`) profiles use their physical
//! density, without the reweighting polynomial that compensates the minimum
//! distance between sampled nucleons in the Monte Carlo.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::Path;

use rayon::prelude::*;
use serde::Deserialize;

use crate::config_file::expand_home;
use crate::error::{Error, Result};
use crate::impact::ImpactProfile;
use crate::numeric::{composite_gauss_legendre, gauss_legendre};

/// The nuclei of oxy-glauber.
pub const NUCLEI_RON: &str = include_str!("../data/nuclei.ron");

/// Points of the tabulated radial density, thickness function and `T_AB`.
const N_RADIUS: usize = 4000;
const N_THICKNESS: usize = 800;
const N_IMPACT: usize = 1000;
/// 1 mb = 0.1 fm².
const FM2_PER_MB: f64 = 0.1;

/// Radial density profiles of oxy-glauber (`DensityProfile`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum DensityProfile {
    ProtonExp,
    WoodsSaxon3PF,
    WoodsSaxon3PG,
    Hulthen,
    HulthenConstrained,
    Ellipsoid,
    FromFile,
    DeformedBox,
    DeformedTF2,
    ProtonGaussian,
    ProtonDGaussian,
    ProtonNeutron3PF,
    Reweighted,
    ProtonNeutronReweighted,
    DeformedReweighted,
    HarmonicOscillator,
    Oxygen1970,
    FromGraph,
    HardSphere,
}

/// One entry of `nuclei.ron`, with the defaults of oxy-glauber.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NucleusParams {
    pub n: i32,
    pub z: i32,
    pub profile: DensityProfile,
    #[serde(default)]
    pub r: f64,
    #[serde(default)]
    pub a: f64,
    #[serde(default)]
    pub w: f64,
    #[serde(default)]
    pub r2: f64,
    #[serde(default)]
    pub a2: f64,
    #[serde(default)]
    pub w2: f64,
    #[serde(default)]
    pub beta2: f64,
    #[serde(default)]
    pub beta3: f64,
    #[serde(default)]
    pub beta4: f64,
    #[serde(default)]
    pub gamma: f64,
    #[serde(default = "default_max_r")]
    pub max_r: f64,
    #[serde(default = "default_recenter")]
    pub recenter: i32,
    #[serde(default = "default_smax")]
    pub smax: f64,
    #[serde(default)]
    pub r0: f64,
    #[serde(default)]
    pub r1: f64,
    #[serde(default)]
    pub r2_factor: f64,
    #[serde(default)]
    pub file: Option<String>,
}

fn default_max_r() -> f64 {
    15.0
}
fn default_recenter() -> i32 {
    1
}
fn default_smax() -> f64 {
    99.0
}

/// Reads the nuclei from `path`, or the embedded oxy-glauber table if `None`.
pub fn load_nuclei(path: Option<&Path>) -> Result<BTreeMap<String, NucleusParams>> {
    let (text, source) = match path {
        Some(p) => {
            let p = expand_home(p);
            let text = std::fs::read_to_string(&p)
                .map_err(|e| Error::Config(format!("cannot read {}: {e}", p.display())))?;
            (text, p.display().to_string())
        }
        None => (NUCLEI_RON.to_string(), "the embedded nuclei.ron".to_string()),
    };
    ron::Options::default()
        .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
        .from_str(&text)
        .map_err(|e| Error::Config(format!("{source}: {e}")))
}

/// Woods-Saxon 3-parameter Fermi: `(1 + w (r/R)²) / (1 + exp((r - R)/a))`.
fn ws_3pf(r: f64, rr: f64, a: f64, w: f64) -> f64 {
    (1. + w * (r / rr).powi(2)) / (1. + ((r - rr) / a).exp())
}

impl NucleusParams {
    /// Mass number.
    pub fn mass(&self) -> f64 {
        self.n as f64
    }

    /// Outer radius of the density: `max_r` (the range in which oxy-glauber
    /// samples the nucleons), halved for the deuteron, whose profile gives the
    /// separation of its two nucleons.
    pub fn r_max(&self) -> f64 {
        match self.profile {
            DensityProfile::Hulthen | DensityProfile::HulthenConstrained => 0.5 * self.max_r,
            _ => self.max_r,
        }
    }

    /// Radial density of protons (`proton`) or neutrons, unnormalized,
    /// averaged over the orientations of deformed nuclei.
    fn species_density(&self, r: f64, proton: bool) -> f64 {
        use DensityProfile::*;
        let (rr, a, w) = if proton || !matches!(self.profile, ProtonNeutron3PF | ProtonNeutronReweighted) {
            (self.r, self.a, self.w)
        } else {
            (self.r2, self.a2, self.w2)
        };
        match self.profile {
            ProtonExp => (-r / rr).exp(),
            ProtonGaussian => (-(r * r) / (2. * rr * rr)).exp(),
            ProtonDGaussian => {
                let (p0, s2) = (0.5, 0.4 * rr);
                (1. - p0) / rr.powi(3) * (-(r / rr).powi(2)).exp()
                    + p0 / s2.powi(3) * (-(r / s2).powi(2)).exp()
            }
            Hulthen | HulthenConstrained => {
                /* the profile is that of the separation 2r */
                let d = 2. * r;
                if d <= 0. {
                    return hulthen(1e-9, rr, a);
                }
                hulthen(d, rr, a)
            }
            WoodsSaxon3PF | Reweighted | ProtonNeutron3PF | ProtonNeutronReweighted => {
                ws_3pf(r, rr, a, w)
            }
            WoodsSaxon3PG => {
                (1. + w * (r / rr).powi(2)) / (1. + ((r * r - rr * rr) / (a * a)).exp())
            }
            HarmonicOscillator => {
                let x = r / a;
                (1. + rr * x * x) * (-x * x).exp()
            }
            Oxygen1970 => {
                let osc = if r.abs() < 1e-12 {
                    0.102
                } else {
                    0.102 * (2.76 * r).sin() / (2.76 * r) * (-(0.35 * r).powi(2)).exp()
                };
                (1. - osc + w * (r / rr).powi(2)) / (1. + ((r - rr) / a).exp())
            }
            HardSphere => {
                if r <= rr {
                    1.
                } else {
                    0.
                }
            }
            Ellipsoid | DeformedBox | DeformedTF2 | DeformedReweighted => self.deformed_average(r),
            FromFile | FromGraph => 0.,
        }
    }

    /// `½ ∫ ρ(r, θ) sin θ dθ` for the deformed profiles.
    fn deformed_average(&self, r: f64) -> f64 {
        let (x, wt) = gauss_legendre(64);
        let density = |c: f64| -> f64 {
            let (c2, c3, c4) = (c * c, c * c * c, c * c * c * c);
            match self.profile {
                DensityProfile::Ellipsoid => {
                    let rt = self.r + self.beta2 * c2;
                    ws_3pf_relative(r, rt, self.a, self.w)
                }
                DensityProfile::DeformedBox => {
                    let rt = self.r
                        * (1.
                            + self.beta2 * 0.315 * (3. * c2 - 1.)
                            + self.beta3 * 0.373176 * (5. * c3 - 3. * c)
                            + self.beta4 * 0.105 * (35. * c4 - 30. * c2 + 3.));
                    ws_3pf_relative(r, rt, self.a, self.w)
                }
                _ => {
                    let rt = self.r
                        * (1.
                            + self.beta2 * 0.315 * (3. * c2 - 1.)
                            + self.beta4 * 0.105 * (35. * c4 - 30. * c2 + 3.));
                    1. / (1. + ((r - rt) / self.a).exp())
                }
            }
        };
        0.5 * x.iter().zip(&wt).map(|(&c, &w)| w * density(c)).sum::<f64>()
    }
}

/// Woods-Saxon with the `w` term relative to the deformed radius.
fn ws_3pf_relative(r: f64, rt: f64, a: f64, w: f64) -> f64 {
    (1. + w * (r / rt).powi(2)) / (1. + ((r - rt) / a).exp())
}

/// Hulthén density of the deuteron (unnormalized).
fn hulthen(d: f64, rr: f64, a: f64) -> f64 {
    let diff = (-rr * d).exp() - (-a * d).exp();
    (diff / d).powi(2)
}

/// A nucleus with its tabulated, normalized thickness function.
#[derive(Debug, Clone)]
pub struct Nucleus {
    pub name: String,
    pub a: f64,
    pub z: f64,
    /// `T(s)` on `s_k = k · ds`, `∫ 2πs T(s) ds = 1`.
    thickness: Vec<f64>,
    ds: f64,
}

impl Nucleus {
    /// Builds the nucleus `name` from the table.
    pub fn new(name: &str, nuclei: &BTreeMap<String, NucleusParams>) -> Result<Self> {
        let params = nuclei.get(name).ok_or_else(|| {
            Error::Config(format!(
                "unknown nucleus '{name}', expected one of: {}",
                nuclei.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })?;
        Self::from_params(name, params)
    }

    pub fn from_params(name: &str, params: &NucleusParams) -> Result<Self> {
        if matches!(params.profile, DensityProfile::FromFile | DensityProfile::FromGraph) {
            return Err(Error::Config(format!(
                "nucleus '{name}': profile {:?} has no analytic density for the optical Glauber model",
                params.profile
            )));
        }
        if params.n < 1 || params.z < 0 || params.z > params.n || !(params.r_max() > 0.) {
            return Err(Error::Config(format!("nucleus '{name}': invalid parameters")));
        }
        let r_max = params.r_max();
        let a = params.mass();
        let z = params.z as f64;

        /* radial density, with protons and neutrons normalized separately
         * when they have different profiles */
        let dr = r_max / N_RADIUS as f64;
        let radii: Vec<f64> = (0..=N_RADIUS).map(|i| i as f64 * dr).collect();
        let normalized = |proton: bool| -> Vec<f64> {
            let rho: Vec<f64> = radii.iter().map(|&r| params.species_density(r, proton)).collect();
            let norm = 4. * PI * trapezoid(&radii.iter().zip(&rho).map(|(r, v)| r * r * v).collect::<Vec<_>>(), dr);
            rho.iter().map(|v| v / norm).collect()
        };
        let density = if matches!(
            params.profile,
            DensityProfile::ProtonNeutron3PF | DensityProfile::ProtonNeutronReweighted
        ) {
            let (p, n) = (normalized(true), normalized(false));
            p.iter().zip(&n).map(|(p, n)| (z * p + (a - z) * n) / a).collect()
        } else {
            normalized(true)
        };
        if !density.iter().all(|v| v.is_finite() && *v >= 0.) {
            return Err(Error::Config(format!("nucleus '{name}': invalid density")));
        }
        let rho = |r: f64| -> f64 {
            let x = r / dr;
            if x >= N_RADIUS as f64 {
                return 0.;
            }
            let i = x.floor() as usize;
            let f = x - i as f64;
            density[i] * (1. - f) + density[(i + 1).min(N_RADIUS)] * f
        };

        /* thickness T(s) = 2 ∫₀^√(R²-s²) ρ(√(s² + z²)) dz */
        let ds = r_max / N_THICKNESS as f64;
        let thickness: Vec<f64> = (0..=N_THICKNESS)
            .into_par_iter()
            .map(|k| {
                let s = k as f64 * ds;
                let z_max = (r_max * r_max - s * s).max(0.).sqrt();
                if z_max <= 0. {
                    return 0.;
                }
                let (zs, ws) = composite_gauss_legendre(0., z_max, 200, 4);
                2. * zs
                    .iter()
                    .zip(&ws)
                    .map(|(&z, &w)| w * rho((s * s + z * z).sqrt()))
                    .sum::<f64>()
            })
            .collect();
        Ok(Nucleus {
            name: name.to_string(),
            a,
            z,
            thickness,
            ds,
        })
    }

    /// Outer radius of the thickness function.
    pub fn r_max(&self) -> f64 {
        self.ds * N_THICKNESS as f64
    }

    /// `T(s)`, linearly interpolated, 0 outside.
    pub fn thickness(&self, s: f64) -> f64 {
        let x = s / self.ds;
        if !(x < N_THICKNESS as f64) {
            return 0.;
        }
        let i = x.floor() as usize;
        let f = x - i as f64;
        self.thickness[i] * (1. - f) + self.thickness[i + 1] * f
    }
}

/// `T_AB(b)` on `b_k = k · db` for `b` up to the sum of the radii.
pub fn overlap(a: &Nucleus, b: &Nucleus) -> (Vec<f64>, f64) {
    let b_max = a.r_max() + b.r_max();
    let db = b_max / N_IMPACT as f64;
    let (ss, ws) = composite_gauss_legendre(0., a.r_max(), 200, 4);
    let (phis, wphi) = composite_gauss_legendre(0., PI, 32, 4);
    let t_ab = (0..=N_IMPACT)
        .into_par_iter()
        .map(|k| {
            let bb = k as f64 * db;
            ss.iter()
                .zip(&ws)
                .map(|(&s, &w)| {
                    let ta = a.thickness(s);
                    if ta == 0. {
                        return 0.;
                    }
                    let angular: f64 = phis
                        .iter()
                        .zip(&wphi)
                        .map(|(&phi, &wp)| {
                            let d2 = s * s + bb * bb - 2. * s * bb * phi.cos();
                            wp * b.thickness(d2.max(0.).sqrt())
                        })
                        .sum();
                    /* the angular integral over [0, 2π] is twice that over [0, π] */
                    w * s * ta * 2. * angular
                })
                .sum::<f64>()
        })
        .collect();
    (t_ab, db)
}

/// Inelastic nucleon-nucleon cross section (mb) at `√s_NN` (GeV) from the
/// parametrization of Loizides et al., arXiv:2011.14909 (as in oxy-glauber),
/// valid from 10 GeV.
pub fn sigma_nn_from_energy(sqrt_snn: f64) -> f64 {
    let log_s = (sqrt_snn * sqrt_snn).ln();
    28.84374 + 0.04584121 * log_s.powf(2.374257)
}

/// Settings of the optical Glauber model.
#[derive(Debug, Clone)]
pub struct OpticalGlauber {
    pub nucleus_a: Nucleus,
    pub nucleus_b: Nucleus,
    /// Effective inelastic nucleon-nucleon cross section (mb).
    pub sigma_nn: f64,
}

impl OpticalGlauber {
    /// The nuclei `a` and `b` with `σ_NN` (mb) given, or from `√s_NN`; with
    /// `sigma_np`, `sigma_nn` is the pp (and nn) cross section and the
    /// effective one is averaged over the nucleon pairs.
    pub fn new(
        a: &str,
        b: &str,
        sigma_nn: Option<f64>,
        sigma_np: Option<f64>,
        sqrt_snn: Option<f64>,
        nuclei_file: Option<&Path>,
    ) -> Result<Self> {
        let nuclei = load_nuclei(nuclei_file)?;
        let nucleus_a = Nucleus::new(a, &nuclei)?;
        let nucleus_b = Nucleus::new(b, &nuclei)?;
        let sigma_pp = match (sigma_nn, sqrt_snn) {
            (Some(s), None) => s,
            (None, Some(e)) => sigma_nn_from_energy(e),
            _ => {
                return Err(Error::Config(
                    "Glauber impact parameter needs exactly one of sigma_nn and sqrt_snn".into(),
                ));
            }
        };
        let sigma = match sigma_np {
            Some(np) => {
                let (za, zb) = (nucleus_a.z, nucleus_b.z);
                let (na, nb) = (nucleus_a.a - za, nucleus_b.a - zb);
                ((za * zb + na * nb) * sigma_pp + (za * nb + na * zb) * np)
                    / (nucleus_a.a * nucleus_b.a)
            }
            None => sigma_pp,
        };
        Ok(OpticalGlauber {
            nucleus_a,
            nucleus_b,
            sigma_nn: sigma,
        })
    }

    /// `P_inel(b)` on the grid of `T_AB`, and its step.
    pub fn inelastic_probability(&self) -> (Vec<f64>, f64) {
        let (t_ab, db) = overlap(&self.nucleus_a, &self.nucleus_b);
        let sigma = self.sigma_nn * FM2_PER_MB;
        let ab = self.nucleus_a.a * self.nucleus_b.a;
        let p = t_ab
            .iter()
            .map(|&t| {
                let x = sigma * t;
                if x >= 1. {
                    1.
                } else {
                    -(ab * (-x).ln_1p()).exp_m1()
                }
            })
            .collect();
        (p, db)
    }

    /// The impact parameter profile `c_b(b)` and `σ_inel`.
    pub fn profile(&self) -> ImpactProfile {
        let (p_inel, db) = self.inelastic_probability();
        let dsigma: Vec<f64> = p_inel
            .iter()
            .enumerate()
            .map(|(k, p)| 2. * PI * k as f64 * db * p)
            .collect();
        let mut cum = vec![0.; dsigma.len()];
        for k in 1..dsigma.len() {
            cum[k] = cum[k - 1] + 0.5 * db * (dsigma[k - 1] + dsigma[k]);
        }
        let sigma_inel = cum[cum.len() - 1];
        let b: Vec<f64> = (0..dsigma.len()).map(|k| k as f64 * db).collect();
        let cb: Vec<f64> = cum.iter().map(|c| c / sigma_inel).collect();
        ImpactProfile::Table {
            b,
            cb,
            sigma_inel,
            description: format!(
                "optical Glauber P(b) for {}+{}, sigma_NN = {:.3} mb, sigma_inel = {:.2} fm^2 ({:.4} b)",
                self.nucleus_a.name,
                self.nucleus_b.name,
                self.sigma_nn,
                sigma_inel,
                sigma_inel / 100.
            ),
        }
    }
}

fn trapezoid(values: &[f64], step: f64) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.;
    }
    step * (values.iter().sum::<f64>() - 0.5 * (values[0] + values[n - 1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn all_analytic_nuclei_are_normalized() {
        let nuclei = load_nuclei(None).unwrap();
        assert!(nuclei.len() > 100);
        for (name, params) in &nuclei {
            if matches!(params.profile, DensityProfile::FromFile | DensityProfile::FromGraph) {
                assert!(Nucleus::from_params(name, params).is_err());
                continue;
            }
            let nucleus = Nucleus::from_params(name, params).unwrap();
            let (s, w) = composite_gauss_legendre(0., nucleus.r_max(), 400, 4);
            let norm: f64 = s
                .iter()
                .zip(&w)
                .map(|(&s, &w)| w * 2. * PI * s * nucleus.thickness(s))
                .sum();
            assert!(close(norm, 1., 2e-3), "{name}: {norm}");
        }
    }

    #[test]
    fn overlap_is_normalized() {
        let nuclei = load_nuclei(None).unwrap();
        for (a, b) in [("Au", "Au"), ("Pb", "Pb"), ("p", "Pb"), ("d", "Au"), ("U", "U")] {
            let (na, nb) = (Nucleus::new(a, &nuclei).unwrap(), Nucleus::new(b, &nuclei).unwrap());
            let (t, db) = overlap(&na, &nb);
            let norm = trapezoid(
                &t.iter()
                    .enumerate()
                    .map(|(k, t)| 2. * PI * k as f64 * db * t)
                    .collect::<Vec<_>>(),
                db,
            );
            assert!(close(norm, 1., 3e-3), "{a}+{b}: {norm}");
        }
    }

    #[test]
    fn black_hard_spheres_give_the_geometric_cross_section() {
        /* with a huge σ_NN, every overlapping pair interacts: σ = π (R_A + R_B)² */
        let glauber = OpticalGlauber::new("IHS", "Xe124HS", Some(1e6), None, None, None).unwrap();
        let profile = glauber.profile();
        let expected = PI * (5.66f64 + 5.431).powi(2);
        let sigma = profile.sigma_inel().unwrap();
        assert!(close(sigma, expected, 2e-3), "{sigma} vs {expected}");
    }

    #[test]
    fn heavy_ion_cross_sections() {
        /* Au+Au at √s = 200 GeV (σ_NN ≈ 42 mb) and Pb+Pb at 5.02 TeV (≈ 68 mb):
         * the optical model gives about 7 b and 7.7 b, a few percent above
         * the Monte Carlo values (6.9-7.0 b and 7.6-7.7 b) */
        let au = OpticalGlauber::new("Au", "Au", Some(42.), None, None, None).unwrap();
        let sigma = au.profile().sigma_inel().unwrap() / 100.;
        assert!((6.8..7.5).contains(&sigma), "Au+Au: {sigma} b");
        let pb = OpticalGlauber::new("Pb", "Pb", None, None, Some(5020.), None).unwrap();
        assert!(close(pb.sigma_nn, 67.6, 0.01), "{}", pb.sigma_nn);
        let sigma = pb.profile().sigma_inel().unwrap() / 100.;
        assert!((7.5..8.2).contains(&sigma), "Pb+Pb: {sigma} b");
        /* c_b(b) rises from 0 to 1, P_inel ≈ 1 in central collisions */
        let profile = au.profile();
        assert!(profile.cb_of_b(0.) == 0. && profile.cb_of_b(30.) == 1.);
        let (p_inel, _) = au.inelastic_probability();
        assert!(p_inel[0] > 0.999999);
    }

    #[test]
    fn isospin_averaged_cross_section() {
        let g = OpticalGlauber::new("Au", "Au", Some(20.), Some(30.), None, None).unwrap();
        let (z, n) = (79., 118.);
        let expected = ((z * z + n * n) * 20. + 2. * z * n * 30.) / (197. * 197.);
        assert!(close(g.sigma_nn, expected, 1e-12));
        assert!(OpticalGlauber::new("Au", "Au", None, None, None, None).is_err());
        assert!(OpticalGlauber::new("Nope", "Au", Some(30.), None, None, None).is_err());
    }
}
