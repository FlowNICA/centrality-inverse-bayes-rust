//! Functions of `c_b` given as `TF1`s in a ROOT file, for the `Idrisov` mean
//! and variance: the mean and variance of the observable versus `c_b` of a
//! model, as fitted in the framework of Idrisov et al. (e.g. `F_mM` and
//! `F_DM` of its "MeanVar" files).
//!
//! The functions are read with their formula and parameters and tabulated
//! once on a fine grid in `c_b`, so the fit does not evaluate formulas.

use std::path::PathBuf;

use oxiroot::prelude::*;

use crate::error::{Error, Result};

/// Intervals of the table in `c_b`.
const TABLE_INTERVALS: usize = 10_000;

/// A `TF1` of `c_b` in a ROOT file.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSpec {
    pub file: PathBuf,
    pub func: String,
}

impl FunctionSpec {
    /// RON form of the fields, `file: "f.root", func: "F_mM"`.
    pub fn ron_fields(&self) -> String {
        format!(
            "file: {:?}, func: {:?}",
            self.file.display().to_string(),
            self.func
        )
    }

    fn description(&self) -> String {
        format!("TF1 '{}' of {}", self.func, self.file.display())
    }

    /// Reads the function and tabulates it on `[0, 1]`.
    pub fn build(&self) -> Result<CbFunction> {
        let f = read_tf1(&self.file, &self.func)?;
        let values: Vec<f64> = (0..=TABLE_INTERVALS)
            .map(|i| f.eval(i as f64 / TABLE_INTERVALS as f64))
            .collect();
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            return Err(Error::Input(format!(
                "{}: not finite at c_b = {}",
                self.description(),
                i as f64 / TABLE_INTERVALS as f64
            )));
        }
        Ok(CbFunction {
            values,
            description: self.description(),
        })
    }
}

/// A function of `c_b` tabulated on a uniform grid of `[0, 1]`, linearly
/// interpolated.
#[derive(Debug, Clone, PartialEq)]
pub struct CbFunction {
    values: Vec<f64>,
    pub description: String,
}

impl CbFunction {
    pub fn at(&self, cb: f64) -> f64 {
        let x = cb.clamp(0., 1.) * TABLE_INTERVALS as f64;
        let i = (x.floor() as usize).min(TABLE_INTERVALS - 1);
        let t = x - i as f64;
        self.values[i] + t * (self.values[i + 1] - self.values[i])
    }
}

/// Reads the `TF1` `name`. Formulas that oxiroot cannot parse directly, such
/// as the `polN` of ROOT expanded with `TMath::Sq`, are rewritten from the
/// stored formula string and parameters.
fn read_tf1(path: &std::path::Path, name: &str) -> Result<Func1D> {
    let file = FileReader::open(path)
        .map_err(|e| Error::Input(format!("cannot open {}: {e}", path.display())))?;
    let direct = match Func1D::read_root(&file, name) {
        Ok(f) => return Ok(f),
        Err(e) => e,
    };
    let what = || format!("TF1 '{name}' of {}", path.display());
    let value = file
        .get_value(name)
        .map_err(|e| Error::Input(format!("cannot read {}: {e}", what())))?;
    let formula = value.get("fFormula");
    let text = formula
        .and_then(|f| f.get("fFormula"))
        .and_then(|f| f.as_str())
        .unwrap_or("");
    if text.trim().is_empty() {
        return Err(Error::Input(format!(
            "{}: no formula (a compiled C++ function?): {direct}",
            what()
        )));
    }
    let params: Vec<f64> = formula
        .and_then(|f| f.get("fClingParameters"))
        .and_then(|p| p.as_array())
        .map(|p| p.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();
    let range =
        |key: &str, default: f64| value.get(key).and_then(|v| v.as_f64()).unwrap_or(default);
    let rewritten = rewrite_tmath(text);
    let mut f = Func1D::new(name, &rewritten, range("fXmin", 0.), range("fXmax", 1.))
        .map_err(|e| Error::Input(format!("{}: cannot parse formula '{text}': {e}", what())))?;
    if f.npar() != params.len() {
        return Err(Error::Input(format!(
            "{}: {} parameters stored for a formula with {}",
            what(),
            params.len(),
            f.npar()
        )));
    }
    f.set_params(&params);
    Ok(f)
}

/// Replaces the `TMath` functions of ROOT formulas by the plain ones:
/// `TMath::Sq(e)` by `pow(e,2)`, and `TMath::Exp`, `Log`, `Sqrt`, `Power`,
/// `Abs`, `Sin`, `Cos` by `exp`, `log`, `sqrt`, `pow`, `abs`, `sin`, `cos`.
pub fn rewrite_tmath(formula: &str) -> String {
    let mut s = formula.to_string();
    for (from, to) in [
        ("TMath::Exp(", "exp("),
        ("TMath::Log(", "log("),
        ("TMath::Sqrt(", "sqrt("),
        ("TMath::Power(", "pow("),
        ("TMath::Abs(", "abs("),
        ("TMath::Sin(", "sin("),
        ("TMath::Cos(", "cos("),
    ] {
        s = s.replace(from, to);
    }
    const SQ: &str = "TMath::Sq(";
    while let Some(start) = s.find(SQ) {
        let open = start + SQ.len();
        /* the closing parenthesis of the argument */
        let mut depth = 1;
        let mut close = None;
        for (i, c) in s[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { break };
        s = format!(
            "{}pow({},2){}",
            &s[..start],
            &s[open..close],
            &s[close + 1..]
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmath_functions_are_rewritten() {
        assert_eq!(
            rewrite_tmath("[p0]+[p1]*x+[p2]*TMath::Sq(x)+[p3]*pow(x,3)"),
            "[p0]+[p1]*x+[p2]*pow(x,2)+[p3]*pow(x,3)"
        );
        assert_eq!(
            rewrite_tmath("TMath::Sq(TMath::Exp(-(x-1)))*TMath::Sq(x)"),
            "pow(exp(-(x-1)),2)*pow(x,2)"
        );
        let mut f = Func1D::new("f", &rewrite_tmath("[0]+[1]*TMath::Sq(x)"), 0., 1.).unwrap();
        f.set_params(&[1., 2.]);
        assert!((f.eval(0.5) - 1.5).abs() < 1e-14);
    }

    #[test]
    fn table_interpolates() {
        let g = CbFunction {
            values: (0..=TABLE_INTERVALS)
                .map(|i| 2. * i as f64 / TABLE_INTERVALS as f64)
                .collect(),
            description: String::new(),
        };
        assert!((g.at(0.123456) - 0.246912).abs() < 1e-12);
        assert_eq!(g.at(1.), 2.);
        assert_eq!(g.at(-1.), 0.);
    }
}
