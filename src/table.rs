//! Tables of the centrality classes: plain text, CSV, LaTeX and a ROOT macro.

use std::fmt::Write as _;

use crate::centrality::{CentralityClass, CentralityResult, integer_borders, percent};
use crate::config::TableFormat;

/// A column: header, LaTeX header, value.
type Column = (
    &'static str,
    &'static str,
    Box<dyn Fn(&CentralityClass) -> String>,
);

/// The table in `format`.
pub fn table(result: &CentralityResult, format: TableFormat) -> String {
    match format {
        TableFormat::Csv => csv_table(result),
        TableFormat::Tex => tex_table(result),
        TableFormat::Cpp => cpp_table(result),
    }
}

fn border(v: f64, integer: bool) -> String {
    if integer {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn num(v: f64, digits: usize) -> String {
    if v.is_finite() {
        format!("{v:.digits$}")
    } else {
        "nan".into()
    }
}

fn columns(result: &CentralityResult) -> Vec<Column> {
    let integer = integer_borders(&result.classes);
    let mut cols: Vec<Column> = vec![
        (
            "Centrality",
            "Centrality",
            Box::new(|c| c.label().replace('%', "")),
        ),
        (
            "MinBorder",
            "$n_{\\min}$",
            Box::new(move |c| border(c.min_border, integer)),
        ),
        (
            "MaxBorder",
            "$n_{\\max}$",
            Box::new(move |c| border(c.max_border, integer)),
        ),
        (
            "FitPercent",
            "Fit, \\%",
            Box::new(|c| {
                format!(
                    "{}-{}",
                    num(c.fit_min_percent, 2),
                    num(c.fit_max_percent, 2)
                )
            }),
        ),
        (
            "DataPercent",
            "Data, \\%",
            Box::new(|c| num(c.data_percent, 2)),
        ),
        (
            "<cb>",
            "$\\langle c_b\\rangle$",
            Box::new(|c| num(c.cb.mean, 4)),
        ),
        ("RMS(cb)", "RMS($c_b$)", Box::new(|c| num(c.cb.rms, 4))),
    ];
    if result.pile_up_probability > 0. {
        cols.insert(
            5,
            (
                "PileUp%",
                "Pile-up, \\%",
                Box::new(|c| num(c.pile_up_percent, 3)),
            ),
        );
    }
    let with_errors = result.classes.iter().all(|c| c.errors.is_some());
    if with_errors {
        cols.push((
            "Err<cb>",
            "$\\delta\\langle c_b\\rangle$",
            Box::new(|c| num(c.errors.map_or(f64::NAN, |e| e.cb_mean), 4)),
        ));
    }
    if result.impact.is_some() {
        let b = |f: fn(&crate::centrality::Stats) -> f64| {
            move |c: &CentralityClass| num(c.b.as_ref().map_or(f64::NAN, f), 3)
        };
        cols.push(("<b>", "$\\langle b\\rangle$, fm", Box::new(b(|s| s.mean))));
        if with_errors {
            cols.push((
                "Err<b>",
                "$\\delta\\langle b\\rangle$, fm",
                Box::new(|c| num(c.errors.map_or(f64::NAN, |e| e.b_mean), 3)),
            ));
        }
        cols.push(("RMS(b)", "RMS($b$), fm", Box::new(b(|s| s.rms))));
        cols.push(("b_min", "$b_{\\min}$, fm", Box::new(b(|s| s.min))));
        cols.push(("b_max", "$b_{\\max}$, fm", Box::new(b(|s| s.max))));
    }
    if with_errors {
        cols.push((
            "ErrMinBorder",
            "$\\delta n_{\\min}$",
            Box::new(|c| num(c.errors.map_or(f64::NAN, |e| e.min_border), 3)),
        ));
    }
    cols
}

/// Aligned plain text table, printed to stdout.
pub fn plain_table(result: &CentralityResult) -> String {
    let cols = columns(result);
    let rows: Vec<Vec<String>> = result
        .classes
        .iter()
        .map(|c| cols.iter().map(|(_, _, f)| f(c)).collect())
        .collect();
    let widths: Vec<usize> = cols
        .iter()
        .enumerate()
        .map(|(i, (h, _, _))| {
            rows.iter()
                .map(|r| r[i].len())
                .max()
                .unwrap_or(0)
                .max(h.len())
        })
        .collect();
    let mut s = String::new();
    let line = |s: &mut String, cells: Vec<&str>| {
        let cells: Vec<String> = cells
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!("{c:>w$}"))
            .collect();
        let _ = writeln!(s, "{}", cells.join("  "));
    };
    line(&mut s, cols.iter().map(|(h, _, _)| *h).collect());
    for r in &rows {
        line(&mut s, r.iter().map(String::as_str).collect());
    }
    let _ = writeln!(
        s,
        "n_knee = {:.4}, sigma(n_knee) = {:.4}, c_knee = {:.4}%",
        result.n_knee,
        result.sigma_knee,
        100. * result.c_knee
    );
    if result.pile_up_probability > 0. {
        let _ = writeln!(
            s,
            "pile-up probability p = {:.5}; tail not classified: {}",
            result.pile_up_probability,
            result
                .pile_up_tail
                .map_or("none".to_string(), |x| format!("n >= {x}"))
        );
    }
    if let Some(ip) = &result.impact {
        let _ = writeln!(s, "impact parameter: {ip}");
    }
    if result.n_error_samples > 0 {
        let _ = writeln!(
            s,
            "errors from {} parameter sets drawn from the fit covariance",
            result.n_error_samples
        );
    }
    s
}

pub fn csv_table(result: &CentralityResult) -> String {
    let cols = columns(result);
    let mut s = format!("# Generated from a file: {}\n", result.source.display());
    let _ = writeln!(
        s,
        "{}",
        cols.iter()
            .map(|(h, _, _)| *h)
            .collect::<Vec<_>>()
            .join(",")
    );
    for c in &result.classes {
        let _ = writeln!(
            s,
            "{}",
            cols.iter()
                .map(|(_, _, f)| f(c))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    s
}

pub fn tex_table(result: &CentralityResult) -> String {
    let cols = columns(result);
    let mut s = String::from(
        "\\documentclass{article}\n\\usepackage[landscape]{geometry}\n\\begin{document}\n\\begin{table}[h]\n\\centering\n",
    );
    let _ = writeln!(s, "\\begin{{tabular}}{{{}}}", "c".repeat(cols.len()));
    let _ = writeln!(s, "\\hline");
    let _ = writeln!(
        s,
        "{} \\\\",
        cols.iter()
            .map(|(_, t, _)| *t)
            .collect::<Vec<_>>()
            .join(" & ")
    );
    let _ = writeln!(s, "\\hline");
    for c in &result.classes {
        let _ = writeln!(
            s,
            "{} \\\\",
            cols.iter()
                .map(|(_, _, f)| f(c))
                .collect::<Vec<_>>()
                .join(" & ")
        );
    }
    let _ = writeln!(s, "\\hline\n\\end{{tabular}}");
    let _ = writeln!(
        s,
        "\\caption{{Centrality classes from the inverse Bayes fit ({}).}}",
        result.source.display().to_string().replace('_', "\\_")
    );
    s.push_str("\\end{table}\n\\end{document}\n");
    s
}

/// ROOT macro with the arrays of the table and `GetCentMult(mult)`.
pub fn cpp_table(result: &CentralityResult) -> String {
    let classes = &result.classes;
    let n = classes.len();
    let integer = integer_borders(classes);
    let border_type = if integer { "Int_t" } else { "Double_t" };
    let mut s = String::new();
    let _ = writeln!(s, "// Generated from a file: {}", result.source.display());
    let mut array = |ty: &str, name: &str, values: Vec<String>| {
        let _ = writeln!(s, "{ty} {name} [{n}] = {{ {} }};", values.join(", "));
    };
    let float = |v: f64| {
        if v.is_finite() {
            format!("{}", v as f32)
        } else {
            "NAN".into()
        }
    };
    let column =
        |get: &dyn Fn(&CentralityClass) -> f64| classes.iter().map(|c| float(get(c))).collect();
    array(
        "Float_t",
        "minCentPercent",
        classes.iter().map(|c| percent(c.min_percent)).collect(),
    );
    array(
        "Float_t",
        "maxCentPercent",
        classes.iter().map(|c| percent(c.max_percent)).collect(),
    );
    array(
        border_type,
        "minMult",
        classes
            .iter()
            .map(|c| border(c.min_border, integer))
            .collect(),
    );
    array(
        border_type,
        "maxMult",
        classes
            .iter()
            .map(|c| border(c.max_border, integer))
            .collect(),
    );
    array("Float_t", "meanCb", column(&|c| c.cb.mean));
    array("Float_t", "rmsCb", column(&|c| c.cb.rms));
    if result.impact.is_some() {
        let b = |f: fn(&crate::centrality::Stats) -> f64| {
            move |c: &CentralityClass| c.b.as_ref().map_or(f64::NAN, f)
        };
        array("Float_t", "meanB", column(&b(|s| s.mean)));
        array("Float_t", "rmsB", column(&b(|s| s.rms)));
        array("Float_t", "minB", column(&b(|s| s.min)));
        array("Float_t", "maxB", column(&b(|s| s.max)));
    }
    let _ = write!(
        s,
        "\nFloat_t GetCentMult({border_type} mult)\n\
         // Returns the centrality (middle of the class, in percent) for a given multiplicity, or -1;\n\
         // the most central class (i = 0) is open above\n\
         {{\n\
         \tInt_t centBin = -1;\n\
         \tfor (Int_t i = 0; i < {n}; i++)\n\
         \t{{\n\
         \t\tif (mult >= minMult[i] && (mult < maxMult[i] || (i == 0 && minCentPercent[0] == 0)))\n\
         \t\t\tcentBin = i;\n\
         \t}}\n\
         \tif (centBin == -1) return -1.;\n\
         \treturn (maxCentPercent[centBin] - minCentPercent[centBin]) / 2. + minCentPercent[centBin];\n\
         }}\n"
    );
    s
}
