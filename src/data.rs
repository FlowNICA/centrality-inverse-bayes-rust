//! The measured distribution of the centrality observable.

use std::path::Path;

use oxiroot::prelude::*;

use crate::error::{Error, Result};

/// A data histogram with its binning, contents and errors.
#[derive(Debug, Clone)]
pub struct DataHistogram {
    pub hist: Hist1D,
    /// The `nbins + 1` bin edges.
    pub edges: Vec<f64>,
    /// Contents of bins `1..=nbins`.
    pub counts: Vec<f64>,
    /// Statistical errors of bins `1..=nbins`.
    pub errors: Vec<f64>,
    /// All entries, including under- and overflow.
    pub total: f64,
    /// Whether the errors differ from `sqrt(content)` (weighted or scaled data).
    pub weighted: bool,
}

impl DataHistogram {
    /// Reads histogram `name` from the ROOT file at `path`.
    pub fn read(path: &Path, name: &str) -> Result<Self> {
        let file = FileReader::open(path)
            .map_err(|e| Error::Input(format!("cannot open {}: {e}", path.display())))?;
        let hist = Hist1D::read_root(&file, name).map_err(|e| {
            Error::Input(format!(
                "cannot read histogram '{name}' from {}: {e}",
                path.display()
            ))
        })?;
        Self::new(hist)
    }

    pub fn new(hist: Hist1D) -> Result<Self> {
        let nbins = hist.xaxis.nbins.max(0) as usize;
        if nbins == 0 {
            return Err(Error::Input(format!(
                "histogram '{}' has no bins",
                hist.name
            )));
        }
        let edges = hist.edges();
        if edges.len() != nbins + 1 || edges.windows(2).any(|w| !(w[0] < w[1])) {
            return Err(Error::Input(format!(
                "histogram '{}' has invalid bin edges",
                hist.name
            )));
        }
        let counts: Vec<f64> = (1..=nbins).map(|i| hist.contents[i]).collect();
        let errors: Vec<f64> = (1..=nbins).map(|i| hist.bin_error(i)).collect();
        if counts.iter().any(|c| !c.is_finite() || *c < 0.) {
            return Err(Error::Input(format!(
                "histogram '{}' has negative or non-finite bin contents",
                hist.name
            )));
        }
        let total: f64 = hist.contents.iter().sum();
        if !(total > 0.) {
            return Err(Error::Input(format!("histogram '{}' is empty", hist.name)));
        }
        let weighted = counts
            .iter()
            .zip(&errors)
            .any(|(c, e)| (e * e - c).abs() > 1e-6 * c.max(1.));
        Ok(DataHistogram {
            hist,
            edges,
            counts,
            errors,
            total,
            weighted,
        })
    }

    pub fn n_bins(&self) -> usize {
        self.counts.len()
    }

    pub fn center(&self, i: usize) -> f64 {
        0.5 * (self.edges[i] + self.edges[i + 1])
    }

    /// Bins (0-based, inclusive) whose centers lie in `[min, max]`.
    pub fn fit_bins(&self, min: Option<f64>, max: Option<f64>) -> Result<(usize, usize)> {
        let lo = min.unwrap_or(f64::NEG_INFINITY);
        let hi = max.unwrap_or(f64::INFINITY);
        let inside: Vec<usize> = (0..self.n_bins())
            .filter(|&i| (lo..=hi).contains(&self.center(i)))
            .collect();
        match (inside.first(), inside.last()) {
            (Some(&a), Some(&b)) => Ok((a, b)),
            _ => Err(Error::Config(format!(
                "no bin of '{}' has its center in the fit range [{lo}, {hi}]",
                self.hist.name
            ))),
        }
    }

    /// Fraction of the entries above each bin's center (half of the bin
    /// itself included), from the highest bin down.
    pub fn upper_fractions(&self) -> Vec<f64> {
        let overflow = self.hist.contents.last().copied().unwrap_or(0.);
        let mut above = overflow;
        let mut out = vec![0.; self.n_bins()];
        for i in (0..self.n_bins()).rev() {
            out[i] = (above + 0.5 * self.counts[i]) / self.total;
            above += self.counts[i];
        }
        out
    }
}
