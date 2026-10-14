//! Special functions, quadrature and small dense linear algebra.

use std::f64::consts::SQRT_2;

const EPS: f64 = 1e-16;
const TINY: f64 = 1e-300;
const MAX_ITER: usize = 100_000;

/// `ln Γ(x)` for `x > 0`.
pub fn ln_gamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}

/// Regularized incomplete gamma functions `(P(a, x), Q(a, x))` for `a > 0`,
/// with `ln Γ(a)` precomputed by the caller (it is shared by many `x`).
///
/// The function that is small on the side of `x` is computed directly (series
/// for `P` below `a + 1`, continued fraction for `Q` above), so both tails keep
/// full relative precision; the other one is its complement.
pub fn gamma_pq(a: f64, x: f64, ln_gamma_a: f64) -> (f64, f64) {
    if !(x > 0.) {
        return (0., 1.);
    }
    if x == f64::INFINITY {
        return (1., 0.);
    }
    let log_prefactor = a * x.ln() - x - ln_gamma_a;
    if x < a + 1. {
        let mut ap = a;
        let mut del = 1. / a;
        let mut sum = del;
        for _ in 0..MAX_ITER {
            ap += 1.;
            del *= x / ap;
            sum += del;
            if del < sum * EPS {
                break;
            }
        }
        let p = (log_prefactor.exp() * sum).min(1.);
        (p, 1. - p)
    } else {
        /* modified Lentz's method */
        let mut b = x + 1. - a;
        let mut c = 1. / TINY;
        let mut d = 1. / b;
        let mut h = d;
        for i in 1..MAX_ITER {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1. / d;
            let del = d * c;
            h *= del;
            if (del - 1.).abs() < EPS {
                break;
            }
        }
        let q = (log_prefactor.exp() * h).min(1.);
        (1. - q, q)
    }
}

/// Standard normal cumulative distribution function.
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-z / SQRT_2)
}

/// Standard normal survival function `1 - Φ(z)`, accurate in the right tail.
pub fn normal_sf(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(z / SQRT_2)
}

/// Nodes and weights of the `n`-point Gauss-Legendre rule on `[-1, 1]`.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n > 0, "Gauss-Legendre rule needs at least one point");
    let mut nodes = vec![0.; n];
    let mut weights = vec![0.; n];
    let nf = n as f64;
    for i in 0..n.div_ceil(2) {
        /* initial guess for the i-th root, refined with Newton's method */
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut dp = 1.;
        for _ in 0..100 {
            /* Legendre recurrence: p1 = P_n(z), p2 = P_{n-1}(z) */
            let (mut p1, mut p2) = (1., 0.);
            for j in 1..=n {
                let jf = j as f64;
                let p3 = p2;
                p2 = p1;
                p1 = ((2. * jf - 1.) * z * p2 - (jf - 1.) * p3) / jf;
            }
            dp = nf * (z * p1 - p2) / (z * z - 1.);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        let w = 2. / ((1. - z * z) * dp * dp);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

/// Composite Gauss-Legendre rule: `panels` equal panels on `[a, b]`, each with
/// `order` points. Returns (nodes, weights).
pub fn composite_gauss_legendre(
    a: f64,
    b: f64,
    panels: usize,
    order: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (x, w) = gauss_legendre(order);
    let h = (b - a) / panels as f64;
    let mut nodes = Vec::with_capacity(panels * order);
    let mut weights = Vec::with_capacity(panels * order);
    for p in 0..panels {
        let mid = a + (p as f64 + 0.5) * h;
        for (xi, wi) in x.iter().zip(&w) {
            nodes.push(mid + 0.5 * h * xi);
            weights.push(0.5 * h * wi);
        }
    }
    (nodes, weights)
}

/// Solves the dense system `m · x = rhs` (`m` row-major `n × n`) by Gaussian
/// elimination with partial pivoting. `None` if `m` is singular.
pub fn solve(mut m: Vec<f64>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    assert_eq!(m.len(), n * n);
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&a, &b| m[a * n + col].abs().total_cmp(&m[b * n + col].abs()))?;
        if !(m[pivot * n + col].abs() > 1e-300) {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                m.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        for row in col + 1..n {
            let f = m[row * n + col] / m[col * n + col];
            for k in col..n {
                m[row * n + k] -= f * m[col * n + k];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    let mut x = vec![0.; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| m[row * n + k] * x[k]).sum();
        x[row] = (rhs[row] - s) / m[row * n + row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Lower triangular Cholesky factor `L` (row-major) of the symmetric matrix
/// `m`, `m = L Lᵀ`. Tiny negative pivots from rounding are clamped to zero;
/// `None` if `m` is clearly not positive semi-definite.
pub fn cholesky(m: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let d = m[i * n + i] - s;
                if d < -1e-10 * m[i * n + i].abs().max(1e-300) {
                    return None;
                }
                l[i * n + i] = d.max(0.).sqrt();
            } else {
                let d = l[j * n + j];
                l[i * n + j] = if d > 0. { (m[i * n + j] - s) / d } else { 0. };
            }
        }
    }
    Some(l)
}

/// Bisection for the root of a function that changes sign on `[lo, hi]`.
pub fn bisect(f: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64, iterations: usize) -> f64 {
    let f_lo = f(lo);
    for _ in 0..iterations {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        if (f(mid) > 0.) == (f_lo > 0.) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn incomplete_gamma_known_values() {
        /* P(1, x) = 1 - exp(-x) */
        for x in [0.01, 0.5, 1., 3., 10., 50.] {
            let (p, q) = gamma_pq(1., x, ln_gamma(1.));
            assert!(close(q, (-x).exp(), 1e-13), "x = {x}: {q}");
            assert!(close(p, -(-x).exp_m1(), 1e-13), "x = {x}: {p}");
        }
        /* P(a, x) for integer a: Poisson sum */
        let a = 5.;
        for x in [0.3, 2., 5., 9., 30.] {
            let poisson: f64 = (0..5)
                .map(|k| (-x + k as f64 * f64::ln(x) - ln_gamma(k as f64 + 1.)).exp())
                .sum();
            let (_, q) = gamma_pq(a, x, ln_gamma(a));
            assert!(close(q, poisson, 1e-12), "x = {x}: {q} vs {poisson}");
        }
        /* agreement with statrs in the bulk for a large shape */
        let a = 2500.;
        for x in [2300., 2450., 2500., 2600.] {
            let (p, _) = gamma_pq(a, x, ln_gamma(a));
            let reference = statrs::function::gamma::gamma_lr(a, x);
            assert!(close(p, reference, 1e-9), "x = {x}: {p} vs {reference}");
        }
        assert_eq!(gamma_pq(2., 0., ln_gamma(2.)), (0., 1.));
    }

    #[test]
    fn gauss_legendre_integrates_polynomials() {
        for n in [1, 2, 5, 8, 16] {
            let (x, w) = gauss_legendre(n);
            assert!(close(w.iter().sum::<f64>(), 2., 1e-13));
            /* exact for degree 2n - 1 */
            let deg = 2 * n - 2;
            let integral: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(deg as i32)).sum();
            assert!(close(integral, 2. / (deg as f64 + 1.), 1e-12), "n = {n}");
        }
        let (x, w) = composite_gauss_legendre(0., 2., 10, 4);
        let integral: f64 = x.iter().zip(&w).map(|(x, w)| w * x.exp()).sum();
        assert!(close(integral, 2f64.exp() - 1., 1e-13));
    }

    #[test]
    fn linear_algebra() {
        let m = vec![4., 2., 0.6, 2., 5., 1., 0.6, 1., 3.];
        let x = solve(m.clone(), vec![1., 2., 3.]).unwrap();
        for i in 0..3 {
            let r: f64 = (0..3).map(|k| m[i * 3 + k] * x[k]).sum();
            assert!(close(r, [1., 2., 3.][i], 1e-12));
        }
        let l = cholesky(&m, 3).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let r: f64 = (0..3).map(|k| l[i * 3 + k] * l[j * 3 + k]).sum();
                assert!(close(r, m[i * 3 + j], 1e-12));
            }
        }
        assert!(solve(vec![1., 2., 2., 4.], vec![1., 1.]).is_none());
    }

    #[test]
    fn normal_and_bisection() {
        assert!(close(normal_cdf(0.), 0.5, 1e-15));
        assert!(close(normal_sf(10.), 7.619853024160527e-24, 1e-10));
        let root = bisect(|x| x * x - 2., 0., 2., 200);
        assert!(close(root, SQRT_2, 1e-14));
    }
}
