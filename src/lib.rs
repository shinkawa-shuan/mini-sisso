use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2, Axis, s, ArrayView1, ArrayView2};
use ndarray_linalg::Solve; // For OLS
use rayon::prelude::*;
use std::collections::HashSet;
use std::cmp::Ordering;

/// 安全な計算を行うヘルパー関数
fn safe_eval(op: &str, data: &Array1<f64>) -> Option<Array1<f64>> {
    match op {
        "exp" => {
            if data.iter().any(|&v| v > 700.0) { return None; }
            Some(data.mapv(f64::exp))
        },
        "log" => {
            if data.iter().any(|&v| v <= 1e-9) { return None; }
            Some(data.mapv(f64::ln))
        },
        "inv" | "div" => {
            if data.iter().any(|&v| v.abs() < 1e-9) { return None; }
            Some(data.mapv(|v| 1.0 / v))
        },
        "sqrt" => {
            if data.iter().any(|&v| v < 0.0) { return None; }
            Some(data.mapv(f64::sqrt))
        },
        "cbrt" => Some(data.mapv(f64::cbrt)),
        "abs" => Some(data.mapv(f64::abs)),
        "sin" => Some(data.mapv(f64::sin)),
        "cos" => Some(data.mapv(f64::cos)),
        "pow2" => {
             if data.iter().any(|&v| v.abs() > 1.34e154) { return None; }
             Some(data.mapv(|v| v * v))
        },
        "pow3" => {
            if data.iter().any(|&v| v.abs() > 5.64e102) { return None; }
            Some(data.mapv(|v| v * v * v))
        },
        _ => Some(data.clone())
    }
}

fn safe_binary_eval(op: &str, a: &Array1<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    match op {
        "add" => Some(a + b),
        "sub" => Some(a - b),
        "mul" => {
            let max_val = f64::MAX.sqrt();
            if a.iter().zip(b.iter()).any(|(&x, &y)| x.abs() > max_val || y.abs() > max_val) {
                 // simplified check
            }
            let res = a * b;
            if res.iter().any(|&v| v.is_infinite() || v.is_nan()) { return None; }
            Some(res)
        },
        "div" => {
            if b.iter().any(|&v| v.abs() < 1e-9) { return None; }
            Some(a / b)
        },
        _ => None
    }
}

#[derive(Clone, Debug)]
struct Feature {
    data: Array1<f64>,
    expr: String,
}

// Implement PartialEq and Eq for deterministic sorting based on expr
impl PartialEq for Feature {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}
impl Eq for Feature {}

fn generate_features(
    base_features: &Array2<f64>,
    base_names: &[String],
    operators: &[String],
    n_expansion: usize
) -> Vec<Feature> {
    let mut current_features: Vec<Feature> = base_features.axis_iter(Axis(1))
        .zip(base_names.iter())
        .map(|(col, name)| Feature {
            data: col.to_owned(),
            expr: name.clone(),
        })
        .collect();
    
    let mut seen_exprs: HashSet<String> = current_features.iter().map(|f| f.expr.clone()).collect();
    
    for _ in 0..n_expansion {
        let mut new_features = Vec::new();
        
        // Unary
        for feat in &current_features {
            for op in operators {
                if ["exp", "log", "inv", "sqrt", "cbrt", "abs", "sin", "cos", "pow2", "pow3"].contains(&op.as_str()) {
                     if let Some(new_data) = safe_eval(op, &feat.data) {
                         let new_expr = format!("{}({})", op, feat.expr);
                         if !seen_exprs.contains(&new_expr) {
                             new_features.push(Feature { data: new_data, expr: new_expr.clone() });
                             seen_exprs.insert(new_expr);
                         }
                     }
                }
            }
        }
        
        // Binary
        let binary_ops: Vec<&String> = operators.iter().filter(|&op| ["add", "sub", "mul", "div"].contains(&op.as_str())).collect();
        
        if !binary_ops.is_empty() {
             let generated: Vec<Feature> = current_features.par_iter().flat_map(|f1| {
                 current_features.par_iter().flat_map(|f2| {
                     let mut local_new = Vec::new();
                     for op in &binary_ops {
                         if ["add", "mul"].contains(&op.as_str()) && f1.expr > f2.expr { continue; }
                         if f1.expr == f2.expr && ["sub", "div"].contains(&op.as_str()) { continue; }

                         if let Some(new_data) = safe_binary_eval(op, &f1.data, &f2.data) {
                             let new_expr = format!("{}({},{})", op, f1.expr, f2.expr);
                             local_new.push(Feature { data: new_data, expr: new_expr });
                         }
                     }
                     local_new
                 })
             }).collect();
             
             for f in generated {
                 if !seen_exprs.contains(&f.expr) {
                     seen_exprs.insert(f.expr.clone());
                     new_features.push(f);
                 }
             }
        }
        current_features.extend(new_features);
    }
    current_features
}

/// SIS (Sure Independence Screening)
/// Returns top k features correlated with residual
fn perform_sis(
    candidates: &[Feature],
    residual: &Array1<f64>,
    k: usize
) -> Vec<Feature> {
    let mut scored_features: Vec<(&Feature, f64)> = candidates.par_iter().map(|f| {
        // Pearson correlation
        // corr(f, r) = cov(f, r) / (std(f) * std(r))
        // We only need to rank, so we can ignore std(r) which is constant.
        // score = |cov(f, r) / std(f)|
        
        let f_mean = f.data.mean().unwrap_or(0.0);
        let r_mean = residual.mean().unwrap_or(0.0);
        
        let f_centered = &f.data - f_mean;
        let r_centered = residual - r_mean;
        
        let num = f_centered.dot(&r_centered);
        let den = f_centered.dot(&f_centered).sqrt();
        
        let score = if den > 1e-9 {
            (num / den).abs()
        } else {
            0.0
        };
        
        (f, score)
    }).collect();
    
    // Sort by score descending, then by expr string for determinism
    scored_features.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal) // Score desc
            .then_with(|| a.0.expr.cmp(&b.0.expr))       // Expr asc (tie-breaker)
    });
    
    scored_features.into_iter().take(k).map(|(f, _)| f.clone()).collect()
}

/// OLS Solver
/// Returns RMSE and Coefficients
fn solve_ols(features: &[&Feature], y: &Array1<f64>) -> (f64, Array1<f64>, f64) {
    let n_samples = y.len();
    let n_features = features.len();
    
    // Construct X matrix with intercept (column of 1s)
    // Or just center X and y to avoid intercept in matrix
    // Let's do centered OLS
    
    let y_mean = y.mean().unwrap_or(0.0);
    let y_centered = y - y_mean;
    
    let mut x_mat = Array2::<f64>::zeros((n_samples, n_features));
    let mut x_means = Array1::<f64>::zeros(n_features);
    let mut x_stds = Array1::<f64>::zeros(n_features);

    for (j, f) in features.iter().enumerate() {
        let f_mean = f.data.mean().unwrap_or(0.0);
        let f_std = f.data.std(0.0);
        x_means[j] = f_mean;
        x_stds[j] = if f_std > 1e-9 { f_std } else { 1.0 };
        
        let col = (&f.data - f_mean) / x_stds[j];
        x_mat.column_mut(j).assign(&col);
    }
    
    // Solve (X^T X)^-1 X^T y
    // Using ndarray-linalg's least_squares or solve
    // Since X is centered and standardized, we are solving for standardized coefficients
    
    // Note: ndarray-linalg might need openblas. 
    // If we want to avoid complex dependencies, we can use simple matrix inversion if dim is small (SISSO dim is usually small < 10)
    
    let xt = x_mat.t();
    let xtx = xt.dot(&x_mat);
    let xty = xt.dot(&y_centered);
    
    // Solve xtx * w = xty
    match xtx.solve_into(xty) {
        Ok(w) => {
            // Calculate RMSE
            let y_pred_centered = x_mat.dot(&w);
            let residuals = &y_centered - &y_pred_centered;
            let mse = residuals.dot(&residuals) / (n_samples as f64);
            let rmse = mse.sqrt();
            
            // Convert back to original coefficients
            // y = y_mean + sum(w_j * (x_j - m_j)/s_j)
            //   = (y_mean - sum(w_j * m_j / s_j)) + sum((w_j/s_j) * x_j)
            
            let real_coeffs = &w / &x_stds;
            let intercept = y_mean - real_coeffs.dot(&x_means);
            
            (rmse, real_coeffs, intercept)
        },
        Err(_) => (f64::INFINITY, Array1::zeros(n_features), 0.0)
    }
}

#[pyfunction]
fn rust_fit_exhaustive(
    py: Python,
    features: PyReadonlyArray2<f64>,
    target: PyReadonlyArray1<f64>,
    feature_names: Vec<String>,
    operators: Vec<String>,
    n_expansion: usize,
    n_term: usize,
    n_sis_features: usize,
) -> PyResult<(f64, String, Vec<f64>, f64)> { // RMSE, Equation, Coeffs, Intercept
    
    let x = features.as_array().to_owned();
    let y = target.as_array().to_owned();
    
    // 1. Feature Generation
    let all_features = generate_features(&x, &feature_names, &operators, n_expansion);
    
    // 2. Iterative SIS + SO
    let mut pool: Vec<Feature> = Vec::new();
    let mut residual = y.clone();
    let mut best_model_info = (f64::INFINITY, String::new(), Vec::new(), 0.0);
    
    // We need to track which features are already in pool to avoid duplicates?
    // perform_sis returns features.
    
    for d in 1..=n_term {
        // SIS
        // Filter out features already in pool? 
        // Original code: [r for r in self.all_recipes if r not in pool]
        // We can do this by expr check
        let pool_exprs: HashSet<String> = pool.iter().map(|f| f.expr.clone()).collect();
        let candidates: Vec<Feature> = all_features.iter()
            .filter(|f| !pool_exprs.contains(&f.expr))
            .cloned()
            .collect();
            
        let top_k = perform_sis(&candidates, &residual, n_sis_features);
        pool.extend(top_k);
        
        // SO (Combinations)
        // Use itertools to generate combinations, then par_bridge for parallel processing
        use itertools::Itertools;
        
        let (rmse, coeffs, intercept, features) = pool.iter()
            .combinations(d)
            .par_bridge() 
            .map(|combo| {
                let (rmse, coeffs, intercept) = solve_ols(&combo, &y);
                (rmse, coeffs, intercept, combo)
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
            .unwrap();
            
        // Update residual
        // y_pred = intercept + sum(c * f)
        let mut y_pred = Array1::from_elem(y.len(), intercept);
        for (i, f) in features.iter().enumerate() {
            y_pred = y_pred + &f.data * coeffs[i];
        }
        residual = &y - &y_pred;
        
        // Format equation
        // Format equation
        let mut eq_parts = Vec::new();
        for (i, f) in features.iter().enumerate() {
            let val: f64 = coeffs[i];
            eq_parts.push(format!("{:.6} * {}", val, f.expr));
        }
        let intercept_val: f64 = intercept;
        let eq_str = format!("{} + {:.6}", eq_parts.join(" + "), intercept_val);
        
        best_model_info = (rmse, eq_str, coeffs.to_vec(), intercept);
    }
    
    Ok(best_model_info)
}

#[pymodule]
fn _mini_sisso_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_rust_connection, m)?)?;
    m.add_function(wrap_pyfunction!(rust_fit_exhaustive, m)?)?;
    Ok(())
}

#[pyfunction]
fn test_rust_connection<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<(usize, f64)> {
    let x_array = x.as_array();
    let y_array = y.as_array();
    Ok((x_array.shape()[0], y_array.sum()))
}
