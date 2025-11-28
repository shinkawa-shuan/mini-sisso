use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Axis};
use rayon::prelude::*;
use std::collections::HashSet;
use std::cmp::Ordering;
use nalgebra::{DMatrix, DVector}; // 外部BLAS不要の線形代数ライブラリ

/// 特徴量構造体
#[derive(Clone, Debug)]
struct Feature {
    data: Array1<f64>,
    expr: String,
}

// 安全な計算を行うヘルパー関数
fn safe_unary_eval(op: &str, data: &Array1<f64>) -> Option<Array1<f64>> {
    let res = match op {
        "exp" => {
            if data.iter().any(|&v| v > 700.0) { return None; }
            data.mapv(f64::exp)
        },
        "log" => {
            if data.iter().any(|&v| v.abs() <= 1e-9) { return None; }
            data.mapv(|v| v.abs().ln()) // log(|x|)
        },
        "inv" => {
            if data.iter().any(|&v| v.abs() <= 1e-9) { return None; }
            data.mapv(|v| 1.0 / v)
        },
        "sqrt" => {
             // sqrt(|x|) を採用してロバストにする
            data.mapv(|v| v.abs().sqrt())
        },
        "sin" => data.mapv(f64::sin),
        "cos" => data.mapv(f64::cos),
        "pow2" => {
             if data.iter().any(|&v| v.abs() > 1e150) { return None; }
             data.mapv(|v| v * v)
        },
        "pow3" => {
            if data.iter().any(|&v| v.abs() > 1e100) { return None; }
            data.mapv(|v| v * v * v)
        },
        _ => return None
    };
    
    if res.iter().any(|&v| v.is_nan() || v.is_infinite()) { return None; }
    Some(res)
}

fn safe_binary_eval(op: &str, a: &Array1<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let res = match op {
        "add" => a + b,
        "sub" => a - b,
        "mul" => {
            // 簡易オーバーフローチェック
            if a.iter().zip(b.iter()).any(|(x, y)| x.abs() > 1e150 || y.abs() > 1e150) { return None; }
            a * b
        },
        "div" => {
            if b.iter().any(|&v| v.abs() < 1e-9) { return None; }
            a / b
        },
        _ => return None
    };
    
    if res.iter().any(|&v| v.is_nan() || v.is_infinite()) { return None; }
    Some(res)
}

/// SIS (Sure Independence Screening)
fn perform_sis(
    candidates: &[Feature],
    residual: &Array1<f64>,
    k: usize
) -> Vec<Feature> {
    if candidates.is_empty() { return Vec::new(); }

    let r_mean = residual.mean().unwrap_or(0.0);
    let r_centered = residual - r_mean;
    let r_norm = r_centered.dot(&r_centered).sqrt();

    let mut scored_features: Vec<(&Feature, f64)> = candidates.par_iter().map(|f| {
        let f_mean = f.data.mean().unwrap_or(0.0);
        let f_centered = &f.data - f_mean;
        let f_norm = f_centered.dot(&f_centered).sqrt();
        
        let score = if f_norm > 1e-9 && r_norm > 1e-9 {
            (f_centered.dot(&r_centered) / (f_norm * r_norm)).abs()
        } else {
            0.0
        };
        (f, score)
    }).collect();
    
    scored_features.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            .then_with(|| a.0.expr.cmp(&b.0.expr))
    });
    
    scored_features.into_iter().take(k).map(|(f, _)| f.clone()).collect()
}

/// OLS Solver using Nalgebra with Standardization (Robust)
fn solve_ols_nalgebra(features: &[&Feature], y: &Array1<f64>) -> (f64, Vec<f64>, f64) {
    let n_samples = y.len();
    let n_features = features.len();
    
    // 1. データを標準化する (平均0, 分散1)
    let mut x_mat = DMatrix::zeros(n_samples, n_features);
    let mut x_means = Vec::with_capacity(n_features);
    let mut x_stds = Vec::with_capacity(n_features);
    
    for (j, f) in features.iter().enumerate() {
        let mean = f.data.mean().unwrap_or(0.0);
        let std = f.data.std(0.0);
        let safe_std = if std > 1e-9 { std } else { 1.0 };
        
        x_means.push(mean);
        x_stds.push(safe_std);
        
        for i in 0..n_samples {
            x_mat[(i, j)] = (f.data[i] - mean) / safe_std;
        }
    }
    
    // y も中心化する (切片計算のため)
    let y_mean = y.mean().unwrap_or(0.0);
    let y_centered: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();
    let y_vec = DVector::from_vec(y_centered);
    
    // 2. 標準化されたデータでOLSを解く (切片なし)
    match x_mat.clone().svd(true, true).solve(&y_vec, 1e-9) {
        Ok(coeffs_std) => {
            // 3. 係数を元のスケールに戻す
            let mut real_coeffs = Vec::with_capacity(n_features);
            let mut intercept = y_mean;
            
            for j in 0..n_features {
                let w = coeffs_std[j];
                let s = x_stds[j];
                let m = x_means[j];
                
                let c = w / s;
                real_coeffs.push(c);
                intercept -= c * m;
            }
            
            // RMSE計算 (元のスケールで再計算して確認)
            let mut mse = 0.0;
            for i in 0..n_samples {
                let mut y_pred = intercept;
                for j in 0..n_features {
                    y_pred += real_coeffs[j] * features[j].data[i];
                }
                let diff = y[i] - y_pred;
                mse += diff * diff;
            }
            let rmse = (mse / n_samples as f64).sqrt();

            (rmse, real_coeffs, intercept)
        },
        Err(_) => (f64::INFINITY, vec![0.0; n_features], 0.0)
    }
}

#[pyfunction]
fn rust_fit_exhaustive(
    _py: Python,
    features: PyReadonlyArray2<f64>,
    target: PyReadonlyArray1<f64>,
    feature_names: Vec<String>,
    operators: Vec<String>,
    n_expansion: usize,
    n_term: usize,
    n_sis_features: usize, // k_per_levelに相当
) -> PyResult<(f64, String, Vec<f64>, f64)> { 
    
    let x = features.as_array().to_owned();
    let y = target.as_array().to_owned();
    
    // 初期特徴量
    let base_features: Vec<Feature> = x.axis_iter(Axis(1))
        .zip(feature_names.iter())
        .map(|(col, name)| Feature {
            data: col.to_owned(),
            expr: name.clone(),
        })
        .collect();

    // --- レベルワイズ特徴生成 ---
    let mut current_level_pool = base_features.clone(); 
    let mut all_promising_pool = base_features.clone();
    let mut seen_exprs: HashSet<String> = base_features.iter().map(|f| f.expr.clone()).collect();

    let unary_ops: Vec<&String> = operators.iter().filter(|&op| !["add", "sub", "mul", "div"].contains(&op.as_str())).collect();
    let binary_ops: Vec<&String> = operators.iter().filter(|&op| ["add", "sub", "mul", "div"].contains(&op.as_str())).collect();

    for _level in 0..n_expansion {
        let mut next_gen_candidates = Vec::new();

        let unary_generated: Vec<Feature> = current_level_pool.par_iter().flat_map_iter(|f| {
            let mut local_res = Vec::new();
            for op in &unary_ops {
                if let Some(new_data) = safe_unary_eval(op, &f.data) {
                     local_res.push(Feature { 
                         data: new_data, 
                         expr: format!("{}({})", op, f.expr) 
                     });
                }
            }
            local_res
        }).collect();
        next_gen_candidates.extend(unary_generated);

        if !binary_ops.is_empty() {
            let binary_ops_local = binary_ops.clone();
            let pool_local = all_promising_pool.clone();
            let binary_generated: Vec<Feature> = current_level_pool.par_iter().flat_map_iter(move |f1| {
                let ops = binary_ops_local.clone();
                pool_local.iter().flat_map(move |f2| {
                    let mut local_res = Vec::new();
                    for op in &ops {
                        if ["add", "mul"].contains(&op.as_str()) && f1.expr > f2.expr { continue; }
                        if f1.expr == f2.expr && ["sub", "div"].contains(&op.as_str()) { continue; }

                        if let Some(new_data) = safe_binary_eval(op, &f1.data, &f2.data) {
                            local_res.push(Feature {
                                data: new_data,
                                expr: format!("({}{}{})", f1.expr, op_symbol(op), f2.expr) 
                            });
                        }
                    }
                    local_res
                }).collect::<Vec<_>>().into_iter()
            }).collect();
            next_gen_candidates.extend(binary_generated);
        }

        let unique_candidates: Vec<Feature> = next_gen_candidates.into_iter()
            .filter(|f| seen_exprs.insert(f.expr.clone())) 
            .collect();

        if unique_candidates.is_empty() { break; }

        let selected_features = perform_sis(&unique_candidates, &y, n_sis_features);
        
        current_level_pool = selected_features.clone();
        all_promising_pool.extend(selected_features);
    }

    // --- SO (Exhaustive Search) ---
    all_promising_pool.sort_by(|a, b| a.expr.cmp(&b.expr));

    let mut best_global_rmse = f64::INFINITY;
    let mut best_global_model = (String::new(), Vec::new(), 0.0);

    let mut current_residual = y.clone();
    let mut final_model_pool: Vec<Feature> = Vec::new();

    for _term_idx in 1..=n_term {
        let pool_exprs: HashSet<String> = final_model_pool.iter().map(|f| f.expr.clone()).collect();
        let candidates: Vec<Feature> = all_promising_pool.iter()
            .filter(|f| !pool_exprs.contains(&f.expr))
            .cloned()
            .collect();
            
        let top_k = perform_sis(&candidates, &current_residual, n_sis_features);
        final_model_pool.extend(top_k);
        
        use itertools::Itertools;
        
        let best_in_term_opt = final_model_pool.iter()
            .combinations(_term_idx)
            .par_bridge()
            .map(|combo| {
                let (rmse, coeffs, intercept) = solve_ols_nalgebra(&combo, &y);
                (rmse, coeffs, intercept, combo)
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        if let Some((rmse, coeffs, intercept, features)) = best_in_term_opt {
            let mut y_pred = Array1::from_elem(y.len(), intercept);
            for (i, f) in features.iter().enumerate() {
                y_pred = y_pred + &f.data * coeffs[i];
            }
            current_residual = &y - &y_pred;

            let mut eq_parts = Vec::new();
            for (i, f) in features.iter().enumerate() {
                let val: f64 = coeffs[i];
                eq_parts.push(format!("{:.6} * {}", val, f.expr));
            }
            let intercept_val: f64 = intercept;
            let eq_str = format!("{} + {:.6}", eq_parts.join(" + "), intercept_val);

            if rmse < best_global_rmse {
                best_global_rmse = rmse;
                best_global_model = (eq_str, coeffs, intercept);
            }
        }
    }

    Ok((best_global_rmse, best_global_model.0, best_global_model.1, best_global_model.2))
}

// 演算子の記号変換
fn op_symbol(op: &str) -> &str {
    match op {
        "add" => "+",
        "sub" => "-",
        "mul" => "*",
        "div" => "/",
        _ => op
    }
}

#[pymodule]
fn _mini_sisso_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_rust_connection, m)?)?;
    m.add_function(wrap_pyfunction!(rust_fit_exhaustive, m)?)?;
    Ok(())
}

#[pyfunction]
fn test_rust_connection<'py>(
    _py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<(usize, f64)> {
    let x_array = x.as_array();
    let y_array = y.as_array();
    Ok((x_array.shape()[0], y_array.sum()))
}
