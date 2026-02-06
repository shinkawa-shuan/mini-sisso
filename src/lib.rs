use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashSet;
use std::cmp::Ordering;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

// --- レシピ定義 ---
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Recipe {
    Base(usize),
    Unary(String, Box<Recipe>),
    Binary(String, Box<Recipe>, Box<Recipe>),
}

impl Recipe {
    fn to_py_object(&self, py: Python) -> PyObject {
        match self {
            Recipe::Base(idx) => idx.into_pyobject(py).unwrap().into_any().unbind(),
            Recipe::Unary(op, child) => (op, child.to_py_object(py)).into_pyobject(py).unwrap().into_any().unbind(),
            Recipe::Binary(op, left, right) => (op, left.to_py_object(py), right.to_py_object(py)).into_pyobject(py).unwrap().into_any().unbind(),
        }
    }
}

// --- 軽量化された特徴量構造体 (データを保持しない) ---
#[derive(Clone, Debug)]
struct Feature {
    expr: String,
    recipe: Recipe,
}

// --- 遅延評価ロジック (On-the-fly) ---

// レシピからデータをその場で計算する関数
fn evaluate_recipe(recipe: &Recipe, base_features: &Array2<f64>) -> Option<Array1<f64>> {
    match recipe {
        Recipe::Base(idx) => {
            if *idx < base_features.ncols() {
                Some(base_features.column(*idx).to_owned())
            } else {
                None
            }
        },
        Recipe::Unary(op, child) => {
            let data = evaluate_recipe(child, base_features)?;
            safe_unary_eval(op, &data)
        },
        Recipe::Binary(op, left, right) => {
            let data_l = evaluate_recipe(left, base_features)?;
            let data_r = evaluate_recipe(right, base_features)?;
            safe_binary_eval(op, &data_l, &data_r)
        }
    }
}

// 安全な単項演算
fn safe_unary_eval(op: &str, data: &Array1<f64>) -> Option<Array1<f64>> {
    let res = match op {
        "exp" => {
            if data.iter().any(|&v| v > 700.0) { return None; }
            data.mapv(f64::exp)
        },
        "log" => {
            if data.iter().any(|&v| v <= 1e-9) { return None; } // logは正の値のみ
            data.mapv(f64::ln)
        },
        "inv" | "^-1" => {
            if data.iter().any(|&v| v.abs() <= 1e-9) { return None; }
            data.mapv(|v| 1.0 / v)
        },
        "sqrt" => {
            // sqrt(|x|) でロバスト化
            data.mapv(|v| v.abs().sqrt())
        },
        "cbrt" => data.mapv(f64::cbrt),
        "abs" => data.mapv(f64::abs),
        "sin" => data.mapv(f64::sin),
        "cos" => data.mapv(f64::cos),
        "pow2" | "^2" => {
             if data.iter().any(|&v| v.abs() > 1e150) { return None; }
             data.mapv(|v| v * v)
        },
        "pow3" | "^3" => {
            if data.iter().any(|&v| v.abs() > 1e100) { return None; }
            data.mapv(|v| v * v * v)
        },
        "scd" => {
            // Standard Cauchy Distribution: 1 / (pi * (1 + x^2))
            if data.iter().any(|&v| v.abs() > 1e100) { return None; }
            data.mapv(|v| 1.0 / (PI * (1.0 + v * v)))
        },
        _ => return None
    };
    
    if res.iter().any(|&v| v.is_nan() || v.is_infinite()) { return None; }
    Some(res)
}

// 安全な二項演算
fn safe_binary_eval(op: &str, a: &Array1<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let res = match op {
        "add" | "+" => a + b,
        "sub" | "-" => a - b,
        "mul" | "*" => {
            if a.iter().zip(b.iter()).any(|(x, y)| x.abs() > 1e150 || y.abs() > 1e150) { return None; }
            a * b
        },
        "div" | "/" => {
            if b.iter().any(|&v| v.abs() < 1e-9) { return None; }
            a / b
        },
        "|-|" | "abs_diff" => {
             (a - b).mapv(f64::abs)
        },
        _ => return None
    };
    
    if res.iter().any(|&v| v.is_nan() || v.is_infinite()) { return None; }
    Some(res)
}

// --- SIS (遅延評価・選抜枠分割版) ---
fn perform_sis_lazy(
    candidates: &[Feature],
    base_data: &Array2<f64>,
    residual: &Array1<f64>,
    k: usize,
    use_split_selection: bool
) -> Vec<Feature> {
    if candidates.is_empty() { return Vec::new(); }

    let r_mean = residual.mean().unwrap_or(0.0);
    let r_centered = residual - r_mean;
    let r_norm = r_centered.dot(&r_centered).sqrt();

    // 並列評価とスコアリング (ここで一時的にデータを生成)
    // 戻り値: (Index, Score, Category)
    // Category: 0=Base, 1=Unary, 2=Binary
    let mut scored_indices: Vec<(usize, f64, u8)> = candidates.par_iter().enumerate().map(|(i, f)| {
        if let Some(data) = evaluate_recipe(&f.recipe, base_data) {
            let f_mean = data.mean().unwrap_or(0.0);
            let f_centered = &data - f_mean;
            let f_norm = f_centered.dot(&f_centered).sqrt();
            
            let score = if f_norm > 1e-9 && r_norm > 1e-9 {
                (f_centered.dot(&r_centered) / (f_norm * r_norm)).abs()
            } else {
                0.0
            };

            let category = match f.recipe {
                Recipe::Base(_) => 0,
                Recipe::Unary(_, _) => 1,
                Recipe::Binary(_, _, _) => 2,
            };
            (i, score, category)
        } else {
            (i, 0.0, 255) // 計算失敗
        }
    }).filter(|&(_, score, cat)| cat != 255 && !score.is_nan()).collect();

    // スコア降順ソート
    scored_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // --- Branch Logic ---
    if !use_split_selection {
        // Traditional Top-K
        return scored_indices.into_iter().take(k)
            .map(|(i, _, _)| candidates[i].clone())
            .collect();
    }

    // 選抜ロジック: UnaryとBinaryで枠を分ける (Crowding Out対策)
    let k_split = k / 2 + 1; // それぞれ約半分ずつ
    
    let mut selected_indices = HashSet::new();
    let mut count_unary = 0;
    let mut count_binary = 0;

    // パス1: 各カテゴリの枠を埋める
    for &(idx, _, cat) in &scored_indices {
        let added = match cat {
            0 => true, // ベース特徴量は上位なら常に残す
            1 => if count_unary < k_split { count_unary += 1; true } else { false },
            2 => if count_binary < k_split { count_binary += 1; true } else { false },
            _ => false
        };
        if added {
            selected_indices.insert(idx);
        }
    }

    // パス2: 残りの枠をスコア順に埋める
    for &(idx, _, _) in &scored_indices {
        if selected_indices.len() >= k { break; }
        selected_indices.insert(idx);
    }

    // 選ばれた特徴量を返す
    candidates.iter().enumerate()
        .filter(|(i, _)| selected_indices.contains(i))
        .map(|(_, f)| f.clone())
        .collect()
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
    n_sis_features: usize,
    use_split_selection: bool,
) -> PyResult<(f64, String, Vec<f64>, f64, Vec<PyObject>)> { 
    
    let x_base = features.as_array().to_owned();
    let y = target.as_array().to_owned();
    let y_mean = y.mean().unwrap_or(0.0);
    let y_var = y.var(0.0);
    
    let base_features: Vec<Feature> = feature_names.iter().enumerate()
        .map(|(i, name)| Feature {
            expr: name.clone(),
            recipe: Recipe::Base(i),
        })
        .collect();

    let mut current_level_pool = base_features.clone(); 
    let mut all_promising_pool = base_features.clone();
    let mut seen_exprs: HashSet<String> = base_features.iter().map(|f| f.expr.clone()).collect();

    let binary_symbols = ["+", "-", "*", "/", "|-|", "add", "sub", "mul", "div"];
    let unary_ops: Vec<&String> = operators.iter().filter(|&op| !binary_symbols.contains(&op.as_str())).collect();
    let binary_ops: Vec<&String> = operators.iter().filter(|&op| binary_symbols.contains(&op.as_str())).collect();

    println!("***************** SISSO Feature Expansion (Rust) *****************");
    for level in 1..=n_expansion {
        let mut next_gen_candidates = Vec::new();

        let unary_generated: Vec<Feature> = current_level_pool.par_iter().flat_map_iter(|f| {
            let mut local_res = Vec::new();
            for op in &unary_ops {
                let new_expr = format!("{}({})", op, f.expr);
                let new_recipe = Recipe::Unary(op.to_string(), Box::new(f.recipe.clone()));
                local_res.push(Feature { expr: new_expr, recipe: new_recipe });
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
                        let is_commutative = ["+", "*", "add", "mul", "|-|"].contains(&op.as_str());
                        if is_commutative && f1.expr > f2.expr { continue; }
                        let is_anti_identity = ["-", "/", "sub", "div", "|-|"].contains(&op.as_str());
                        if is_anti_identity && f1.expr == f2.expr { continue; }

                        let op_sym = op_symbol(op);
                        let new_expr = format!("({}{}{})", f1.expr, op_sym, f2.expr);
                        let new_recipe = Recipe::Binary(
                            op_sym.to_string(), 
                            Box::new(f1.recipe.clone()), 
                            Box::new(f2.recipe.clone())
                        );
                        local_res.push(Feature { expr: new_expr, recipe: new_recipe });
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

        let selected_features = perform_sis_lazy(&unique_candidates, &x_base, &y, n_sis_features, use_split_selection);
        
        println!("Level {}: Generated Unique={}, SIS Selected={}", level, unique_candidates.len(), selected_features.len());
        
        current_level_pool = selected_features.clone();
        all_promising_pool.extend(selected_features);
    }

    println!("Pool Size for Exhaustive Search: {}", all_promising_pool.len());
    all_promising_pool.sort_by(|a, b| a.expr.cmp(&b.expr));

    // --- SO (Optimized Performance & Progress Logging) ---
    let mut best_global_rmse = f64::INFINITY;
    let mut best_global_model = (String::new(), Vec::new(), 0.0, Vec::new());

    let mut current_residual = y.clone();
    let mut final_model_pool: Vec<Feature> = Vec::new();

    println!("\n***************** Starting SISSO Regressor (Rust) *****************");
    let n_samples = y.len();
    let y_centered: Array1<f64> = &y - y_mean;
    let y_sum_sq = y_centered.dot(&y_centered);

    for term_idx in 1..=n_term {
        println!("\n===== Searching for {}-term models =====", term_idx);
        
        let pool_exprs: HashSet<String> = final_model_pool.iter().map(|f| f.expr.clone()).collect();
        let candidates: Vec<Feature> = all_promising_pool.iter()
            .filter(|f| !pool_exprs.contains(&f.expr))
            .cloned()
            .collect();
            
        let top_k = perform_sis_lazy(&candidates, &x_base, &current_residual, n_sis_features, use_split_selection);
        final_model_pool.extend(top_k);
        
        // --- Gram Matrix Pre-calculation ---
        let m_pool = final_model_pool.len();
        let mut p_mat = DMatrix::zeros(n_samples, m_pool);
        let mut pool_means = Vec::with_capacity(m_pool);
        let mut pool_stds = Vec::with_capacity(m_pool);

        for (j, f) in final_model_pool.iter().enumerate() {
            let data = evaluate_recipe(&f.recipe, &x_base).unwrap();
            let mean = data.mean().unwrap_or(0.0);
            let std = data.std(0.0);
            let safe_std = if std > 1e-9 { std } else { 1.0 };
            
            pool_means.push(mean);
            pool_stds.push(safe_std);
            
            for i in 0..n_samples {
                p_mat[(i, j)] = (data[i] - mean) / safe_std;
            }
        }

        // Gram Matrix G = P^T * P and Vector V = P^T * y_centered
        let g_mat = &p_mat.transpose() * &p_mat;
        
        let y_vec_centered = DVector::from_vec(y_centered.to_vec());
        let pty_vec = &p_mat.transpose() * &y_vec_centered;

        use itertools::Itertools;
        let combinations: Vec<Vec<usize>> = (0..m_pool).combinations(term_idx).collect();
        let total_combos = combinations.len();
        
        let report_step = (total_combos / 10).max(1);
        let atomic_counter = std::sync::atomic::AtomicUsize::new(0);

        let best_in_term_opt = combinations.par_iter()
            .map(|combo_indices| {
                let step = atomic_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                
                // Extract sub-Gram matrix and sub-V
                let mut g_sub = DMatrix::zeros(term_idx, term_idx);
                let mut v_sub = DVector::zeros(term_idx);
                
                for r in 0..term_idx {
                    v_sub[r] = pty_vec[combo_indices[r]];
                    for c in 0..term_idx {
                        g_sub[(r, c)] = g_mat[(combo_indices[r], combo_indices[c])];
                    }
                }

                // Solve G_sub * w = V_sub
                let res_w = if let Some(chol) = g_sub.clone().cholesky() {
                    Some(chol.solve(&v_sub))
                } else {
                    g_sub.clone().svd(true, true).solve(&v_sub, 1e-9).ok()
                };

                let (rmse, coeffs_real, intercept) = match res_w {
                    Some(w) => {
                        // SSE = y_sum_sq - w^T * V_sub
                        let sse = (y_sum_sq - w.dot(&v_sub)).max(0.0);
                        let rmse = (sse / n_samples as f64).sqrt();
                        
                        let mut coeffs = Vec::with_capacity(term_idx);
                        let mut icpt = y_mean;
                        for i in 0..term_idx {
                            let idx = combo_indices[i];
                            let c = w[i] / pool_stds[idx];
                            coeffs.push(c);
                            icpt -= c * pool_means[idx];
                        }
                        (rmse, coeffs, icpt)
                    },
                    None => (f64::INFINITY, vec![0.0; term_idx], 0.0)
                };
                
                if step % report_step == 0 {
                    let r2 = 1.0 - (rmse * rmse) / y_var;
                    println!("  Progress: {:>3}% ({}/{}), Current Best R2: {:.6}", 
                             (step * 100) / total_combos, step, total_combos, r2);
                }
                
                let combo_features: Vec<&Feature> = combo_indices.iter().map(|&i| &final_model_pool[i]).collect();
                (rmse, coeffs_real, intercept, combo_features)
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        if let Some((rmse, coeffs, intercept, features)) = best_in_term_opt {
            let mut y_pred = Array1::from_elem(y.len(), intercept);
            for i in 0..features.len() {
                let data = evaluate_recipe(&features[i].recipe, &x_base).unwrap();
                y_pred = y_pred + &data * coeffs[i];
            }
            current_residual = &y - &y_pred;

            let mut eq_parts = Vec::new();
            for (i, f) in features.iter().enumerate() {
                eq_parts.push(format!("{:.6} * {}", coeffs[i], f.expr));
            }
            let eq_str = format!("{} + {:.6}", eq_parts.join(" + "), intercept);

            let r2 = 1.0 - (rmse * rmse) / y_var;
            println!("Best {}-term model: RMSE={:.6}, R2={:.6}, Eq: {}", term_idx, rmse, r2, eq_str);

            if rmse < best_global_rmse {
                best_global_rmse = rmse;
                let selected_recipes = features.iter().map(|f| f.recipe.clone()).collect();
                best_global_model = (eq_str, coeffs, intercept, selected_recipes);
            }
        }
    }

    return Python::with_gil(|py| {
        let mut selected_recipes_py = Vec::new();
        for r in &best_global_model.3 { 
            selected_recipes_py.push(r.to_py_object(py));
        }

        Ok((
            best_global_rmse, 
            best_global_model.0, 
            best_global_model.1, 
            best_global_model.2, 
            selected_recipes_py
        ))
    });
}

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
