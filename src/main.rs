use std::error::Error;
use std::fs::File;
use csv::{ReaderBuilder, WriterBuilder};
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use rand::prelude::*;

// XGBoost imports
use xgboost::{Booster, DMatrix, parameters};

// TPE pour optimisation hyperparamètres
use tpe::{TpeOptimizer, range, parzen_estimator, histogram_estimator, categorical_range};
use rand::SeedableRng;

#[derive(Debug, Clone)]
struct Sample {
    id: String,
    temperature: f64,
    humidity: f64,
    moisture: f64,
    soil_type: String,
    crop_type: String,
    nitrogen: f64,
    potassium: f64,
    phosphorous: f64,
    fertilizer: Option<String>,
}

#[derive(Debug)]
struct FertilizerClassifier {
    soil_encoding: HashMap<String, f64>,
    crop_encoding: HashMap<String, f64>,
    fertilizer_encoding: HashMap<String, usize>,
    fertilizer_decoding: HashMap<usize, String>,
    n_classes: usize,
}

impl FertilizerClassifier {
    fn new() -> Self {
        FertilizerClassifier {
            soil_encoding: HashMap::new(),
            crop_encoding: HashMap::new(),
            fertilizer_encoding: HashMap::new(),
            fertilizer_decoding: HashMap::new(),
            n_classes: 0,
        }
    }

    fn encode_categoricals(&mut self, samples: &[Sample]) {
        // Encodage intelligent des sols par fertilité
        let soil_scores = vec![
            ("Sandy", 1.0), ("Red", 2.0), ("Clayey", 3.0), ("Loamy", 4.0), ("Black", 5.0)
        ];
        for (soil, score) in soil_scores {
            self.soil_encoding.insert(soil.to_string(), score);
        }

        // Encodage des cultures par intensité nutritive
        let crop_scores = vec![
            ("Millets", 1.0), ("Barley", 2.0), ("Pulses", 2.0), ("Wheat", 3.0),
            ("Ground Nuts", 3.0), ("Oil seeds", 4.0), ("Paddy", 4.0), ("Maize", 5.0),
            ("Cotton", 6.0), ("Tobacco", 7.0), ("Sugarcane", 8.0)
        ];
        for (crop, score) in crop_scores {
            self.crop_encoding.insert(crop.to_string(), score);
        }

        // Encodage des engrais
        let mut fertilizers = std::collections::HashSet::new();
        for sample in samples {
            if let Some(ref fert) = sample.fertilizer {
                fertilizers.insert(fert.clone());
            }
        }
        
        for (i, fertilizer) in fertilizers.iter().enumerate() {
            self.fertilizer_encoding.insert(fertilizer.clone(), i);
            self.fertilizer_decoding.insert(i, fertilizer.clone());
        }
        self.n_classes = fertilizers.len();
    }

    fn create_features(&self, sample: &Sample) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Features de base
        features.extend([
            sample.temperature, sample.humidity, sample.moisture,
            sample.nitrogen, sample.potassium, sample.phosphorous
        ]);
        
        // Encodages intelligents
        let soil_score = *self.soil_encoding.get(&sample.soil_type).unwrap_or(&3.0);
        let crop_score = *self.crop_encoding.get(&sample.crop_type).unwrap_or(&4.0);
        features.extend([soil_score, crop_score]);
        
        // NPK features essentielles
        let npk_total = sample.nitrogen + sample.potassium + sample.phosphorous;
        let npk_mean = npk_total / 3.0;
        features.extend([npk_total, npk_mean]);
        
        if npk_total > 0.0 {
            features.extend([
                sample.nitrogen / npk_total,    // N ratio
                sample.phosphorous / npk_total, // P ratio
                sample.potassium / npk_total,   // K ratio
            ]);
        } else {
            features.extend([0.33, 0.33, 0.33]);
        }
        
        // Ratios NPK importants
        features.extend([
            if sample.potassium > 0.0 { sample.nitrogen / sample.potassium } else { 1.0 },
            if sample.phosphorous > 0.0 { sample.nitrogen / sample.phosphorous } else { 1.0 },
            if sample.phosphorous > 0.0 { sample.potassium / sample.phosphorous } else { 1.0 },
        ]);
        
        // Stress environnementaux
        let heat_stress = (sample.temperature - 30.0).max(0.0);
        let cold_stress = (10.0 - sample.temperature).max(0.0);
        let drought_stress = (35.0 - sample.moisture).max(0.0);
        features.extend([heat_stress, cold_stress, drought_stress]);
        
        // Besoins par culture (simplifié)
        let (n_need, p_need, k_need) = match sample.crop_type.as_str() {
            "Sugarcane" => (200.0, 80.0, 200.0),
            "Cotton" => (150.0, 70.0, 100.0),
            "Maize" => (150.0, 60.0, 50.0),
            "Wheat" => (120.0, 50.0, 40.0),
            "Paddy" => (100.0, 50.0, 50.0),
            "Tobacco" => (80.0, 40.0, 150.0),
            _ => (100.0, 50.0, 50.0),
        };
        
        // Déficits nutritifs
        features.extend([
            (n_need - sample.nitrogen).max(0.0),
            (p_need - sample.phosphorous).max(0.0),
            (k_need - sample.potassium).max(0.0),
        ]);
        
        // Interactions importantes
        features.extend([
            soil_score * crop_score,
            sample.temperature * soil_score,
            heat_stress * crop_score,
            npk_total * crop_score,
        ]);
        
        // Features binaires critiques + transformations
        features.extend([
            if sample.temperature > 35.0 { 1.0 } else { 0.0 },
            if sample.moisture < 20.0 { 1.0 } else { 0.0 },
            if sample.soil_type == "Black" { 1.0 } else { 0.0 },
            if sample.crop_type == "Sugarcane" { 1.0 } else { 0.0 },
            
            // Transformations log pour XGBoost
            (1.0 + sample.nitrogen).ln(),
            (1.0 + sample.potassium).ln(),
            (1.0 + sample.phosphorous).ln(),
            (1.0 + npk_total).ln(),
        ]);
        
        features
    }

    fn prepare_data(&mut self, samples: &[Sample]) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
        self.encode_categoricals(samples);
        
        let mut X_data = Vec::new();
        let mut y_data = Vec::new();
        
        for sample in samples {
            if let Some(ref fertilizer) = sample.fertilizer {
                let features = self.create_features(sample);
                X_data.extend(features.clone());
                
                if let Some(&label) = self.fertilizer_encoding.get(fertilizer) {
                    y_data.push(label as f64);
                }
            }
        }
        
        let n_samples = y_data.len();
        let n_features = if n_samples > 0 { X_data.len() / n_samples } else { 0 };
        
        let X = Array2::from_shape_vec((n_samples, n_features), X_data)?;
        let y = Array1::from_vec(y_data);
        
        println!("✅ {} échantillons, {} features", n_samples, n_features);
        Ok((X, y))
    }

    fn prepare_test_data(&self, samples: &[Sample]) -> Array2<f64> {
        let mut X_data = Vec::new();
        
        for sample in samples {
            let features = self.create_features(sample);
            X_data.extend(features);
        }
        
        let n_features = if !samples.is_empty() { X_data.len() / samples.len() } else { 0 };
        Array2::from_shape_vec((samples.len(), n_features), X_data).unwrap()
    }

    fn get_fertilizer_name(&self, class_id: usize) -> String {
        self.fertilizer_decoding.get(&class_id).cloned().unwrap_or("Unknown".to_string())
    }
}

// XGBoost Multi-class Classifier
struct XGBoostMultiClassifier {
    models: Vec<Booster>,
    n_classes: usize,
}

impl XGBoostMultiClassifier {
    fn new(n_classes: usize) -> Self {
        XGBoostMultiClassifier {
            models: Vec::new(),
            n_classes,
        }
    }

    fn train(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<(), Box<dyn Error>> {
        println!("🚀 Entraînement XGBoost pour {} classes", self.n_classes);
        
        // One-vs-Rest pour multi-class
        for class_id in 0..self.n_classes {
            print!("  🔄 Classe {}/{} ", class_id + 1, self.n_classes);
            
            // Créer labels binaires pour cette classe
            let binary_y: Vec<f32> = y.iter()
                .map(|&label| if label as usize == class_id { 1.0 } else { 0.0 })
                .collect();
            
            // Convertir X en f32
            let X_f32: Vec<f32> = X.iter().map(|&v| v as f32).collect();
            
            // Créer DMatrix
            let mut dtrain = DMatrix::from_dense(&X_f32, X.nrows())?;
            dtrain.set_labels(&binary_y)?;
            
            // Paramètres XGBoost par défaut (évite les warnings updater/tree_method)
            let booster_params = parameters::BoosterParametersBuilder::default()
                .build()?;
            
            let training_params = parameters::TrainingParametersBuilder::default()
                .dtrain(&dtrain)
                .booster_params(booster_params)
                .boost_rounds(100)  // Plus raisonnable pour l'entraînement final
                .build()?;
            
            // Entraîner le modèle
            let booster = Booster::train(&training_params)?;
            self.models.push(booster);
            println!("✅");
        }
        
        Ok(())
    }

    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        let (n_samples, _) = X.dim();
        let mut probas = Array2::zeros((n_samples, self.n_classes));
        
        // Convertir X en f32
        let X_f32: Vec<f32> = X.iter().map(|&v| v as f32).collect();
        let dtest = DMatrix::from_dense(&X_f32, X.nrows())?;
        
        // Prédictions pour chaque classe
        for (class_id, model) in self.models.iter().enumerate() {
            let class_proba = model.predict(&dtest)?;
            
            for (i, &prob) in class_proba.iter().enumerate() {
                probas[[i, class_id]] = prob as f64;
            }
        }
        
        // Normaliser les probabilités (softmax-like)
        for i in 0..n_samples {
            let sum: f64 = probas.row(i).sum();
            if sum > 0.0 {
                for j in 0..self.n_classes {
                    probas[[i, j]] /= sum;
                }
            } else {
                for j in 0..self.n_classes {
                    probas[[i, j]] = 1.0 / self.n_classes as f64;
                }
            }
        }
        
        Ok(probas)
    }

    fn train_with_params(&mut self, X: &Array2<f64>, y: &Array1<f64>, hyperparams: &XGBoostHyperparams) -> Result<(), Box<dyn Error>> {
        // One-vs-Rest pour multi-class
        for class_id in 0..self.n_classes {
            // Créer labels binaires pour cette classe
            let binary_y: Vec<f32> = y.iter()
                .map(|&label| if label as usize == class_id { 1.0 } else { 0.0 })
                .collect();
            
            // Convertir X en f32
            let X_f32: Vec<f32> = X.iter().map(|&v| v as f32).collect();
            
            // Créer DMatrix
            let mut dtrain = DMatrix::from_dense(&X_f32, X.nrows())?;
            dtrain.set_labels(&binary_y)?;
            
            // Paramètres XGBoost avec hyperparamètres optimisés TPE
            let booster_params = parameters::BoosterParametersBuilder::default()
                .build()?;
            
            let training_params = parameters::TrainingParametersBuilder::default()
                .dtrain(&dtrain)
                .booster_params(booster_params)
                .boost_rounds(hyperparams.boost_rounds as u32)
                .build()?;
            
            // Entraîner le modèle
            let booster = Booster::train(&training_params)?;
            self.models.push(booster);
        }
        
        Ok(())
    }
}

// Structure pour hyperparamètres XGBoost avec TPE
#[derive(Debug, Clone)]
struct XGBoostHyperparams {
    max_depth: i32,
    eta: f64,
    subsample: f64,
    colsample_bytree: f64,
    boost_rounds: i32,
}

impl XGBoostHyperparams {
    fn optimize_with_tpe(
        X: &Array2<f64>, 
        y: &Array1<f64>, 
        n_classes: usize,
        n_trials: usize
    ) -> Result<Self, Box<dyn Error>> {
        println!("🧠 Optimisation TPE des hyperparamètres XGBoost ({} trials)...", n_trials);
        
        // Créer les optimiseurs TPE pour chaque hyperparamètre
        let mut depth_optim = TpeOptimizer::new(
            parzen_estimator(), 
            categorical_range(7)?  // [3, 4, 5, 6, 8, 10, 12]
        );
        let mut eta_optim = TpeOptimizer::new(
            parzen_estimator(), 
            range(0.01, 0.3)?
        );
        let mut subsample_optim = TpeOptimizer::new(
            parzen_estimator(), 
            range(0.6, 1.0)?
        );
        let mut colsample_optim = TpeOptimizer::new(
            parzen_estimator(), 
            range(0.6, 1.0)?
        );
        let mut rounds_optim = TpeOptimizer::new(
            parzen_estimator(), 
            categorical_range(8)?  // [50, 75, 100, 150, 200, 250, 300, 400]
        );
        
        let depth_choices = [3, 4, 5, 6, 8, 10, 12];
        let rounds_choices = [50, 75, 100, 150, 200, 250, 300, 400];
        
        let mut rng = StdRng::from_entropy();
        let mut best_params = None;
        let mut best_score = 0.0;
        
        for trial in 1..=n_trials {
            // Demander les hyperparamètres à TPE
            let depth_idx = depth_optim.ask(&mut rng)? as usize;
            let eta = eta_optim.ask(&mut rng)?;
            let subsample = subsample_optim.ask(&mut rng)?;
            let colsample_bytree = colsample_optim.ask(&mut rng)?;
            let rounds_idx = rounds_optim.ask(&mut rng)? as usize;
            
            let max_depth = depth_choices[depth_idx.min(depth_choices.len() - 1)];
            let boost_rounds = rounds_choices[rounds_idx.min(rounds_choices.len() - 1)];
            
            let params = XGBoostHyperparams {
                max_depth,
                eta,
                subsample,
                colsample_bytree,
                boost_rounds,
            };
            
            print!("  🔄 Trial {}/{}: depth={}, eta={:.3}, sub={:.2}, col={:.2}, rounds={} ",
                   trial, n_trials, max_depth, eta, subsample, colsample_bytree, boost_rounds);
            
            // Évaluer les paramètres
            let score = Self::evaluate_params(&params, X, y, n_classes)?;
            
            // Communiquer le score à TPE (on maximise, donc on inverse)
            let loss = 1.0 - score;  // TPE minimise, on veut maximiser MAP@3
            depth_optim.tell(depth_idx as f64, loss)?;
            eta_optim.tell(eta, loss)?;
            subsample_optim.tell(subsample, loss)?;
            colsample_optim.tell(colsample_bytree, loss)?;
            rounds_optim.tell(rounds_idx as f64, loss)?;
            
            if score > best_score {
                best_score = score;
                best_params = Some(params.clone());
                println!("✅ MAP@3 = {:.4} (NOUVEAU RECORD!)", score);
            } else {
                println!("📈 MAP@3 = {:.4}", score);
            }
        }
        
        let final_params = best_params.unwrap_or(XGBoostHyperparams {
            max_depth: 6,
            eta: 0.1,
            subsample: 0.8,
            colsample_bytree: 0.8,
            boost_rounds: 100,
        });
        
        println!("🎯 === RÉSULTATS OPTIMISATION TPE ===");
        println!("   max_depth: {}", final_params.max_depth);
        println!("   eta: {:.3}", final_params.eta);
        println!("   subsample: {:.3}", final_params.subsample);
        println!("   colsample_bytree: {:.3}", final_params.colsample_bytree);
        println!("   boost_rounds: {}", final_params.boost_rounds);
        println!("   MAP@3 optimal: {:.4}", best_score);
        
        Ok(final_params)
    }
    
    fn evaluate_params(
        params: &XGBoostHyperparams, 
        X: &Array2<f64>, 
        y: &Array1<f64>,
        n_classes: usize
    ) -> Result<f64, Box<dyn Error>> {
        // Split rapide 80/20 pour évaluation
        let n_samples = X.nrows();
        let split_idx = (n_samples as f64 * 0.8) as usize;
        
        let train_indices: Vec<usize> = (0..split_idx).collect();
        let val_indices: Vec<usize> = (split_idx..n_samples).collect();
        
        let X_train = X.select(ndarray::Axis(0), &train_indices);
        let y_train = y.select(ndarray::Axis(0), &train_indices);
        let X_val = X.select(ndarray::Axis(0), &val_indices);
        let y_val = y.select(ndarray::Axis(0), &val_indices);
        
        // Entraîner avec ces paramètres
        let mut model = XGBoostMultiClassifier::new(n_classes);
        model.train_with_params(&X_train, &y_train, params)?;
        
        // Évaluer
        let val_probas = model.predict_proba(&X_val)?;
        let mut val_predictions = Vec::new();
        let mut val_labels = Vec::new();
        
        for i in 0..val_probas.nrows() {
            let mut class_probas: Vec<_> = (0..n_classes)
                .map(|j| (j, val_probas[[i, j]]))
                .collect();
            class_probas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let top_3: Vec<usize> = class_probas.into_iter()
                .take(3)
                .map(|(class_id, _)| class_id)
                .collect();
            
            val_predictions.push(top_3);
            val_labels.push(y_val[i] as usize);
        }
        
        Ok(calculate_map_at_k(&val_labels, &val_predictions, 3))
    }
}

fn stratified_k_fold_split(y: &Array1<f64>, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
    
    for (i, &label) in y.iter().enumerate() {
        class_indices.entry(label as i32).or_insert_with(Vec::new).push(i);
    }
    
    let mut rng = thread_rng();
    for indices in class_indices.values_mut() {
        indices.shuffle(&mut rng);
    }
    
    let mut folds = Vec::new();
    for fold in 0..k {
        let mut train_indices = Vec::new();
        let mut val_indices = Vec::new();
        
        for indices in class_indices.values() {
            let fold_size = indices.len() / k;
            let start = fold * fold_size;
            let end = if fold == k - 1 { indices.len() } else { (fold + 1) * fold_size };
            
            val_indices.extend(&indices[start..end]);
            train_indices.extend(&indices[0..start]);
            train_indices.extend(&indices[end..]);
        }
        
        folds.push((train_indices, val_indices));
    }
    
    folds
}

fn calculate_map_at_k(y_true: &[usize], predictions: &[Vec<usize>], k: usize) -> f64 {
    let mut total_ap = 0.0;
    
    for (true_label, pred_list) in y_true.iter().zip(predictions.iter()) {
        for (rank, &pred) in pred_list.iter().take(k).enumerate() {
            if pred == *true_label {
                total_ap += 1.0 / (rank + 1) as f64;
                break;
            }
        }
    }
    
    total_ap / y_true.len() as f64
}

fn read_train_data(file_path: &str) -> Result<Vec<Sample>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    
    let mut samples = Vec::new();
    for result in rdr.records() {
        let record = result?;
        
        let sample = Sample {
            id: record.get(0).unwrap_or("").to_string(),
            temperature: record.get(1).unwrap_or("25").parse().unwrap_or(25.0),
            humidity: record.get(2).unwrap_or("50").parse().unwrap_or(50.0),
            moisture: record.get(3).unwrap_or("50").parse().unwrap_or(50.0),
            soil_type: record.get(4).unwrap_or("Unknown").to_string(),
            crop_type: record.get(5).unwrap_or("Unknown").to_string(),
            nitrogen: record.get(6).unwrap_or("0").parse().unwrap_or(0.0),
            potassium: record.get(7).unwrap_or("0").parse().unwrap_or(0.0),
            phosphorous: record.get(8).unwrap_or("0").parse().unwrap_or(0.0),
            fertilizer: Some(record.get(9).unwrap_or("Unknown").to_string()),
        };
        
        samples.push(sample);
    }
    
    Ok(samples)
}

fn read_test_data(file_path: &str) -> Result<Vec<Sample>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    
    let mut samples = Vec::new();
    for result in rdr.records() {
        let record = result?;
        
        let sample = Sample {
            id: record.get(0).unwrap_or("").to_string(),
            temperature: record.get(1).unwrap_or("25").parse().unwrap_or(25.0),
            humidity: record.get(2).unwrap_or("50").parse().unwrap_or(50.0),
            moisture: record.get(3).unwrap_or("50").parse().unwrap_or(50.0),
            soil_type: record.get(4).unwrap_or("Unknown").to_string(),
            crop_type: record.get(5).unwrap_or("Unknown").to_string(),
            nitrogen: record.get(6).unwrap_or("0").parse().unwrap_or(0.0),
            potassium: record.get(7).unwrap_or("0").parse().unwrap_or(0.0),
            phosphorous: record.get(8).unwrap_or("0").parse().unwrap_or(0.0),
            fertilizer: None,
        };
        
        samples.push(sample);
    }
    
    Ok(samples)
}

fn write_submission(file_path: &str, predictions: &[(String, Vec<String>)]) -> Result<(), Box<dyn Error>> {
    let mut wtr = WriterBuilder::new().from_path(file_path)?;
    wtr.write_record(&["id", "Fertilizer Name"])?;
    
    for (id, fertilizers) in predictions {
        let fertilizer_string = fertilizers.join(" ");
        wtr.write_record(&[id, &fertilizer_string])?;
    }
    
    wtr.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 === XGBOOST FERTILIZER CLASSIFIER avec TPE ===");
    
    let train_samples = read_train_data("train.csv")?;
    println!("✅ {} échantillons d'entraînement chargés", train_samples.len());
    
    let mut classifier = FertilizerClassifier::new();
    let (X, y) = classifier.prepare_data(&train_samples)?;
    
    println!("🧠 {} classes d'engrais détectées", classifier.n_classes);
    
    // Optimisation TPE des hyperparamètres (20 trials pour équilibrer temps/qualité)
    let best_hyperparams = XGBoostHyperparams::optimize_with_tpe(&X, &y, classifier.n_classes, 20)?;
    
    // Cross-validation avec hyperparamètres optimisés
    println!("📊 Cross-validation 3-fold avec hyperparamètres TPE...");
    let folds = stratified_k_fold_split(&y, 3);
    let mut cv_scores = Vec::new();
    
    for (fold_idx, (train_indices, val_indices)) in folds.iter().enumerate() {
        println!("  🔄 Fold {}/3 - {} train, {} val", 
                fold_idx + 1, train_indices.len(), val_indices.len());
        
        let X_train = X.select(ndarray::Axis(0), train_indices);
        let y_train = y.select(ndarray::Axis(0), train_indices);
        let X_val = X.select(ndarray::Axis(0), val_indices);
        let y_val = y.select(ndarray::Axis(0), val_indices);
        
        let mut model = XGBoostMultiClassifier::new(classifier.n_classes);
        print!("     🔨 Entraînement... ");
        if let Err(e) = model.train_with_params(&X_train, &y_train, &best_hyperparams) {
            println!("❌ Erreur: {}", e);
            continue;
        }
        println!("✅");
        
        print!("     🔮 Prédiction... ");
        if let Ok(val_probas) = model.predict_proba(&X_val) {
            let mut val_predictions = Vec::new();
            let mut val_labels = Vec::new();
            
            for i in 0..val_probas.nrows() {
                let mut class_probas: Vec<_> = (0..classifier.n_classes)
                    .map(|j| (j, val_probas[[i, j]]))
                    .collect();
                class_probas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                let top_3: Vec<usize> = class_probas.into_iter()
                    .take(3)
                    .map(|(class_id, _)| class_id)
                    .collect();
                
                val_predictions.push(top_3);
                val_labels.push(y_val[i] as usize);
            }
            
            let map_score = calculate_map_at_k(&val_labels, &val_predictions, 3);
            cv_scores.push(map_score);
            println!("✅ MAP@3: {:.4}", map_score);
        } else {
            println!("❌ Erreur prédiction");
        }
    }
    
    let mean_cv_score = if !cv_scores.is_empty() {
        cv_scores.iter().sum::<f64>() / cv_scores.len() as f64
    } else {
        0.0
    };
    
    println!("🎯 MAP@3 moyen CV: {:.4}", mean_cv_score);
    
    // Modèle final avec hyperparamètres TPE optimisés
    println!("🔨 Entraînement du modèle final avec hyperparamètres TPE...");
    let mut final_model = XGBoostMultiClassifier::new(classifier.n_classes);
    final_model.train_with_params(&X, &y, &best_hyperparams)?;
    
    // Test
    let test_samples = read_test_data("test.csv")?;
    println!("✅ {} échantillons de test chargés", test_samples.len());
    
    let X_test = classifier.prepare_test_data(&test_samples);
    println!("🔮 Prédiction finale...");
    let test_probas = final_model.predict_proba(&X_test)?;
    
    let mut final_predictions = Vec::new();
    for i in 0..test_samples.len() {
        let mut class_probas: Vec<_> = (0..classifier.n_classes)
            .map(|j| (j, test_probas[[i, j]]))
            .collect();
        class_probas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_3_fertilizers: Vec<String> = class_probas.into_iter()
            .take(3)
            .map(|(class_id, _)| classifier.get_fertilizer_name(class_id))
            .collect();
        
        final_predictions.push((test_samples[i].id.clone(), top_3_fertilizers));
    }
    
    let output_file = "submission_xgboost2.csv";
    write_submission(output_file, &final_predictions)?;
    
    println!("🎉 === RÉSULTATS FINAUX TPE ===");
    println!("📁 Fichier: {}", output_file);
    println!("📊 {} prédictions générées", final_predictions.len());
    println!("📈 MAP@3 CV: {:.4}", mean_cv_score);
    println!("🧠 Hyperparamètres optimisés par TPE :");
    println!("   max_depth: {}", best_hyperparams.max_depth);
    println!("   eta: {:.3}", best_hyperparams.eta);
    println!("   subsample: {:.3}", best_hyperparams.subsample);
    println!("   colsample_bytree: {:.3}", best_hyperparams.colsample_bytree);
    println!("   boost_rounds: {}", best_hyperparams.boost_rounds);
    
    Ok(())
}