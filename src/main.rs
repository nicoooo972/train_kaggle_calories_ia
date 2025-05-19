use std::error::Error;
use std::fs::File;
use csv::{ReaderBuilder, WriterBuilder};
use linfa_linear::LinearRegression; // Import du modèle de régression linéaire
use linfa::prelude::*; // Import des traits et types utiles pour Linfa
use ndarray::{ Array1, Array2}; // Pour créer facilement des tableaux multidimensionnels et utiliser la macro s!


fn read_csv_and_prepare_data(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let headers = rdr.headers()?.clone();
    let mut id_col_index: Option<usize> = None;
    let mut calories_col_index: Option<usize> = None;
    let mut sex_col_index: Option<usize> = None;

    for (i, header) in headers.iter().enumerate() {
        if header == "id" {
            id_col_index = Some(i);
        } else if header == "Calories" {
            calories_col_index = Some(i);
        } else if header == "Sex" {
            sex_col_index = Some(i);
        }
    }

    let calories_col_index = calories_col_index.ok_or("Colonne 'Calories' non trouvée")?;

    let mut x_data: Vec<Vec<f64>> = Vec::new();
    let mut y_data: Vec<f64> = Vec::new();

    for (row_idx, result) in rdr.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Avertissement : Ligne {} ignorée en raison d'une erreur de lecture de l'enregistrement : {}", row_idx + 1, e);
                continue;
            }
        };
        let mut current_x_row: Vec<f64> = Vec::new();
        for (i, field) in record.iter().enumerate() {
            if Some(i) == id_col_index || Some(i) == sex_col_index { // Ignorer id et Sex
                continue;
            }
            if i == calories_col_index {
                match field.trim().parse::<f64>() {
                    Ok(val) => y_data.push(val),
                    Err(e) => {
                        eprintln!("Avertissement (train) : Ligne {}, Colonne 'Calories', valeur '{}' non convertible. Remplacée par NaN. Erreur: {}", row_idx + 1, field, e);
                        y_data.push(f64::NAN);
                    }
                }
            } else {
                match field.trim().parse::<f64>() {
                    Ok(val) => current_x_row.push(val),
                    Err(e) => {
                        eprintln!("Avertissement (train) : Ligne {}, Colonne {}, valeur '{}' non convertible. Remplacée par NaN. Erreur: {}", row_idx + 1, headers.get(i).unwrap_or(&"Inconnue".to_string()), field, e);
                        current_x_row.push(f64::NAN);
                    }
                }
            }
        }
        if !current_x_row.is_empty() {
             x_data.push(current_x_row);
        }
    }

    if x_data.is_empty() {
        return Err("Aucune donnée X n'a été lue.".into());
    }
    let n_rows = x_data.len();
    let n_cols = x_data[0].len();
    let x_flat: Vec<f64> = x_data.into_iter().flatten().collect();
    let x_array = Array2::from_shape_vec((n_rows, n_cols), x_flat)?;

    let y_array = Array1::from(y_data);

    Ok((x_array, y_array))
}

fn read_test_csv_and_prepare_data(file_path: &str) -> Result<(Array2<f64>, Vec<String>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let headers = rdr.headers()?.clone();
    let mut id_col_index_opt: Option<usize> = None;
    let mut sex_col_index_opt: Option<usize> = None;

    for (i, header) in headers.iter().enumerate() {
        if header == "id" {
            id_col_index_opt = Some(i);
        } else if header == "Sex" {
            sex_col_index_opt = Some(i);
        }
    }

    let id_col_index = id_col_index_opt.ok_or("Colonne 'id' non trouvée dans test.csv")?;

    let mut x_test_data: Vec<Vec<f64>> = Vec::new();
    let mut ids: Vec<String> = Vec::new();

    for (row_idx, result) in rdr.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Avertissement (test) : Ligne {} ignorée : {}", row_idx + 1, e);
                continue;
            }
        };
        
        let mut current_x_row: Vec<f64> = Vec::new();
        let mut current_id: Option<String> = None;

        for (i, field) in record.iter().enumerate() {
            if i == id_col_index {
                current_id = Some(field.to_string());
                continue; 
            }
            if Some(i) == sex_col_index_opt {
                continue; 
            }
            // Pour toutes les autres colonnes, on essaie de parser en f64
            match field.trim().parse::<f64>() {
                Ok(val) => current_x_row.push(val),
                Err(e) => {
                    eprintln!(
                        "Avertissement (test) : Ligne {}, Colonne {}, valeur '{}' non convertible. Remplacée par NaN. Erreur: {}",
                        row_idx + 1,
                        headers.get(i).unwrap_or(&"Inconnue".to_string()),
                        field,
                        e
                    );
                    current_x_row.push(f64::NAN);
                }
            }
        }
        if let Some(id_val) = current_id {
            if !current_x_row.is_empty() {
                x_test_data.push(current_x_row);
                ids.push(id_val);
            } else {
                 eprintln!("Avertissement (test) : Ligne {} pour id {} n'a pas de données X valides après filtrage.", row_idx + 1, id_val);
            }
        } else {
            eprintln!("Avertissement (test) : Ligne {} n'a pas d'id. Ignorée.", row_idx + 1);
        }
    }

    if x_test_data.is_empty() {
        return Err("Aucune donnée X de test n'a été lue.".into());
    }
    let n_rows = x_test_data.len();
    let n_cols = x_test_data[0].len(); 
    let x_flat: Vec<f64> = x_test_data.into_iter().flatten().collect();
    let x_array = Array2::from_shape_vec((n_rows, n_cols), x_flat)?;

    Ok((x_array, ids))
}

fn write_predictions_to_csv(file_path: &str, ids: &[String], predictions: &Array1<f64>) -> Result<(), Box<dyn Error>> {
    let mut wtr = WriterBuilder::new().from_path(file_path)?;
    wtr.write_record(&["id", "Calories"])?;

    for (id, pred) in ids.iter().zip(predictions.iter()) {
        wtr.write_record(&[id, &pred.to_string()])?;
    }

    wtr.flush()?;
    Ok(())
}


fn main() {
    match read_csv_and_prepare_data("train.csv") {
        Ok((x_train, y_train)) => {
            let dataset_train = Dataset::new(x_train.clone(), y_train.clone()); // Cloner si x_train est réutilisé
            println!("Entraînement du modèle...");
            let model = LinearRegression::new().fit(&dataset_train).expect("Échec de l'entraînement du modèle");
            println!("Modèle entraîné.");

            println!("Lecture et préparation de test.csv...");
            match read_test_csv_and_prepare_data("test.csv") {
                Ok((x_test, ids_test)) => {
                    println!("Prédiction sur les données de test...");
                    if x_train.ncols() != x_test.ncols() {
                        eprintln!(
                            "Erreur : Le nombre de caractéristiques dans train.csv ({}) ne correspond pas à test.csv ({}). Vérifiez le prétraitement.",
                            x_train.ncols(), x_test.ncols()
                        );
                        return;
                    }

                    let mut y_pred_test = model.predict(&x_test);
                    println!("Prédictions terminées.");

                    for pred_val in y_pred_test.iter_mut() {
                        if *pred_val < 0.0 {
                            *pred_val = 0.0;
                        }
                    }

                    let output_file = "submission.csv";
                    println!("Écriture des prédictions dans {}...", output_file);
                    if let Err(e) = write_predictions_to_csv(output_file, &ids_test, &y_pred_test) {
                        eprintln!("Erreur lors de l'écriture du fichier de soumission : {}", e);
                    } else {
                        println!("Fichier de soumission {} créé avec succès.", output_file);
                    }
                }
                Err(e) => {
                    eprintln!("Erreur lors de la lecture ou préparation de test.csv : {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Erreur lors de la préparation des données d'entraînement : {}", e);
        }
    }
}
