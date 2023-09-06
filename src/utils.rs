use std::collections::{HashMap, HashSet};

use ndarray::{Array2, Array1};

use polars::prelude::*;
use linfa::prelude::*;

use linfa::traits::{Fit, Predict};
use linfa_datasets::winequality;
use linfa_logistic::LogisticRegression;
use linfa_trees::{DecisionTree, SplitQuality};

pub fn run_polar_example() {
    // Create a DataFrame
    let df = DataFrame::new(vec![
        Series::new("Age", vec![25, 30, 22]),
        Series::new("Name", vec!["Alice", "Bob", "Charlie"])
    ]).expect("Failed to created df.");

    // Filter rows where Age is greater than 25
    let filtered = df.filter(&df["Age"].gt(25).unwrap()).unwrap();
    println!("{:?}", filtered);
}

pub fn run_linfa_example() {
    let (train, valid) = winequality()
        .map_targets(|x| if *x > 6 { "good" } else { "bad" })
        .split_with_ratio(0.9);

    let model = LogisticRegression::default().max_iterations(150).fit(&train).unwrap();
    let pred = model.predict(&valid);
    let cm = pred.confusion_matrix(&valid).unwrap();
    println!("{:?}", cm);
}

pub fn load_dataset() -> PolarsResult<DataFrame> {
    CsvReader::from_path("./data/iris.csv")?
        .infer_schema(None)
        .has_header(true)
        .finish()
}

fn dataframe_into_ndarray(df: &DataFrame) -> Array2<f64>{
    let mut rows = vec![];

    for row in df.iter() {
        let values: Vec<f64> = row.f64().unwrap().into_iter()
            .map(|opt_value| opt_value.unwrap_or_default())
            .collect();
        rows.push(values);
    }
    
    Array2::from_shape_vec((df.height() as usize, df.width() as usize), rows.concat()).unwrap()
}

pub fn run_end_to_end_example() {
    // Load the dataset and show the first 5 rows
    let df = load_dataset().unwrap();
    println!("{:?}", df.head(Some(5)));

    // Assume that the target columns is called "target"
    let targets_series = df.column("target").unwrap().clone();
    let features_df = df.drop("target").unwrap();

    // Convert feature DataFrame into 2D ndarray
    let features_array: Array2<f64> = dataframe_into_ndarray(&features_df);

    // Convert target Series into 1D ndarray
    let targets_array: Array1<String> = Array1::from_vec(targets_series.utf8().unwrap().clone().into_iter().filter_map(|opt_s| opt_s.map(|s| s.to_string())).collect());

    println!("{}", targets_array);
    // Create a linfa Dataset
    let dataset = Dataset::new(features_array, targets_array);

    // Split the dataset into train and test
    let (train, test) = dataset.split_with_ratio(0.9);

    // Train the model
    let gini_model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(&train)
        .unwrap();

    // Evaluate the model
    let gini_pred_y = gini_model.predict(&test);
    let cm = gini_pred_y.confusion_matrix(&test);
    println!("Test accuracy with Gini criterion: {:.2}%", 100.0 * cm.unwrap().accuracy());
    
}