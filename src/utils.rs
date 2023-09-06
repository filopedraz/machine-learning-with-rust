use std::collections::HashMap;

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

    // Encode the target from string to int
    // let target_mapping = df.column("target").unwrap().unique().unwrap();
    // let target_mapping_str: Vec<&str> = target_mapping.utf8().unwrap().into_iter().flatten().collect();
    
    // let mut new_target_values: Vec<i32> = Vec::new();
    // for value in df.column("target").unwrap().utf8().unwrap().into_iter() {
    //     match value {
    //         Some(v) if v == target_mapping_str[0] => new_target_values.push(0),
    //         Some(v) if v == target_mapping_str[1] => new_target_values.push(1),
    //         Some(v) if v == target_mapping_str[2] => new_target_values.push(2),
    //         _ => panic!("Unexpected target value!"),
    //     }
    // }
    // let new_target = Series::new("target", new_target_values);
    // let df: &mut DataFrame = df.with_column(new_target).unwrap();
    // println!("{:?}", df.head(Some(5))); 

    // Assume that the target columns is called "target"
    let targets_series = df.column("target").unwrap().clone();
    let features_df = df.drop("target").unwrap();

    // Convert feature DataFrame into 2D ndarray
    let features_array: Array2<f64> = dataframe_into_ndarray(&features_df);

    // Convert target Series into 1D ndarray
    // let targets_vec: Vec<i32> = targets_series.utf8().unwrap().into_iter()
    //     .map(|opt_value| opt_value.unwrap_or_default())
    //     .collect();
    // let targets_array: Array1<i32> = Array1::from_vec(targets_vec);

    // Convert the Series of str into Vec<String>
    let targets_array: Array1<String> = Array1::from_vec(targets_series.utf8().unwrap().clone().into_iter().filter_map(|opt_s| opt_s.map(|s| s.to_string())).collect());

    println!("{:?}", targets_array);
    // Create a linfa Dataset
    let dataset = Dataset::new(features_array, targets_array);

    // Split the dataset into train and test
    let (train, test) = dataset.split_with_ratio(0.9);

    // Train the model
    let model = LogisticRegression::default().max_iterations(150).fit(&train).unwrap();

    // Evaluate the model
    let pred = model.predict(&test);
    let cm = pred.confusion_matrix(&test).unwrap();
    println!("{:?}", cm);
    
}