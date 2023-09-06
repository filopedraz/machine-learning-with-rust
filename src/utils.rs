use polars::prelude::*;

use linfa::prelude::*;
use linfa::traits::{Fit, Predict};
use linfa_datasets::winequality;
use linfa_logistic::LogisticRegression;

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