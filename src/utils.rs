use polars::prelude::*;

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