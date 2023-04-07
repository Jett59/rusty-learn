use neuron::{Activation, DenseLayer, Model};

use crate::{
    neuron::OutputLayer,
    optimizer::{fit, mean_squared_error, DatasetItem},
};

mod neuron;
mod optimizer;

fn main() {
    let mut model = Model::new(vec![
        Box::new(DenseLayer::new(5, Activation::Relu)),
        Box::new(OutputLayer::new(5)),
    ]);
    let numbers = [
        1.0,
        2.0,
        0.5,
        6.0,
        7.0,
        5.0,
        13.0,
        4.0,
        3.0,
        50.0,
        16.0,
        8.0,
        4.0,
        12.0,
        1.0,
        14.0,
        3.0,
        6.0,
        0.25,
        10.0,
        6.5,
        18.75,
        100.0,
        1024.0,
        8.0,
        1.0,
        9.0,
        3.0,
        1.0,
        62.0,
        0.01,
        3.73,
        1024.08,
        0.0,
        16.37825,
        10.0,
        128.0,
        65536.0,
        83.32385,
        19234.8,
        1920.0,
        1080.0,
        24.0,
        54.8,
        19.28,
        57.1,
        833.9,
        1048576.65536,
        10293857.012,
        1234.4321,
        1234567890.3,
        19385.1,
        1.8,
        8.4,
        4.8,
        2.4,
        1.2,
        6.0,
        3.0,
        12.38,
        0.0,
        1045.324,
        0.123,
        10547.8,
        10983.543,
    ];
    let mut dataset = Vec::new();
    // Group into groups of 5.
    for i in 0..numbers.len() / 5 {
        let mut input = vec![0.0; 5];
        let mut output = vec![0.0; 5];
        for j in 0..5 {
            input[j] = numbers[i * 5 + j];
            output[j] = numbers[i * 5 + j] * 2.0;
        }
        dataset.push(DatasetItem {
            input,
            label: output,
        });
    }
    println!(
        "{:?}",
        fit(
            &mut model,
            &mean_squared_error(dataset.to_vec()),
            &dataset,
            0.01,
            40000
        )
    );

    println!("{model:?}");
    let input = [2.0; 5];
    println!("{:?}", model.evaluate(&input));
}
