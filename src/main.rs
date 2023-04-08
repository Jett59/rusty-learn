use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use data::{flatten, one_hot_encode, pad_end};
use neuron::{Activation, DenseLayer, Model};

use crate::{
    neuron::OutputLayer,
    optimizer::{fit, DatasetItem, MeanSquaredError},
};

mod data;
mod neuron;
mod optimizer;

const INPUT_LETTER_COUNT: usize = 20;

fn read_inputs() -> Vec<DatasetItem> {
    let file = File::open("spelling.txt").expect("Missing spelling.txt");
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|line| {
            line.unwrap()
                .split_whitespace()
                .map(|string| string.to_string())
                .collect::<Vec<_>>()
        })
        .map(|terms| (terms[0].clone(), terms[1].parse::<bool>().unwrap()))
        .map(|(word, correct)| {
            (
                pad_end(
                    flatten(one_hot_encode(
                        word.as_bytes(),
                        b"abcdefghijklmnopqrstuvwxyz",
                    )),
                    INPUT_LETTER_COUNT * 26,
                ),
                vec![if correct { 1.0 } else { 0.0 }],
            )
        })
        .map(|(input, label)| DatasetItem { input, label })
        .collect()
}

fn main() {
    let mut model = Model::new(vec![
        Box::new(DenseLayer::new(INPUT_LETTER_COUNT * 26, Activation::Relu)),
        Box::new(DenseLayer::new(INPUT_LETTER_COUNT * 26, Activation::Relu)),
        Box::new(DenseLayer::new(1, Activation::Sigmoid)),
        Box::new(OutputLayer::new(1)),
    ]);
    let dataset = read_inputs();
    println!(
        "{:?}",
        fit(&mut model, &MeanSquaredError, &dataset, 0.01, 40000)
    );

    println!("{model:?}");
    let input = [2.0; 5];
    println!("{:?}", model.evaluate(&input));
}
