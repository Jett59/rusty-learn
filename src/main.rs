use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use data::{flatten, one_hot_encode, pad_end};
use neuron::{DenseLayer, Model};

use crate::{
    neuron::{activation, OutputLayer},
    optimizer::{calculate_loss, fit, BinaryCrossEntropy, DatasetItem},
};

mod data;
mod neuron;
mod optimizer;
mod util;

const INPUT_LETTER_COUNT: usize = 20;
const INPUT_NEURON_COUNT: usize = INPUT_LETTER_COUNT * 26;

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
                    INPUT_NEURON_COUNT,
                ),
                vec![if correct { 1.0 } else { 0.0 }],
            )
        })
        .map(|(input, label)| DatasetItem { input, label })
        .collect()
}

fn main() {
    let mut model = Model::new(vec![
        Box::new(DenseLayer::<INPUT_NEURON_COUNT, activation::Relu>::new()),
        //Box::new(DenseLayer::<INPUT_NEURON_COUNT, activation::Relu>::new()),
        Box::new(OutputLayer::<activation::Sigmoid>::new(1)),
    ]);
    let dataset = read_inputs();
    let training_data = dataset
        .iter()
        .take(dataset.len() / 5 * 4)
        .cloned()
        .collect::<Vec<_>>();
    let validation_data = dataset
        .iter()
        .skip(dataset.len() / 5 * 4)
        .cloned()
        .collect::<Vec<_>>();
    println!(
        "{:?}",
        fit(
            &mut model,
            &BinaryCrossEntropy::<1>,
            &training_data,
            32,
            0.01,
            1
        )
    );
    let mut execution_context = model.create_execution_context();
    let validation_loss = calculate_loss(
        &mut model,
        &validation_data,
        &BinaryCrossEntropy::<1>,
        0,
        &validation_data
            .iter()
            .map(|item| item.input.clone())
            .collect::<Vec<_>>(),
        &mut execution_context,
    );
    println!("Validation loss: {}", validation_loss);
}
