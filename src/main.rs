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
        Box::new(DenseLayer::<INPUT_NEURON_COUNT>::new(Activation::Relu)),
        //Box::new(DenseLayer::<INPUT_NEURON_COUNT>::new(Activation::Relu)),
        Box::new(DenseLayer::<1>::new(Activation::Sigmoid)),
        Box::new(OutputLayer::new(1)),
    ]);
    let dataset = read_inputs();
    println!(
        "{:?}",
        fit(&mut model, &MeanSquaredError::<1>, &dataset, 32, 10.0, 100)
    );

    //println!("{model:?}");
}
