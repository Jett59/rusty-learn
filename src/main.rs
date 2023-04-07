use neuron::{Activation, DenseLayer, Model};

use crate::neuron::OutputLayer;

mod neuron;

fn main() {
    let model = Model::new(vec![
        Box::new(DenseLayer::new(10, Activation::Relu)),
        Box::new(OutputLayer::new(5)),
    ]);
    let input = [2.0; 10];
    println!("{:?}", model.evaluate(&input));
}
