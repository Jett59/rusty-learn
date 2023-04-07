use rand::{thread_rng, Rng, RngCore};

use crate::neuron::Model;

#[derive(Debug, Clone)]
pub struct ModelStats {
    loss: f64,
}

#[derive(Debug, Clone)]
pub struct DatasetItem {
    pub input: Vec<f64>,
    pub label: Vec<f64>,
}

pub trait LossFunction {
    fn loss(&self, predicted: &[f64], actual: &[f64]) -> f64;
}

pub fn mean_squared_error(data: Vec<DatasetItem>) -> impl LossFunction {
    struct MeanSquaredError {
        data: Vec<DatasetItem>,
    }
    impl LossFunction for MeanSquaredError {
        fn loss(&self, predicted: &[f64], actual: &[f64]) -> f64 {
            let mut loss = 0.0;
            for (predicted, actual) in predicted.iter().zip(actual.iter()) {
                loss += (predicted - actual).powi(2);
            }
            loss / predicted.len() as f64
        }
    }
    MeanSquaredError { data }
}

fn calculate_loss(
    model: &mut Model,
    dataset: &[DatasetItem],
    loss_function: &dyn LossFunction,
    first_layer: usize,
    inputs: &[Vec<f64>],
) -> f64 {
    let mut loss = 0.0;
    for (item, input) in dataset.iter().zip(inputs.iter()) {
        let predicted = model.evaluate_from(first_layer, input);
        loss += loss_function.loss(&predicted, &item.label);
    }
    loss / dataset.len() as f64
}

pub fn fit(
    model: &mut Model,
    loss_function: &dyn LossFunction,
    dataset: &[DatasetItem],
    learning_rate: f64,
    epochs: usize,
) -> ModelStats {
    let mut loss = f64::INFINITY;
    for _epoch in 1..=epochs {
        // We could calculate the derivative and do all that.
        // We could also approximate the derivative by doing a finite difference.
        // We'll do that because it is simpler.
        loss = calculate_loss(
            model,
            dataset,
            loss_function,
            0,
            &dataset
                .iter()
                .map(|item| item.input.clone())
                .collect::<Vec<Vec<f64>>>(),
        );
        println!("Loss: {loss}");
        for layer_index in 0..model.layers().len() {
            // Cache the calculations for the previous layers to make it faster.
            let output_from_previous_layers = dataset
                .iter()
                .map(|item| model.evaluate_range(0, layer_index, &item.input))
                .collect::<Vec<Vec<f64>>>();
            let trainable_parameter_count = model.layers()[layer_index].trainable_parameter_count();
            if trainable_parameter_count > 0 {
                let mut improved = false;
                while !improved {
                    let selected_parameter_index =
                        thread_rng().next_u64() % trainable_parameter_count as u64;
                    let change_amount = thread_rng().gen_range(-1.0..1.0);
                    let parameter = model.layers_mut()[layer_index]
                        .trainable_parameter(selected_parameter_index as usize);
                    let old_value = *parameter;
                    *parameter += change_amount;
                    let new_loss = calculate_loss(
                        model,
                        dataset,
                        loss_function,
                        layer_index,
                        &output_from_previous_layers,
                    );
                    // We have to get the reference to the parameter again here so Rust knows that we aren't changing the variable while evaluating the model.
                    let parameter = model.layers_mut()[layer_index]
                        .trainable_parameter(selected_parameter_index as usize);
                    if new_loss < loss {
                        loss = new_loss;
                        improved = true;
                    } else {
                        *parameter = old_value;
                    }
                }
            }
        }
    }
    ModelStats { loss }
}
