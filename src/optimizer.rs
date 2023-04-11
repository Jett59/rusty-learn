use std::{
    io::{stdout, Write},
    time::Instant,
};

use crate::neuron::{ExecutionContext, Model};

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
    fn loss(&self, expected: &[f64], actual: &[f64]) -> f64;
}

pub struct MeanSquaredError<const OUTPUT_COUNT: usize>;

impl<const OUTPUT_COUNT: usize> LossFunction for MeanSquaredError<OUTPUT_COUNT> {
    fn loss(&self, expected: &[f64], actual: &[f64]) -> f64 {
        assert_eq!(expected.len(), actual.len());
        assert_eq!(expected.len(), OUTPUT_COUNT);
        let mut loss = 0.0;
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            loss += (expected - actual).powi(2);
        }
        loss / OUTPUT_COUNT as f64
    }
}

pub fn calculate_loss(
    model: &mut Model,
    dataset: &[DatasetItem],
    loss_function: &dyn LossFunction,
    first_layer: usize,
    inputs: &[Vec<f64>],
    execution_context: &mut ExecutionContext,
) -> f64 {
    let mean_loss = dataset
        .iter()
        .zip(inputs.iter())
        .map(|(item, input)| {
            let output = model.evaluate_from(first_layer, input, execution_context);
            loss_function.loss(&item.label, output)
        })
        .sum::<f64>()
        / dataset.len() as f64;
    mean_loss
}

#[allow(clippy::too_many_arguments)] // We need them all.
pub fn estimate_derivative(
    model: &mut Model,
    execution_context: &mut ExecutionContext,
    layer_index: usize,
    parameter_index: usize,
    previous_loss: f64,
    loss_function: &dyn LossFunction,
    dataset: &[DatasetItem],
    inputs: &[Vec<f64>],
) -> f64 {
    const CHANGE: f64 = 0.00001;
    let layer = &mut model.layers_mut()[layer_index];
    let trainable_parameter = layer.trainable_parameter(parameter_index);
    let original_value = *trainable_parameter;
    *trainable_parameter += CHANGE;
    let new_loss = calculate_loss(
        model,
        dataset,
        loss_function,
        layer_index,
        inputs,
        execution_context,
    );
    let layer = &mut model.layers_mut()[layer_index];
    let trainable_parameter = layer.trainable_parameter(parameter_index);
    *trainable_parameter = original_value;
    (new_loss - previous_loss) / CHANGE
}

pub fn fit(
    model: &mut Model,
    loss_function: &dyn LossFunction,
    dataset: &[DatasetItem],
    batch_size: usize,
    learning_rate: f64,
    epochs: usize,
) -> ModelStats {
    let batch_count = (dataset.len() + batch_size - 1) / batch_size;
    let final_batch_size = if dataset.len() % batch_size == 0 {
        batch_size
    } else {
        dataset.len() % batch_size
    };
    let mut loss = 1.0;
    let mut execution_context = model.create_execution_context();
    let whole_dataset_inputs = dataset
        .iter()
        .map(|item| item.input.clone())
        .collect::<Vec<Vec<f64>>>();
    for epoch in 1..=epochs {
        println!("Epoch {epoch} / {epochs}");
        let mut last_step_time = Instant::now();
        for batch_index in 0..batch_count {
            let dataset = &dataset[batch_index * batch_size
                ..batch_index * batch_size
                    + if batch_index == batch_count - 1 {
                        final_batch_size
                    } else {
                        batch_size
                    }];
            loss = calculate_loss(
                model,
                dataset,
                loss_function,
                0,
                &dataset
                    .iter()
                    .map(|item| item.input.clone())
                    .collect::<Vec<Vec<f64>>>(),
                &mut execution_context,
            );
            for layer_index in 0..model.layers().len() {
                // Cache the calculations for the previous layers to make it faster.
                let output_from_previous_layers = dataset
                    .iter()
                    .map(|item| {
                        model
                            .evaluate_range(0, layer_index, &item.input, &mut execution_context)
                            .to_vec()
                    })
                    .collect::<Vec<Vec<f64>>>();
                let trainable_parameter_count =
                    model.layers()[layer_index].trainable_parameter_count();
                let mut changes = Vec::with_capacity(trainable_parameter_count);
                for trainable_parameter_index in 0..trainable_parameter_count {
                    let derivative = estimate_derivative(
                        model,
                        &mut execution_context,
                        layer_index,
                        trainable_parameter_index,
                        loss,
                        loss_function,
                        dataset,
                        &output_from_previous_layers,
                    );
                    let change = derivative * learning_rate;
                    changes.push(change);
                }
                let layer = &mut model.layers_mut()[layer_index];
                for (trainable_parameter_index, change) in
                    (0..trainable_parameter_count).zip(changes.into_iter())
                {
                    let trainable_parameter = layer.trainable_parameter(trainable_parameter_index);
                    *trainable_parameter -= change;
                }
                loss = calculate_loss(
                    model,
                    dataset,
                    loss_function,
                    layer_index,
                    &output_from_previous_layers,
                    &mut execution_context,
                );
            }
            let new_step_time = Instant::now();
            let step_duration = new_step_time - last_step_time;
            print!(
                "{batch_index}/{batch_count}: loss: {loss}, step_duration: {step_duration:?}\t\t\r"
            );
            last_step_time = new_step_time;
            stdout().lock().flush().unwrap();
        }
        loss = calculate_loss(
            model,
            dataset,
            loss_function,
            0,
            &whole_dataset_inputs,
            &mut execution_context,
        );
        println!("Loss: {loss}");
    }
    ModelStats { loss }
}
