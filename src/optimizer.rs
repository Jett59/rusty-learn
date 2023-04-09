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

fn calculate_loss(
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
    learning_rate: f64,
    epochs: usize,
) -> ModelStats {
    let mut loss = f64::INFINITY;
    let mut execution_context = model.create_execution_context();
    let mut per_parameter_learning_rates = model
        .layers()
        .iter()
        .map(|layer| vec![learning_rate; layer.trainable_parameter_count()])
        .collect::<Vec<_>>();
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
            &mut execution_context,
        );
        println!("Loss: {loss}");
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
            let trainable_parameter_count = model.layers()[layer_index].trainable_parameter_count();
            // This contains tuples with the first element the derivative, the second the predicted change in loss.
            let mut derivatives = Vec::with_capacity(trainable_parameter_count);
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
                derivatives.push((
                    derivative,
                    derivative
                        * per_parameter_learning_rates[layer_index][trainable_parameter_index],
                ));
            }
            // Now we commit the one which gave the best loss.
            if trainable_parameter_count > 0 {
                // What we want is for the derivative's magnitude to be larger.
                // It doesn't matter which way we have to push the parameter, only how much impact it will have.
                let (best_derivative_index, (best_derivative, _best_predicted_loss)) = derivatives
                    .iter()
                    .enumerate()
                    .max_by(|(_, (_, a)), (_, (_, b))| a.abs().partial_cmp(&b.abs()).unwrap())
                    .unwrap();
                let layer = &mut model.layers_mut()[layer_index];
                let trainable_parameter = layer.trainable_parameter(best_derivative_index);
                // The amount we change it by should be related to the derivative. Larger derivatives mean we can go further before it starts to level off.
                // Since we want to decrease the value, we have to go forwards if the derivative is negative.
                let change = best_derivative
                    * per_parameter_learning_rates[layer_index][best_derivative_index];
                *trainable_parameter -= change;
                let new_loss = calculate_loss(
                    model,
                    dataset,
                    loss_function,
                    layer_index,
                    &output_from_previous_layers,
                    &mut execution_context,
                );
                let new_derivative = estimate_derivative(
                    model,
                    &mut execution_context,
                    layer_index,
                    best_derivative_index,
                    new_loss,
                    loss_function,
                    dataset,
                    &output_from_previous_layers,
                );
                // If the new derivative is a differnt sign to the old one, we've gone too far.
                if new_derivative.signum() != best_derivative.signum() {
                    per_parameter_learning_rates[layer_index][best_derivative_index] *= 0.5;
                } else {
                    per_parameter_learning_rates[layer_index][best_derivative_index] *= 1.25;
                }
                loss = new_loss;
            }
        }
    }
    ModelStats { loss }
}
