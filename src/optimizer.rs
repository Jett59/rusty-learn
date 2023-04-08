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

pub fn fit(
    model: &mut Model,
    loss_function: &dyn LossFunction,
    dataset: &[DatasetItem],
    learning_rate: f64,
    epochs: usize,
) -> ModelStats {
    let mut loss = f64::INFINITY;
    let mut execution_context = model.create_execution_context();
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
            let mut new_losses = Vec::with_capacity(trainable_parameter_count);
            for trainable_parameter_index in 0..trainable_parameter_count {
                let mut layer = &mut model.layers_mut()[layer_index];
                let mut trainable_parameter = layer.trainable_parameter(trainable_parameter_index);
                let original_value = *trainable_parameter;
                // Change it by the learning rate and see what happens to the loss.
                *trainable_parameter = original_value + learning_rate;
                let new_loss_from_increasing = calculate_loss(
                    model,
                    dataset,
                    loss_function,
                    layer_index,
                    &output_from_previous_layers,
                    &mut execution_context,
                );
                // Refresh our references so rust doesn't complain.
                layer = &mut model.layers_mut()[layer_index];
                trainable_parameter = layer.trainable_parameter(trainable_parameter_index);
                // Now likewise for decreasing the value.
                *trainable_parameter = original_value + -1.0 * learning_rate;
                let new_loss_from_decreasing = calculate_loss(
                    model,
                    dataset,
                    loss_function,
                    layer_index,
                    &output_from_previous_layers,
                    &mut execution_context,
                );
                // Refresh our references so rust doesn't complain.
                layer = &mut model.layers_mut()[layer_index];
                trainable_parameter = layer.trainable_parameter(trainable_parameter_index);
                // Now restore the original value of the parameter.
                *trainable_parameter = original_value;
                // Whichever one gave a better loss, that is the one we submit for this parameter.
                if new_loss_from_increasing < new_loss_from_decreasing {
                    new_losses.push((
                        new_loss_from_increasing,
                        *trainable_parameter + learning_rate,
                        trainable_parameter_index,
                    ));
                } else {
                    new_losses.push((
                        new_loss_from_decreasing,
                        *trainable_parameter - learning_rate,
                        trainable_parameter_index,
                    ));
                }
            }
            // Now we commit the one which gave the best loss.
            if trainable_parameter_count > 0 {
                let best = new_losses
                    .into_iter()
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap();
                let layer = &mut model.layers_mut()[layer_index];
                if best.0 < loss {
                    let trainable_parameter = layer.trainable_parameter(best.2);
                    *trainable_parameter = best.1;
                }
            }
        }
    }
    ModelStats { loss }
}
