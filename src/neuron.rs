use std::fmt;

use crate::util::boxed_array;

pub trait Activation: fmt::Debug + Default {
    fn activate(&self, value: f64) -> f64;
}

pub mod activation {
    use super::*;

    #[derive(Debug, Default)]
    pub struct Linear;
    impl Activation for Linear {
        fn activate(&self, value: f64) -> f64 {
            value
        }
    }

    #[derive(Debug, Default)]
    pub struct Relu;
    impl Activation for Relu {
        fn activate(&self, value: f64) -> f64 {
            value.max(0.0)
        }
    }

    /// Fast sigmoid (not using the exp function)
    ///
    /// The formula is (x/(1+abs(x))+1)/2.
    #[derive(Debug, Default)]
    pub struct Sigmoid;
    impl Activation for Sigmoid {
        fn activate(&self, value: f64) -> f64 {
            (value / (1.0 + value.abs()) + 1.0) / 2.0
        }
    }
}

#[derive(Clone, Debug)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

pub struct LayerExecutionContext {
    outputs: Vec<f64>,
}

impl LayerExecutionContext {
    fn new(number_of_outputs: usize) -> Self {
        Self {
            outputs: vec![0.0; number_of_outputs],
        }
    }
}

pub struct ExecutionContext {
    layer_execution_contexts: Vec<LayerExecutionContext>,
}

impl ExecutionContext {
    fn new() -> Self {
        Self {
            layer_execution_contexts: Vec::new(),
        }
    }

    fn add_layer(&mut self, number_of_outputs: usize) {
        self.layer_execution_contexts
            .push(LayerExecutionContext::new(number_of_outputs));
    }
}

pub trait Layer: fmt::Debug {
    fn init(&mut self, next_layer_size: usize);
    fn input_size(&self) -> usize;
    /// Outputs are stored in the execution context.
    fn evaluate(&self, input: &[f64], execution_context: &mut LayerExecutionContext);
    fn trainable_parameter_count(&self) -> usize;
    fn trainable_parameter(&mut self, index: usize) -> &mut f64;
}

#[derive(Clone, Debug)]
pub struct DenseLayer<const NEURON_COUNT: usize, ActivationFunction: Activation> {
    neurons: Box<[Neuron; NEURON_COUNT]>,
    weight_count: usize,
    activation: ActivationFunction,
}

impl<const NEURON_COUNT: usize, ActivationFunction: Activation>
    DenseLayer<NEURON_COUNT, ActivationFunction>
{
    pub fn new() -> Self {
        Self {
            neurons: boxed_array(Neuron {
                weights: Vec::new(),
                bias: 0.0,
            }),
            weight_count: 0,
            activation: Default::default(),
        }
    }
}

impl<const NEURON_COUNT: usize, ActivationFunction: Activation> Layer
    for DenseLayer<NEURON_COUNT, ActivationFunction>
{
    fn init(&mut self, next_layer_size: usize) {
        self.weight_count = next_layer_size;
        for neuron in self.neurons.as_mut_slice() {
            // Setting the weights to 0 will make it very challenging to optimize, so we set them to 0.01 instead.
            neuron.weights = vec![0.01; self.weight_count];
        }
    }

    fn input_size(&self) -> usize {
        NEURON_COUNT
    }

    fn evaluate(&self, input: &[f64], execution_context: &mut LayerExecutionContext) {
        let outputs = &mut execution_context.outputs;
        outputs.fill(0.0);
        for (neuron, input) in self.neurons.iter().zip(input) {
            let input = self.activation.activate(*input) + neuron.bias;
            for (output, weight) in outputs.iter_mut().zip(neuron.weights.iter()) {
                *output += input * weight;
            }
        }
    }

    fn trainable_parameter_count(&self) -> usize {
        // Each neuron has weight_count weights and 1 bias.
        NEURON_COUNT * (self.weight_count + 1)
    }
    fn trainable_parameter(&mut self, index: usize) -> &mut f64 {
        // For each neuron, it starts with its weights, then its bias, then the next neuron.
        let neuron_index = index / (self.weight_count + 1);
        let neuron = &mut self.neurons[neuron_index];
        let index = index % (self.weight_count + 1);
        if index < self.weight_count {
            &mut neuron.weights[index]
        } else {
            &mut neuron.bias
        }
    }
}

#[derive(Clone, Debug)]
pub struct OutputLayer<ActivationFunction: Activation> {
    output_count: usize,
    activation: ActivationFunction,
}

impl<ActivationFunction: Activation> OutputLayer<ActivationFunction> {
    pub fn new(output_count: usize) -> Self {
        Self {
            output_count,
            activation: Default::default(),
        }
    }
}

impl<ActivationFunction: Activation> Layer for OutputLayer<ActivationFunction> {
    fn init(&mut self, _next_layer_size: usize) {}

    fn input_size(&self) -> usize {
        self.output_count
    }

    fn evaluate(&self, input: &[f64], execution_context: &mut LayerExecutionContext) {
        for (index, value) in input.iter().enumerate() {
            execution_context.outputs[index] = self.activation.activate(*value);
        }
    }

    fn trainable_parameter_count(&self) -> usize {
        0
    }
    fn trainable_parameter(&mut self, _index: usize) -> &mut f64 {
        panic!("Output layer has no trainable parameters");
    }
}

#[derive(Debug)]
pub struct Model {
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        let mut result = Self { layers };
        for i in 0..(result.layers.len() - 1) {
            let next_layer_size = result.layers[i + 1].input_size();
            result.layers[i].init(next_layer_size);
        }
        result
    }

    pub fn layers(&self) -> &[Box<dyn Layer>] {
        self.layers.as_slice()
    }
    pub fn layers_mut(&mut self) -> &mut [Box<dyn Layer>] {
        self.layers.as_mut_slice()
    }

    pub fn evaluate_range<'result>(
        &self,
        first: usize,
        last: usize,
        input: &'result [f64],
        execution_context: &'result mut ExecutionContext,
    ) -> &'result [f64] {
        assert!(first <= last);
        let mut output = input;
        for (layer, execution_context) in self
            .layers
            .iter()
            .zip(execution_context.layer_execution_contexts.iter_mut())
            .skip(first)
            .take(last - first)
        {
            assert_eq!(output.len(), layer.input_size());
            layer.evaluate(output, execution_context);
            output = execution_context.outputs.as_slice();
        }
        output
    }
    pub fn evaluate_from<'result>(
        &self,
        first: usize,
        input: &'result [f64],
        execution_context: &'result mut ExecutionContext,
    ) -> &'result [f64] {
        self.evaluate_range(first, self.layers.len(), input, execution_context)
    }
    pub fn evaluate<'result>(
        &self,
        input: &'result [f64],
        execution_context: &'result mut ExecutionContext,
    ) -> &'result [f64] {
        self.evaluate_from(0, input, execution_context)
    }

    pub fn create_execution_context(&self) -> ExecutionContext {
        let mut result = ExecutionContext::new();
        for layer_index in 0..(self.layers.len() - 1) {
            let next_layer_size = self.layers[layer_index + 1].input_size();
            result.add_layer(next_layer_size);
        }
        // The last layer will always have to output the same number of elements as it receives inputs.
        let last_layer_size = self.layers.last().unwrap().input_size();
        result.add_layer(last_layer_size);
        result
    }
}
