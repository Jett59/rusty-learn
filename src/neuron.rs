use std::fmt;

#[derive(Clone, Debug)]
pub enum Activation {
    Linear,
    Relu,
    Sigmoid,
}

impl Activation {
    fn activate(&self, value: f64) -> f64 {
        match self {
            Activation::Linear => value,
            Activation::Relu => value.max(0.0),
            // Uses the fast version of sigmoid ((1/(1+abs(x))+1)/2)
            Activation::Sigmoid => (1.0 / (1.0 + value.abs()) + 1.0) / 2.0,
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
pub struct DenseLayer {
    neurons: Vec<Neuron>,
    weight_count: usize,
    activation: Activation,
}

impl DenseLayer {
    pub fn new(neuron_count: usize, activation: Activation) -> Self {
        let mut neurons = Vec::with_capacity(neuron_count);
        for _ in 0..neuron_count {
            neurons.push(Neuron {
                weights: Vec::new(),
                bias: 0.0,
            });
        }
        Self {
            neurons,
            weight_count: 0,
            activation,
        }
    }
}

impl Layer for DenseLayer {
    fn init(&mut self, next_layer_size: usize) {
        self.weight_count = next_layer_size;
        for neuron in &mut self.neurons {
            // Setting the weights to 0 will make it very challenging to optimize, so we set them to 0.01 instead.
            neuron.weights = vec![0.01; self.weight_count];
        }
    }

    fn input_size(&self) -> usize {
        self.neurons.len()
    }

    fn evaluate(&self, input: &[f64], execution_context: &mut LayerExecutionContext) {
        // Zero it.
        execution_context.outputs.fill(0.0);
        for (input_index, neuron) in self.neurons.iter().enumerate() {
            for (output_index, weight) in neuron.weights.iter().enumerate() {
                execution_context.outputs[output_index] +=
                    (input[input_index] + neuron.bias) * weight;
            }
        }
        for value in &mut execution_context.outputs {
            *value = self.activation.activate(*value);
        }
    }

    fn trainable_parameter_count(&self) -> usize {
        // Each neuron has weight_count weights and 1 bias.
        self.neurons.len() * (self.weight_count + 1)
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
pub struct OutputLayer {
    output_count: usize,
}

impl OutputLayer {
    pub fn new(output_count: usize) -> Self {
        Self { output_count }
    }
}

impl Layer for OutputLayer {
    fn init(&mut self, _next_layer_size: usize) {}

    fn input_size(&self) -> usize {
        self.output_count
    }

    fn evaluate(&self, input: &[f64], execution_context: &mut LayerExecutionContext) {
        execution_context.outputs.copy_from_slice(input);
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
