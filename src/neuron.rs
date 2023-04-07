pub enum Activation {
    Linear,
    Relu,
}

impl Activation {
    fn activate(&self, value: f64) -> f64 {
        match self {
            Activation::Linear => value,
            Activation::Relu => value.max(0.0),
        }
    }
}

struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

pub trait Layer {
    fn init(&mut self, next_layer_size: usize);
    fn input_size(&self) -> usize;
    fn evaluate(&self, input: &[f64]) -> Vec<f64>;
}

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

    fn evaluate(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.weight_count];
        for (input_index, neuron) in self.neurons.iter().enumerate() {
            for (output_index, weight) in neuron.weights.iter().enumerate() {
                output[output_index] += (input[input_index] + neuron.bias) * weight;
            }
        }
        for value in &mut output {
            *value = self.activation.activate(*value);
        }
        output
    }
}

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

    fn evaluate(&self, input: &[f64]) -> Vec<f64> {
        input.to_vec()
    }
}

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

    pub fn evaluate(&self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        for layer in &self.layers {
            debug_assert_eq!(output.len(), layer.input_size());
            output = layer.evaluate(&output);
        }
        output
    }
}
