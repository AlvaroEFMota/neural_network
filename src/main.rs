use std::ops::Add;

use ndarray::{array, Array, Dim};

const EULER: f64 = 2.7182818284590452353;

fn main() {
    let mut weights: Vec<Array<f64, Dim<[usize; 2]>>> = vec![];
    let mut biases: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];

    let bias1: Array<f64, Dim<[usize; 1]>> = array![1., 1.];
    let bias2: Array<f64, Dim<[usize; 1]>> = array![2., 2.];
    biases.push(bias1);
    biases.push(bias2);
    let w1: Array<f64, Dim<[usize; 2]>> = array![[1., 2.], [3., 4.]];
    let w2: Array<f64, Dim<[usize; 2]>> = array![[3., 1.], [2., 4.]];

    weights.push(w1);
    weights.push(w2);

    let mut weight_gradients: Vec<Array<f64, Dim<[usize; 2]>>> = vec![];
    let mut bias_gradients: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];

    for weight in weights.iter() {
        let weight_gradient = Array::zeros(weight.raw_dim());
        let bias_gradient = Array::zeros(weight.ncols());
        weight_gradients.push(weight_gradient);
        bias_gradients.push(bias_gradient);
    }

    let input: Array<f64, Dim<[usize; 1]>> = array![1., 1.];
    let desired_output: Array<f64, Dim<[usize; 1]>> = array![1., 1.];
    let total_training_data = 1;
    for _ in 0..total_training_data*270 {
            let mut layers: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];
        for _ in 0..total_training_data {
            forward_propagation(&input, &weights, &biases, &mut layers);
            back_propagation(
                &weights,
                &biases,
                &mut weight_gradients,
                &mut bias_gradients,
                &layers,
                &desired_output,
                total_training_data as f64,
            );
            layers.clear()
        }
        for i in 0..weights.len() {
            weights[i] = &weights[i] + &weight_gradients[i];
            biases[i] = &biases[i] + &bias_gradients[i];
        }
        println!(
            "Forward propagation result -> {:?}",
            forward_propagation(&input, &weights, &biases, &mut layers)
        );
    }
    print!("finish");
}

fn forward_propagation(
    // TODO store z[i]
    input: &Array<f64, Dim<[usize; 1]>>,
    weights: &Vec<Array<f64, Dim<[usize; 2]>>>,
    biases: &Vec<Array<f64, Dim<[usize; 1]>>>,
    layers: &mut Vec<Array<f64, Dim<[usize; 1]>>>,
) -> Array<f64, Dim<[usize; 1]>> {
    layers.push(input.clone());
    let mut layer: Array<f64, Dim<[usize; 1]>> = input.clone();
    for (i, weight) in weights.iter().enumerate() {
        layer = layer.dot(weight).add(&biases[i]);
        let layer_sigmoid = layer.mapv(|x: f64| sigmoid(x));
        layers.push(layer_sigmoid.clone());
        layer = layer_sigmoid;
    }
    layer
}

fn back_propagation(
    weights: &Vec<Array<f64, Dim<[usize; 2]>>>,
    biases: &Vec<Array<f64, Dim<[usize; 1]>>>,
    weight_gradients: &mut Vec<Array<f64, Dim<[usize; 2]>>>,
    bias_gradients: &mut Vec<Array<f64, Dim<[usize; 1]>>>,
    layers: &Vec<Array<f64, Dim<[usize; 1]>>>,
    network_desired_output: &Array<f64, Dim<[usize; 1]>>,
    total_training_data: f64,
) {
    let mut desired_output = network_desired_output.clone();

    for (i, _layer) in layers.iter().enumerate().rev() {
        if i > 0 {
            let weight_gradient_part = &mut weight_gradients[i - 1];
            let bias_gradient_part = &mut bias_gradients[i - 1];
            let current_layer = &layers[i];
            let previous_layer = &layers[i - 1];
            let weight = &weights[i - 1];

            let z = previous_layer.dot(weight).add(&biases[i - 1]);

            for (j, mut matrix_row) in weight_gradient_part.outer_iter_mut().enumerate() {
                for (k, matrix_element) in matrix_row.iter_mut().enumerate() {
                    *matrix_element -= (1.0 / total_training_data as f64)
                        * previous_layer[j]
                        * sigmoid_derivative(z[k])
                        * 2.0
                        * (current_layer[k] - desired_output[k]);
                }
            }

            for (j, matrix_element) in bias_gradient_part.iter_mut().enumerate() {
                *matrix_element -= (1.0 / total_training_data as f64)
                    * sigmoid_derivative(z[j])
                    * 2.0
                    * (current_layer[j] - desired_output[j]);
            }

            let mut next_iteration_desired_output: Array<f64, Dim<[usize; 1]>> =
                Array::zeros(previous_layer.raw_dim());

            for (j, matrix_element) in next_iteration_desired_output.iter_mut().enumerate() {
                let mut sum = 0.0;
                for column in 0..weight_gradient_part.ncols() {
                    sum += weight[[j, column]]
                        * sigmoid_derivative(z[column])
                        * 2.0
                        * (current_layer[column] - desired_output[column]);
                }
                *matrix_element += sum;
            }

            desired_output = next_iteration_desired_output;
        }
    }
}

#[allow(dead_code)]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1. + EULER.powf(-x))
}

#[allow(dead_code)]
fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
