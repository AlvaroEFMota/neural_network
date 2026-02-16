use csv::Reader;
use ndarray::{array, Array, Dim};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::prelude::*;
use std::error::Error;
use std::ops::Add;

const EULER: f64 = std::f64::consts::E;
const RANDOM_RANGE: f64 = 0.5;
const LEARNING_RATE: f64 = 2.0;
const EPOCHS: i64 = 100;

type Matrix1D = Array<f64, Dim<[usize; 1]>>;
type Matrix2D = Array<f64, Dim<[usize; 2]>>;

fn main() -> Result<(), Box<dyn Error>> {
    let mut all_data: Vec<(Matrix1D, Matrix1D)> = vec![];

    // Proof of concept 1, inputs
    // all_data.push((array![0., 1.], array![0.]));
    // all_data.push((array![1., 0.], array![1.]));
    // all_data.push((array![1.0, 1.0], array![0.5]));
    // all_data.push((array![0., 0.], array![0.2]));
    // all_data.push((array![0.2, 0.7], array![0.8]));
    // let training_data: &[(Matrix1D , Matrix1D)] = &all_data[..];
    // let validation_data: &[(Matrix1D, Matrix1D)] = &all_data[..];
    // let network = vec![2, 3, 1];

    // Proof of concept 2, inputs
    // all_data.push((array![1., 1.], array![1., 0.]));
    // all_data.push((array![1., 0.], array![0., 1.]));
    // all_data.push((array![0.5, 0.5], array![1., 1.]));
    // all_data.push((array![0.1, 0.5], array![0., 0.]));
    // let training_data: &[(Matrix1D , Matrix1D)] = &all_data[..];
    // let validation_data: &[(Matrix1D , Matrix1D)] = &all_data[..];
    // let network = vec![2, 4, 2];

    // Banknotes inputs
    let mut rdr = Reader::from_path("data/banknotes.csv").unwrap();
    for result in rdr.records() {
        let record = result?;
        let input: Matrix1D = array![
            record[0].parse::<f64>()?,
            record[1].parse::<f64>()?,
            record[2].parse::<f64>()?,
            record[3].parse::<f64>()?,
        ];
        let desired_output: Matrix1D = array![record[4].parse::<f64>()?];
        all_data.push((input, desired_output));
    }
    shuffle(&mut all_data[..300]);
    let training_data: &[(Matrix1D, Matrix1D)] = &all_data[..300];
    let validation_data: &[(Matrix1D, Matrix1D)] = &all_data[300..];
    let network = vec![4, 4, 4, 1];

    // The Neural Network
    let mut weights: Vec<Matrix2D> = vec![];
    let mut biases: Vec<Matrix1D> = vec![];
    let mut error: Matrix1D = Array::zeros(network.len());

    for i in 0..network.len() - 1 {
        let w: Matrix2D = Array::random(
            (network[i], network[i + 1]),
            Uniform::new(-RANDOM_RANGE, RANDOM_RANGE),
        );
        let b: Matrix1D = Array::zeros(network[i + 1]);
        weights.push(w);
        biases.push(b);
    }
    for _ in 0..EPOCHS {
        let mut weight_gradients: Vec<Matrix2D> = vec![];
        let mut bias_gradients: Vec<Matrix1D> = vec![];

        for weight in weights.iter() {
            let _x = weight.raw_dim();
            let weight_gradient = Array::zeros(weight.raw_dim());
            let bias_gradient = Array::zeros(weight.ncols());
            weight_gradients.push(weight_gradient);
            bias_gradients.push(bias_gradient);
        }

        for (input, desired_output) in training_data {
            let mut layers: Vec<Matrix1D> = vec![];
            let mut zs: Vec<Matrix1D> = vec![];
            forward_propagation(input, &weights, &biases, &mut layers, &mut zs);
            back_propagation(
                &mut weights,
                &mut biases,
                &layers,
                &zs,
                desired_output,
                &mut error,
            );
            layers.clear();
        }
        println!("{:?}", error);
        error = Array::zeros(network.len());
    }

    let mut hit_count = 0;
    let error_margin = 0.03;
    for (input, output) in validation_data {
        let mut layers: Vec<Matrix1D> = vec![];
        let mut zs: Vec<Matrix1D> = vec![];
        let neural_network_output =
            forward_propagation(input, &weights, &biases, &mut layers, &mut zs);
        //println!("Forward propagation result -> {:?}", neural_network_output);

        let mut hit = true;
        for (i, elem) in neural_network_output.iter().enumerate() {
            if !(elem + error_margin > output[i] && elem - error_margin < output[i]) {
                hit = false
            }
        }

        if hit {
            hit_count += 1;
        }

        layers.clear();
    }
    println!("Total hits = {hit_count}");
    println!(
        "hits percent = {}%",
        (hit_count as f32 / validation_data.len() as f32) * 100.0
    );
    Ok(())
}

fn forward_propagation(
    input: &Matrix1D,
    weights: &[Matrix2D],
    biases: &[Matrix1D],
    layers: &mut Vec<Matrix1D>,
    zs: &mut Vec<Matrix1D>,
) -> Matrix1D {
    layers.push(input.clone());
    zs.push(Array::zeros(input.raw_dim()));
    let mut layer: Matrix1D = input.clone();
    for (i, weight) in weights.iter().enumerate() {
        layer = layer.dot(weight).add(&biases[i]);
        zs.push(layer.clone());
        let layer_sigmoid = layer.mapv(sigmoid);
        layers.push(layer_sigmoid.clone());
        layer = layer_sigmoid;
    }
    layer
}

fn back_propagation(
    weights: &mut [Matrix2D],
    biases: &mut [Matrix1D],
    layers: &[Matrix1D],
    zs: &[Matrix1D],
    network_desired_output: &Matrix1D,
    error: &mut Matrix1D,
) {
    let mut delta = &layers[layers.len() - 1] - network_desired_output;

    for (i, _layer) in layers.iter().enumerate().rev() {
        if i > 0 {
            let current_layer = &layers[i];
            let previous_layer = &layers[i - 1];
            let mut weight = weights[i - 1].clone();
            let z = &zs[i];
            let len = current_layer.len();
            let mut sum = 0.0;
            for j in 0..len {
                sum += delta[j] * delta[j];
            }
            error[i] = sum;

            let mut weight_gradient_part: Matrix2D = Array::zeros(weight.raw_dim());
            let mut bias_gradient_part: Matrix1D = Array::zeros(biases[i - 1].raw_dim());

            for (j, mut matrix_row) in weight_gradient_part.outer_iter_mut().enumerate() {
                for (k, matrix_element) in matrix_row.iter_mut().enumerate() {
                    *matrix_element +=
                        previous_layer[j] * LEARNING_RATE * delta[k] * sigmoid_derivative(z[k]);
                }
            }
            for (j, matrix_element) in bias_gradient_part.iter_mut().enumerate() {
                *matrix_element += LEARNING_RATE * delta[j] * sigmoid_derivative(z[j]);
            }

            weight -= &weight_gradient_part;

            let mut next_iteration_delta: Matrix1D = Array::zeros(previous_layer.raw_dim());

            for (j, matrix_element) in next_iteration_delta.iter_mut().enumerate() {
                let mut sum = 0.0;
                for column in 0..weight_gradient_part.ncols() {
                    sum +=
                        weight[[j, column]] * sigmoid_derivative(previous_layer[j]) * delta[column];
                }
                *matrix_element += sum;
            }

            delta = next_iteration_delta;
            weights[i - 1] = &weights[i - 1] - weight_gradient_part;
            biases[i - 1] = &biases[i - 1] - bias_gradient_part
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1. + EULER.powf(-x))
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[allow(dead_code)]
fn shuffle<T>(data: &mut [T]) {
    let mut rng = rand::thread_rng();
    for i in 0..data.len() - 1 {
        data.swap(i, rng.gen::<usize>() % data.len());
    }
}
