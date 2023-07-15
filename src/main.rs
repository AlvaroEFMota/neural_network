use csv::Reader;
use ndarray::{array, Array, Dim};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;
use std::ops::{Add, Mul};
use rand::prelude::*;

const EULER: f64 = 2.7182818284590452353;
const RANDOM_RANGE: f64 = 0.5;
const LEARNING_RATE: f64 = 2.0;
const EPOCHS: i64 = 100;

fn main() -> Result<(), Box<dyn Error>> {
    let mut all_data: Vec<(Array<f64, Dim<[usize; 1]>>, Array<f64, Dim<[usize; 1]>>)> = vec![]; 

    // Proof of concept 1, inputs
    // all_data.push((array![0., 1.], array![0.]));
    // all_data.push((array![1., 0.], array![1.]));
    // all_data.push((array![1.0, 1.0], array![0.5]));
    // all_data.push((array![0., 0.], array![0.2]));
    // all_data.push((array![0.2, 0.7], array![0.8]));
    // let training_data: &[(Array<f64, Dim<[usize; 1]>>, Array<f64, Dim<[usize; 1]>>)] = &all_data[..];
    // let validation_data: &[(Array<f64, Dim<[usize; 1]>>, Array<f64, Dim<[usize; 1]>>)] = &all_data[..];
    // let network = vec![2, 3, 1];

    // Proof of concept 2, inputs
    // all_data.push((array![1., 1.], array![1., 0.]));
    // all_data.push((array![1., 0.], array![0., 1.]));
    // all_data.push((array![0.5, 0.5], array![1., 1.]));
    // all_data.push((array![0.1, 0.5], array![0., 0.]));
    // let training_data: &[(Array<f64, Dim<[usize; 1]>>, Array<f64, Dim<[usize; 1]>>)] = &all_data[..];
    // let validation_data: &[(Array<f64, Dim<[usize; 1]>>, Array<f64, Dim<[usize; 1]>>)] = &all_data[..];
    // let network = vec![2, 4, 2];    

    // Banknotes inputs
    let mut rdr = Reader::from_path("data/banknotes.csv").unwrap();
    for result in rdr.records() {
        let record = result?;
        let input: Array<f64, Dim<[usize; 1]>> = array![
            record[0].parse::<f64>()?,
            record[1].parse::<f64>()?,
            record[2].parse::<f64>()?,
            record[3].parse::<f64>()?,
        ];
        let desired_output: Array<f64, Dim<[usize; 1]>> = array![record[4].parse::<f64>()?];
        all_data.push((input, desired_output));
    }
    shuffle(&mut all_data[..300]);
    let training_data: &[(Array<f64, Dim<[usize; 1]>>, Array<f64, Dim<[usize; 1]>>)] = &all_data[..300];
    let validation_data: &[(Array<f64, Dim<[usize; 1]>>, Array<f64, Dim<[usize; 1]>>)] = &all_data[300..];
    let network = vec![4, 4, 4, 1];

    // The Neural Network
    let mut weights: Vec<Array<f64, Dim<[usize; 2]>>> = vec![];
    let mut biases: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];
    let mut error: Array<f64, Dim<[usize; 1]>> = Array::zeros(network.len());

    for i in 0..network.len() - 1 {
        let w: Array<f64, Dim<[usize; 2]>> = Array::random(
            (network[i], network[i + 1]),
            Uniform::new(-RANDOM_RANGE, RANDOM_RANGE),
        );
        let b: Array<f64, Dim<[usize; 1]>> =
            Array::zeros(network[i + 1]);
        weights.push(w);
        biases.push(b);
    }
    for _ in 0..EPOCHS {
        let mut weight_gradients: Vec<Array<f64, Dim<[usize; 2]>>> = vec![];
        let mut bias_gradients: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];

        for weight in weights.iter() {
            let _x = weight.raw_dim();
            let weight_gradient = Array::zeros(weight.raw_dim());
            let bias_gradient = Array::zeros(weight.ncols());
            weight_gradients.push(weight_gradient);
            bias_gradients.push(bias_gradient);
        }

        for (input, desired_output) in training_data {
            let mut layers: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];
            let mut zs: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];
            forward_propagation(&input, &weights, &biases, &mut layers, &mut zs);
            back_propagation(
                &mut weights,
                &mut biases,
                &layers,
                &zs,
                &desired_output,
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
        let mut layers: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];
        let mut zs: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];
        let neural_network_output = forward_propagation(&input, &weights, &biases, &mut layers, &mut zs);
        println!(
            "Forward propagation result -> {:?}",
            neural_network_output
        );
        
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
    println!("hits percent = {}%", (hit_count as f32 / validation_data.len() as f32) * 100.0);
    Ok(())
}

fn forward_propagation(
    input: &Array<f64, Dim<[usize; 1]>>,
    weights: &Vec<Array<f64, Dim<[usize; 2]>>>,
    biases: &Vec<Array<f64, Dim<[usize; 1]>>>,
    layers: &mut Vec<Array<f64, Dim<[usize; 1]>>>,
    zs: &mut Vec<Array<f64, Dim<[usize; 1]>>>,
) -> Array<f64, Dim<[usize; 1]>> {
    layers.push(input.clone());
    zs.push(Array::zeros(input.raw_dim()));
    let mut layer: Array<f64, Dim<[usize; 1]>> = input.clone();
    for (i, weight) in weights.iter().enumerate() {
        layer = layer.dot(weight).add(&biases[i]);
        zs.push(layer.clone());
        let layer_sigmoid = layer.mapv(|x: f64| sigmoid(x));
        layers.push(layer_sigmoid.clone());
        layer = layer_sigmoid;
    }
    layer
}

fn back_propagation(
    weights: &mut Vec<Array<f64, Dim<[usize; 2]>>>,
    biases: &mut Vec<Array<f64, Dim<[usize; 1]>>>,
    layers: &Vec<Array<f64, Dim<[usize; 1]>>>,
    zs: &Vec<Array<f64, Dim<[usize; 1]>>>,
    network_desired_output: &Array<f64, Dim<[usize; 1]>>,
    error: &mut Array<f64, Dim<[usize; 1]>>,
) {
    let mut delta = &layers[layers.len()-1] - network_desired_output;

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

            let mut weight_gradient_part: Array<f64, Dim<[usize; 2]>> =
                Array::zeros(weight.raw_dim());
            let mut bias_gradient_part: Array<f64, Dim<[usize; 1]>> =
                Array::zeros(biases[i - 1].raw_dim());

                for (j, mut matrix_row) in weight_gradient_part.outer_iter_mut().enumerate() {
                    for (k, matrix_element) in matrix_row.iter_mut().enumerate() {
                        *matrix_element +=
                            previous_layer[j] * LEARNING_RATE * delta[k] * sigmoid_derivative(z[k]);
                    }
                }
                for (j, matrix_element) in bias_gradient_part.iter_mut().enumerate() {
                    *matrix_element += LEARNING_RATE * delta[j] * sigmoid_derivative(z[j]);
                }

            weight = weight - &weight_gradient_part;

            let mut next_iteration_delta: Array<f64, Dim<[usize; 1]>> =
                Array::zeros(previous_layer.raw_dim());

            for (j, matrix_element) in next_iteration_delta.iter_mut().enumerate() {
                let mut sum = 0.0;
                for column in 0..weight_gradient_part.ncols() {
                    sum += weight[[j, column]] * sigmoid_derivative(previous_layer[j]) * delta[column];
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
    for i in 0..data.len()-1 {
        data.swap(i, rng.gen::<usize>()%data.len());
    }
}
