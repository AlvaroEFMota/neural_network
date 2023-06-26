use csv::Reader;
use ndarray::{array, Array, Dim};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;
use std::ops::{Add, Mul};

const EULER: f64 = 2.7182818284590452353;

fn main() -> Result<(), Box<dyn Error>> {
    let mut weights: Vec<Array<f64, Dim<[usize; 2]>>> = vec![];
    let mut biases: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];

    let network = vec![4, 4, 4, 1];
    for i in 0..network.len() - 1 {
        let w: Array<f64, Dim<[usize; 2]>> =
            Array::random((network[i], network[i + 1]), Uniform::new(-100., 100.));
        let b: Array<f64, Dim<[usize; 1]>> =
            Array::random(network[i + 1], Uniform::new(-100., 100.));
        println!("w = {:?}", w);
        println!("b = {:?}", b);
        weights.push(w);
        biases.push(b);
    }

    // let bias1: Array<f64, Dim<[usize; 1]>> = array![1., 1.];
    // let bias2: Array<f64, Dim<[usize; 1]>> = array![2.];
    // biases.push(bias1);
    // biases.push(bias2);
    // let w1: Array<f64, Dim<[usize; 2]>> = array![[1., 2.], [3., 4.], [1., 2.], [3., 4.]];
    // let w2: Array<f64, Dim<[usize; 2]>> = array![[3.], [2.]];

    // weights.push(w1);
    // weights.push(w2);

    let mut weight_gradients: Vec<Array<f64, Dim<[usize; 2]>>> = vec![];
    let mut bias_gradients: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];

    for weight in weights.iter() {
        let _x = weight.raw_dim();
        let weight_gradient = Array::zeros(weight.raw_dim());
        let bias_gradient = Array::zeros(weight.ncols());
        weight_gradients.push(weight_gradient);
        bias_gradients.push(bias_gradient);
    }

    // let input: Array<f64, Dim<[usize; 1]>> = array![1., 1.];
    // let desired_output: Array<f64, Dim<[usize; 1]>> = array![1., 1.];
    let total_training_data = 100;

    for _ in 0..total_training_data {
        let mut layers: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];
        let mut rdr = Reader::from_path("banknotes.csv")?;
        for result in rdr.records() {
            let record = result?;
            let input: Array<f64, Dim<[usize; 1]>> = array![
                record[0].parse::<f64>()?,
                record[1].parse::<f64>()?,
                record[2].parse::<f64>()?,
                record[3].parse::<f64>()?,
            ];
            let desired_output: Array<f64, Dim<[usize; 1]>> = array![record[4].parse::<f64>()?];
            forward_propagation(&input, &weights, &biases, &mut layers);
            back_propagation(
                &weights,
                &biases,
                &mut weight_gradients,
                &mut bias_gradients,
                &layers,
                &desired_output,
            );
            layers.clear()
        }
        for i in 0..weights.len() {
            weight_gradients[i] = weight_gradients[i].clone().mul(1.0/400.0);
            weights[i] = &weights[i] + &weight_gradients[i];
            bias_gradients[i] = bias_gradients[i].clone().mul(1.0/400.0);
            biases[i] = &biases[i] + &bias_gradients[i];
        }
    }
    let mut rdr = Reader::from_path("banknotes.csv")?;
    for result in rdr.records() {
        let record = result?;
        let input: Array<f64, Dim<[usize; 1]>> = array![
            record[0].parse::<f64>()?,
            record[1].parse::<f64>()?,
            record[2].parse::<f64>()?,
            record[3].parse::<f64>()?,
        ];
        let mut layers: Vec<Array<f64, Dim<[usize; 1]>>> = vec![];
        println!(
            "Forward propagation result -> {:?}",
            forward_propagation(&input, &weights, &biases, &mut layers)
        );
    }
    println!("finish");
    Ok(())
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
                    *matrix_element -= previous_layer[j]
                        * sigmoid_derivative(z[k])
                        * 2.0
                        * (current_layer[k] - desired_output[k]);
                }
            }

            for (j, matrix_element) in bias_gradient_part.iter_mut().enumerate() {
                *matrix_element -= sigmoid_derivative(z[j])
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
