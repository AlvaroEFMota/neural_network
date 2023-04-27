use std::ops::Add;

use ndarray::{array, Array, Dim};

fn main() {
    let mut layers: Vec<Array<f64, Dim<[usize; 2]>>> = vec![];
    let mut weights: Vec<Array<f64, Dim<[usize; 2]>>> = vec![];
    let mut biases: Vec<Array<f64, Dim<[usize; 2]>>> = vec![];

    let bias: Array<f64, Dim<[usize; 2]>> = array![[1.], [1.]];
    biases.push(bias);
    let w1: Array<f64, Dim<[usize; 2]>> = array![[1., 2.], [3., 4.]];

    weights.push(w1);

    let input: Array<f64, Dim<[usize; 2]>> = array![[1., 1.]];
    println!("{:?}", forward_propagation(&input, &weights, &biases));
}

fn forward_propagation(
    input: &Array<f64, Dim<[usize; 2]>>,
    weights: &Vec<Array<f64, Dim<[usize; 2]>>>,
    biases: &Vec<Array<f64, Dim<[usize; 2]>>>,
) -> Array<f64, Dim<[usize; 2]>> {
    let mut layer: Array<f64, Dim<[usize; 2]>> = input.clone();
    println!("{:p}, {:?}", &layer, layer);
    for (i, weight) in weights.iter().enumerate() {
        layer = layer.dot(weight).add(&biases[i]);
        println!("{:p}, {:?}", &layer, layer);
    }
    layer
}
