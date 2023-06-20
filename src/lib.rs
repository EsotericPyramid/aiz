

pub mod aiz {
    pub struct NeuralNetwork {
        e: f32,
        biases: Vec<Vec<f64>>,
        weights: Vec<Vec<Vec<f64>>>,
        node_layout: Vec<usize>
    }

    impl NeuralNetwork {
        pub fn new(node_layout: Vec<usize>) -> Self {
            let length = node_layout.len();
            let mut biases = Vec::with_capacity(length-1);
            for layer_size in node_layout[1..length].iter() {
                let mut layer_biases = Vec::with_capacity(*layer_size);
                for _ in 0..*layer_size {
                    layer_biases.push(1.0); //CHANGE TO A RANDOM NUMBER, also see if rand doc has a fn for this kind of thing
                }
                biases.push(layer_biases);
            }
            let mut weights = Vec::with_capacity(length-1);
            for (previous_layer_size, layer_size) in node_layout[0..length-1].iter().zip(node_layout[1..length].iter()) {
                let mut layer_weights = Vec::with_capacity(*previous_layer_size);
                for _ in 0..*previous_layer_size {
                    let mut node_weights = Vec::with_capacity(*layer_size);
                    for _ in 0..*layer_size {
                        node_weights.push(1.0); //SEE ABOVE
                    }
                    layer_weights.push(node_weights);
                }
                weights.push(layer_weights);
            }
            NeuralNetwork{
            e: 2.71828, //euler's number, should use better approximation
            biases: biases,
            weights: weights,
            node_layout: node_layout
            }
        }

        pub fn run(&self,inputs: Vec<f64>) -> Vec<f64> {
            for (layer_weights, layer_biases) in self.weights.iter().zip(self.biases.iter()) {
                //Code
            }
            inputs //to get the compiler to shut up
        }
    }
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
