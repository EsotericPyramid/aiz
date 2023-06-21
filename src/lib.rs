
pub mod aiz {

    pub fn flip_matrix<T: Copy>(matrix: &Vec<Vec<T>>) -> Vec<Vec<T>> {
        //probably would be good if I could do this with iterators
        //expects a perfectly rectangular matrix / Vec<Vec<t>>
        let mut output_vec: Vec<Vec<T>> = Vec::new();
        for current_index in 0..matrix[0].len() {
            let mut intermediate_vec: Vec<T> = Vec::new();
            for vec in matrix {
                intermediate_vec.push(vec[current_index]);
            }
            output_vec.push(intermediate_vec);
        }
        output_vec
    }

    pub struct NeuralNetwork {
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
            biases: biases,
            weights: weights,
            node_layout: node_layout
            }
        }

        pub fn run(&self,inputs: Vec<f64>) -> Vec<f64> {
            let mut current_layer = inputs;
            for (layer_weights, layer_biases) in self.weights.iter().zip(self.biases.iter()) {
                let mut new_current_layer = Vec::new();
                for (row_weights,row_bias) in flip_matrix(layer_weights).iter().zip(layer_biases.iter()) {
                    new_current_layer.push(self.activation_function(row_weights.iter()
                                                            .zip(current_layer
                                                            .iter())
                                                            .map(|(weight,activation)| weight*activation)
                                                            .sum::<f64>()+row_bias));
                }
                current_layer = new_current_layer;
            }
            current_layer
        }

        pub fn activation_function(&self, x: f64) -> f64 {
            1.0/(1.0+x.exp())
        }

        pub fn derivative_activation_function(&self, x: f64) -> f64 {
            let intermedite_num = self.activation_function(x);
            intermedite_num*(1.0-intermedite_num)
            //see if using a variable is faster than not using it
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
