
pub mod aiz {
    use rand::Rng; //To inititalize networks randomly
    use rand::seq::SliceRandom; //to sample random groups of examples from the data for stochastic training
    use crossbeam; //To use immuttable non-static references in threads through its scoped threads
    use std::sync::mpsc; //communicating between threads

    //could be a iterator, probably
    //assumes the inner vectors are of equal length
    pub fn transpose_matrix<T>(matrix: &Vec<Vec<T>>) -> Vec<Vec<&T>> {
        let inner_matrix_len = matrix[0].len();
        let mut output = Vec::with_capacity(inner_matrix_len);
        for index in 0..inner_matrix_len {
            let mut output_column = Vec::with_capacity(matrix.len());
            for column in matrix {
                output_column.push(&column[index]);
            }
            output.push(output_column);
        }
        output
    }

    //same as above
    pub fn transpose_owned_matrix<T>(matrix: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let inner_matrix_len = matrix[0].len();
        let mut output_vec: Vec<Vec<T>> = Vec::with_capacity(inner_matrix_len);
        for _ in 0..inner_matrix_len {
            output_vec.push(Vec::with_capacity(matrix.len()))
        }
        for vec in matrix {
            for (val, current_output_vec) in vec.into_iter().zip(output_vec.iter_mut()) {
                current_output_vec.push(val)
            }
        }
        output_vec
    }

    //no randomness is implemented in this partitioning
    //this is because it is only intended to split up work
    //for multithreaded training and testing, so no randomness needed
    //Also: num_partitions is f64 for optimization reasons, 
    //in theory you could use that fact on purpose for cores with dif speeds
    //but it isnt for that
    pub fn partition_data<T>(data: &Vec<T>,num_partitions: f64) -> Vec<Vec<&T>> {
        let exact_partition_size = data.len() as f64 / num_partitions;
        let mut partitioned_data = Vec::new();
        let mut current_partition = Vec::new();
        let mut current_partition_point = exact_partition_size;
        for (example_num,example) in data.iter().enumerate() {
            if (example_num as f64) < current_partition_point {
                current_partition.push(example);
            } else {
                current_partition_point += exact_partition_size;
                partitioned_data.push(current_partition);
                current_partition = Vec::new();
                current_partition.push(example);
            }
        }
        partitioned_data.push(current_partition);
        partitioned_data
    }

    pub fn usize_to_8_be_bytes(n: usize) -> [u8; 8] {
        n.to_be_bytes()
    }

#[derive(PartialEq)]
#[derive(Debug)]
    pub struct NeuralNetwork {
        biases: Vec<Vec<f64>>,
        weights: Vec<Vec<Vec<f64>>>,
        node_layout: Vec<usize>
    }

    impl NeuralNetwork {
        //theoretically can and should be a macro like vec!
        pub fn new(node_layout: Vec<usize>,bias_bounds: f64,weight_bound: f64) -> Self {
            let mut rng = rand::thread_rng();
            let mut biases = Vec::with_capacity(node_layout.len());
            let mut weights = Vec::with_capacity(node_layout.len());
            for (layer, previous_layer) in (&node_layout[1..node_layout.len()]).iter().zip((&node_layout[0..node_layout.len()-1]).iter()) { //first iteration is on the 2nd layer, layer num is 0 and refers to 1st layer making the previous layer
                let mut layer_biases = Vec::with_capacity(*layer);
                let mut layer_weights = Vec::with_capacity(*layer);
                for _ in 0..*layer {
                    layer_biases.push(2.0*bias_bounds*rng.gen::<f64>()-bias_bounds);
                    let mut node_weights = Vec::with_capacity(*previous_layer);
                    for _ in 0..*previous_layer {
                        node_weights.push(2.0*weight_bound*rng.gen::<f64>()-weight_bound);
                    }
                    layer_weights.push(node_weights);
                }
                biases.push(layer_biases);
                weights.push(layer_weights);
            }
            NeuralNetwork {
                biases: biases,
                weights: weights,
                node_layout: node_layout
            }
        }

        //on actual testing, it seems it works except that a test was different only after 13 digits after decimal
        //not impactful but good to keep note
        pub fn into_bytes(self) -> Vec<u8> {
            /* turns a NeuralNetwork into the following file format:
            All nums are as BE bytes

            Offset 0: usize in 8 bytes denoting the length of the network; let this be x
            Offset 8: usize in 8 bytes denoting the first num in nodeLayout
            Offset 16: usize in 8 bytes denoting the second num in nodeLayout
            ...
            Offset 8x: usize in 8 bytes denoting the last num in nodeLayout

            The follwing bytes are then all f64's of self.biases and self.weights in that order
            they are ordered as the result of iterating through all the Vecs in them recursively
            The exact layout of how these correspond to the actual biases and weights can be found
            through nodelayout
            */
            let mut output_bytes = Vec::new();
            for byte in usize_to_8_be_bytes(self.node_layout.len()) {
                output_bytes.push(byte);
            }
            for layer in self.node_layout {
                for byte in usize_to_8_be_bytes(layer) {
                    output_bytes.push(byte);
                }
            }
            for layer in self.biases {
                for bias in layer {
                    for byte in bias.to_be_bytes() {
                        output_bytes.push(byte);
                    }
                }
            }
            for layer in self.weights {
                for node in layer {
                    for weight in node {
                        for byte in weight.to_be_bytes() {
                            output_bytes.push(byte);
                        }
                    }
                }
            }

            output_bytes
        }

        pub fn from_bytes(bytes: Vec<u8>) -> Self {
            let mut true_byte_num = 0;
            let mut eight_byte_buffer = [0; 8];
            let mut network_length = 0;
            let mut node_layout = Vec::new();
            let mut last_bias_byte_num = 0;
            let mut biases_buffer = Vec::new();
            let mut weights_buffer = Vec::new();
            for byte in bytes {
                match true_byte_num {
                    0..=6 => {
                        eight_byte_buffer[true_byte_num] = byte;
                        true_byte_num += 1;
                    }
                    7 => {
                        eight_byte_buffer[true_byte_num] = byte;
                        network_length = usize::from_be_bytes(eight_byte_buffer);
                        network_length *= 8; //JANK
                        network_length += 7; //JANK
                        true_byte_num += 1;
                    }
                    8.. if true_byte_num <= network_length => {
                        eight_byte_buffer[true_byte_num % 8] = byte;
                        if true_byte_num % 8 == 7 {
                            node_layout.push(usize::from_be_bytes(eight_byte_buffer));
                        }
                        if true_byte_num == network_length {
                            for layer in node_layout[1..node_layout.len()].iter() {
                                last_bias_byte_num += layer;
                            }
                            last_bias_byte_num *= 8; //Jank
                            last_bias_byte_num += network_length;
                        }
                        true_byte_num += 1;
                    }
                    8.. if (true_byte_num > network_length) && (true_byte_num <= last_bias_byte_num) => { //should still work but annoying
                        eight_byte_buffer[true_byte_num % 8] = byte;
                        if true_byte_num % 8 == 7 {
                            biases_buffer.push(f64::from_be_bytes(eight_byte_buffer));
                        }
                        true_byte_num += 1;
                    }
                    8.. if (true_byte_num > network_length) && (true_byte_num > last_bias_byte_num) => {
                        eight_byte_buffer[true_byte_num % 8] = byte;
                        if true_byte_num % 8 == 7 {
                            weights_buffer.push(f64::from_be_bytes(eight_byte_buffer));
                        }
                        true_byte_num += 1;
                    }
                    _ => {
                        println!("length: {}\nlast_bias_num: {}",network_length,last_bias_byte_num);
                        panic!("Some byte failed to match")
                    }
                }
            }
            let mut biases = Vec::new();
            let mut layer_biases = Vec::new();
            let mut current_layer_num = 1;
            for num in biases_buffer {
                layer_biases.push(num);
                if layer_biases.len() == node_layout[current_layer_num] {
                    biases.push(layer_biases);
                    layer_biases = Vec::new();
                    current_layer_num += 1;
                }
            }
            let mut weights = Vec::new();
            let mut layer_weights = Vec::new();
            let mut node_weights = Vec::new();
            let mut current_layer_num = 1;
            for num in weights_buffer {
                node_weights.push(num);
                if node_weights.len() == node_layout[current_layer_num-1] {
                    layer_weights.push(node_weights);
                    node_weights = Vec::new();
                    if layer_weights.len() == node_layout[current_layer_num] {
                        weights.push(layer_weights);
                        layer_weights = Vec::new();
                        current_layer_num += 1;
                    }
                }
            }
            NeuralNetwork {
                biases: biases,
                weights: weights,
                node_layout: node_layout
            }
        }

        pub fn get_biases(&self) -> &Vec<Vec<f64>> {
            &self.biases
        }
        
        pub fn get_weights(&self) -> &Vec<Vec<Vec<f64>>> {
            &self.weights
        }

        fn activation_fn(&self, x: f64) -> f64 {
            1.0/(1.0+(-x).exp()) //Sigmoid
        }

        //see if could be &f64 input
        fn derivative_activation_fn(&self, x: f64) -> f64 {
            let intermedite_num = self.activation_fn(x);
            intermedite_num*(1.0-intermedite_num) //Sigmoid'
        }

        //TO DO: check the Vec::with_capacity's
        pub fn run(&self,inputs: &Vec<f64>) -> Vec<f64> {
            let mut layer_activations = Vec::with_capacity(self.node_layout[1]);
            for (node_bias,node_weights) in self.biases[0].iter().zip(self.weights[0].iter()) {
                let mut node_val_before_activation_fn = *node_bias;
                for (weight,previous_node_activation) in node_weights.iter().zip(inputs.iter()) {
                    node_val_before_activation_fn += weight*previous_node_activation;
                }
                layer_activations.push(self.activation_fn(node_val_before_activation_fn));
            }
            for (layer_biases,layer_weights) in self.biases[1..self.biases.len()].iter().zip(self.weights[1..self.weights.len()].iter()) {
                let mut new_layer_activations = Vec::with_capacity(layer_biases.len());
                for (node_bias,node_weights) in layer_biases.iter().zip(layer_weights.iter()) {
                    let mut node_val_before_activation_fn = *node_bias;
                    for (weight,previous_node_activation) in node_weights.iter().zip(layer_activations.iter()) {
                        node_val_before_activation_fn += weight*previous_node_activation;
                    }
                    new_layer_activations.push(self.activation_fn(node_val_before_activation_fn));
                }
                layer_activations = new_layer_activations;
            }
            layer_activations
        }

        //clones inputs as part of the output, not good
        pub fn special_run(&self, inputs: &Vec<f64>) -> (Vec<Vec<f64>>,Vec<Vec<f64>>) {
            let mut network_pre_activations = Vec::with_capacity(self.node_layout.len()-1);
            let mut network_activations = Vec::with_capacity(self.node_layout.len());
            network_activations.push(inputs.clone()); //remove this is possible
            let mut layer_activations = Vec::with_capacity(self.node_layout[1]);
            let mut layer_pre_activations = Vec::with_capacity(self.node_layout[1]);
            for (node_bias,node_weights) in self.biases[0].iter().zip(self.weights[0].iter()) {
                let mut node_val_before_activation_fn = *node_bias;
                for (weight,previous_node_activation) in node_weights.iter().zip(inputs.iter()) {
                    node_val_before_activation_fn += weight*previous_node_activation;
                }
                layer_pre_activations.push(node_val_before_activation_fn);
                layer_activations.push(self.activation_fn(node_val_before_activation_fn));
            }
            for (layer_biases,layer_weights) in self.biases[1..self.biases.len()].iter().zip(self.weights[1..self.weights.len()].iter()) {
                let mut new_layer_activations = Vec::with_capacity(layer_biases.len());
                let mut new_layer_pre_activations = Vec::with_capacity(layer_biases.len());
                for (node_bias,node_weights) in layer_biases.iter().zip(layer_weights.iter()) {
                    let mut node_val_before_activation_fn = *node_bias;
                    for (weight,previous_node_activation) in node_weights.iter().zip(layer_activations.iter()) {
                        node_val_before_activation_fn += weight*previous_node_activation;
                    }
                    new_layer_pre_activations.push(node_val_before_activation_fn);
                    new_layer_activations.push(self.activation_fn(node_val_before_activation_fn));
                }
                network_pre_activations.push(layer_pre_activations);
                layer_pre_activations = new_layer_pre_activations;
                network_activations.push(layer_activations);
                layer_activations = new_layer_activations;
            }
            network_pre_activations.push(layer_pre_activations);
            network_activations.push(layer_activations);

            (network_activations,network_pre_activations)
        }

        pub fn test(&self,test_data: &Vec<(Vec<f64>,Vec<f64>)>) -> f64 {
            let mut output = 0.0;
            for (test_in,test_expected_out) in test_data {
                for (single_real_out,single_expected_out) in self.run(test_in).iter().zip(test_expected_out.iter()) {
                    let dif = single_real_out - single_expected_out;
                    output += dif*dif;
                }
            }
            output / test_data.len() as f64
        }

        pub fn prepartitioned_multithreaded_test(&self,partitioned_test_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>) -> f64 {
            let mut output = 0.0;
            crossbeam::scope(|scope| {
                let (original_transmitter,receiver) = mpsc::channel();
                for partition in partitioned_test_data {
                    let cloned_transmitter = original_transmitter.clone();
                    scope.spawn(move |_| {
                        for (test_in,test_expected_out) in partition {
                            for (single_real_out,single_expected_out) in self.run(test_in).iter().zip(test_expected_out.iter()) {
                                let dif = single_real_out - single_expected_out;
                                cloned_transmitter.send(dif*dif).unwrap();
                            }
                        }
                    });
                }
                drop(original_transmitter);
                for single_example_loss in receiver {
                    output += single_example_loss;
                }
            }).unwrap();
            let mut length = 0;
            for partition in partitioned_test_data {
                length += partition.len();
            }
            output / length as f64
        }

        fn apply_gradient(&mut self, biases_gradient: &Vec<Vec<f64>>, weights_gradient: &Vec<Vec<Vec<f64>>>,learning_rate: f64) {
            for (layer_biases,layer_biases_gradient) in self.biases.iter_mut().zip(biases_gradient.iter()) {
                for (bias, bias_gradient) in layer_biases.iter_mut().zip(layer_biases_gradient.iter()) {
                    *bias -= bias_gradient*learning_rate;
                }
            }
            for (layer_weights,layer_weights_gradient) in self.weights.iter_mut().zip(weights_gradient.iter()) {
                for (node_weights,node_weights_gradient) in layer_weights.iter_mut().zip(layer_weights_gradient.iter()) {
                    for (weight, weight_gradient) in node_weights.iter_mut().zip(node_weights_gradient.iter()) {
                        *weight -= weight_gradient*learning_rate
                    }
                }
            }
        }

        //TO DO: check the Vec::with_capacity's
        fn one_example_back_propagation(&self, training_in: &Vec<f64>, training_out: &Vec<f64>) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            let mut biases_gradient = Vec::with_capacity(self.node_layout.len());
            let mut weights_gradient = Vec::with_capacity(self.node_layout.len());
            
            let (network_activations,network_pre_activations) = self.special_run(training_in);

            //last layer specific calculations
            let mut last_layer_biases_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-1]);
            let mut last_layer_weights_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-1]);
            for (pre_activation,(expected_val,real_val)) in 
            network_pre_activations[network_pre_activations.len()-1].iter().zip(
            training_out.iter().zip(
            network_activations[network_activations.len()-1].iter())) {
                let current_node_derivative = 2.0*(real_val-expected_val)*self.derivative_activation_fn(*pre_activation);
                last_layer_biases_gradient.push(current_node_derivative);
                let mut node_weights_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-2]);
                for previous_node_activation in network_activations[network_activations.len()-2].iter() {
                    node_weights_gradient.push(previous_node_activation*current_node_derivative);
                }
                last_layer_weights_gradient.push(node_weights_gradient);
            }
            biases_gradient.push(last_layer_biases_gradient);
            weights_gradient.push(last_layer_weights_gradient);
            //All other layer calculations
            for (forward_layer_weights,(current_pre_activations,previous_activations)) in 
            self.weights.iter().rev().zip(
            network_pre_activations[0..network_pre_activations.len()-1].iter().rev().zip(
            network_activations[0..network_activations.len()-2].iter().rev())) {
                let mut layer_biases_gradient = Vec::with_capacity(current_pre_activations.len());
                let mut layer_weights_gradient = Vec::with_capacity(current_pre_activations.len());
                for (input_node_weights,node_pre_activation) in transpose_matrix(forward_layer_weights).iter().zip(current_pre_activations.iter()) {
                    let mut new_node_derivative = 0.0;
                    for (weight,forward_derivative) in input_node_weights.iter().zip(biases_gradient[biases_gradient.len()-1].iter()) {//layer_node_derivatives.iter()) { //seems to work
                        new_node_derivative += *weight * forward_derivative;
                    }
                    new_node_derivative *= self.derivative_activation_fn(*node_pre_activation);
                    layer_biases_gradient.push(new_node_derivative);
                    let mut node_weights_gradients = Vec::with_capacity(previous_activations.len());
                    for previous_node_activation in previous_activations {
                        node_weights_gradients.push(previous_node_activation*new_node_derivative)
                    }
                    layer_weights_gradient.push(node_weights_gradients);
                }
                biases_gradient.push(layer_biases_gradient);
                weights_gradient.push(layer_weights_gradient);
            }
            biases_gradient = biases_gradient.into_iter().rev().collect();
            weights_gradient = weights_gradient.into_iter().rev().collect();

            (biases_gradient,weights_gradient)
        }
    
        pub fn core_back_propagation(&self,training_data: &Vec<(Vec<f64>,Vec<f64>)>) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            let mut average_biases_gradient = Vec::with_capacity(self.node_layout.len());
            let mut average_weights_gradient = Vec::with_capacity(self.node_layout.len());
            for (previous_layer, layer) in (&self.node_layout[1..self.node_layout.len()]).iter().zip((&self.node_layout[0..self.node_layout.len()-1]).iter()) { //first iteration is on the 2nd layer, layer num is 0 and refers to 1st layer making the previous layer
                let mut layer_biases = Vec::with_capacity(*layer);
                let mut layer_weights = Vec::with_capacity(*layer);
                for _ in 0..*layer {
                    layer_biases.push(0.0);
                    let mut node_weights = Vec::with_capacity(*previous_layer);
                    for _ in 0..*previous_layer {
                        node_weights.push(0.0);
                    }
                    layer_weights.push(node_weights);
                }
                average_biases_gradient.push(layer_biases);
                average_weights_gradient.push(layer_weights);
            }
            for (training_in,training_out) in training_data {
                let (biases_gradient,weights_gradient) = self.one_example_back_propagation(training_in, training_out);
                for (layer_avg_biases_gradient,layer_biases_gradient) in average_biases_gradient.iter_mut().zip(biases_gradient.into_iter()) {
                    for (avg_bias_gradient,bias_gradient) in layer_avg_biases_gradient.iter_mut().zip(layer_biases_gradient.into_iter()) {
                        *avg_bias_gradient += bias_gradient;
                    }
                }
                for (layer_avg_weights_gradient,layer_weights_gradient) in average_weights_gradient.iter_mut().zip(weights_gradient.into_iter()) {
                    for (node_avg_weights_gradient,node_weights_gradient) in layer_avg_weights_gradient.iter_mut().zip(layer_weights_gradient.into_iter()) {
                        for (avg_weight_gradient,weight_gradient) in node_avg_weights_gradient.iter_mut().zip(node_weights_gradient.into_iter()) {
                            *avg_weight_gradient += weight_gradient;
                        }
                    }
                }
            }
            for layer_biases in average_biases_gradient.iter_mut() {
                for bias in layer_biases.iter_mut() {
                    *bias /= training_data.len() as f64
                }
            }
            for layer_weights in average_weights_gradient.iter_mut() {
                for node_weights in layer_weights.iter_mut() {
                    for weight in node_weights.iter_mut() {
                        *weight /= training_data.len() as f64
                    }
                }
            }

            (average_biases_gradient,average_weights_gradient)
        }
    
        pub fn core_multithreaded_back_propagation(&self,partitioned_training_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            let mut average_biases_gradient = Vec::with_capacity(self.node_layout.len());
            let mut average_weights_gradient = Vec::with_capacity(self.node_layout.len());
            for (previous_layer, layer) in (&self.node_layout[1..self.node_layout.len()]).iter().zip((&self.node_layout[0..self.node_layout.len()-1]).iter()) { //first iteration is on the 2nd layer, layer num is 0 and refers to 1st layer making the previous layer
                let mut layer_biases = Vec::with_capacity(*layer);
                let mut layer_weights = Vec::with_capacity(*layer);
                for _ in 0..*layer {
                    layer_biases.push(0.0);
                    let mut node_weights = Vec::with_capacity(*previous_layer);
                    for _ in 0..*previous_layer {
                        node_weights.push(0.0);
                    }
                    layer_weights.push(node_weights);
                }
                average_biases_gradient.push(layer_biases);
                average_weights_gradient.push(layer_weights);
            }
            crossbeam::scope(|scope| {
                let (original_transmitter,receiver) = mpsc::channel();
                for partition in partitioned_training_data {
                    let cloned_transmitter = original_transmitter.clone();
                    scope.spawn(move |_| {
                        for (training_in,training_out) in partition {
                            let gradient_pair = self.one_example_back_propagation(training_in, training_out);
                            cloned_transmitter.send(gradient_pair).unwrap();
                        }
                    });
                }
                drop(original_transmitter);
                for (biases_gradient,weights_gradient) in receiver {
                    for (layer_avg_biases_gradient,layer_biases_gradient) in average_biases_gradient.iter_mut().zip(biases_gradient.into_iter()) {
                        for (avg_bias_gradient,bias_gradient) in layer_avg_biases_gradient.iter_mut().zip(layer_biases_gradient.into_iter()) {
                            *avg_bias_gradient += bias_gradient;
                        }
                    }
                    for (layer_avg_weights_gradient,layer_weights_gradient) in average_weights_gradient.iter_mut().zip(weights_gradient.into_iter()) {
                        for (node_avg_weights_gradient,node_weights_gradient) in layer_avg_weights_gradient.iter_mut().zip(layer_weights_gradient.into_iter()) {
                            for (avg_weight_gradient,weight_gradient) in node_avg_weights_gradient.iter_mut().zip(node_weights_gradient.into_iter()) {
                                *avg_weight_gradient += weight_gradient;
                            }
                        }
                    }
                }
            }).unwrap();
            let mut data_length = 0;
            for partition in partitioned_training_data {
                data_length += partition.len()
            }
            for layer_biases in average_biases_gradient.iter_mut() {
                for bias in layer_biases.iter_mut() {
                    *bias /= data_length as f64
                }
            }
            for layer_weights in average_weights_gradient.iter_mut() {
                for node_weights in layer_weights.iter_mut() {
                    for weight in node_weights.iter_mut() {
                        *weight /= data_length as f64
                    }
                }
            }

            (average_biases_gradient,average_weights_gradient)
        }

        pub fn core_stochastic_multithreaded_back_propagation(&self,partitioned_training_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>,batch_size_per_thread: usize) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            let mut average_biases_gradient = Vec::with_capacity(self.node_layout.len());
            let mut average_weights_gradient = Vec::with_capacity(self.node_layout.len());
            for (previous_layer, layer) in (&self.node_layout[1..self.node_layout.len()]).iter().zip((&self.node_layout[0..self.node_layout.len()-1]).iter()) { //first iteration is on the 2nd layer, layer num is 0 and refers to 1st layer making the previous layer
                let mut layer_biases = Vec::with_capacity(*layer);
                let mut layer_weights = Vec::with_capacity(*layer);
                for _ in 0..*layer {
                    layer_biases.push(0.0);
                    let mut node_weights = Vec::with_capacity(*previous_layer);
                    for _ in 0..*previous_layer {
                        node_weights.push(0.0);
                    }
                    layer_weights.push(node_weights);
                }
                average_biases_gradient.push(layer_biases);
                average_weights_gradient.push(layer_weights);
            }
            crossbeam::scope(|scope| {
                let (original_transmitter,receiver) = mpsc::channel();
                for partition in partitioned_training_data {
                    let cloned_transmitter = original_transmitter.clone();
                    scope.spawn(move |_| {
                        let mut rng = rand::thread_rng();
                        for (training_in,training_out) in partition.choose_multiple(&mut rng, batch_size_per_thread) {
                            let gradient_pair = self.one_example_back_propagation(training_in, training_out);
                            cloned_transmitter.send(gradient_pair).unwrap();
                        }
                    });
                }
                drop(original_transmitter);
                for (biases_gradient,weights_gradient) in receiver {
                    for (layer_avg_biases_gradient,layer_biases_gradient) in average_biases_gradient.iter_mut().zip(biases_gradient.into_iter()) {
                        for (avg_bias_gradient,bias_gradient) in layer_avg_biases_gradient.iter_mut().zip(layer_biases_gradient.into_iter()) {
                            *avg_bias_gradient += bias_gradient;
                        }
                    }
                    for (layer_avg_weights_gradient,layer_weights_gradient) in average_weights_gradient.iter_mut().zip(weights_gradient.into_iter()) {
                        for (node_avg_weights_gradient,node_weights_gradient) in layer_avg_weights_gradient.iter_mut().zip(layer_weights_gradient.into_iter()) {
                            for (avg_weight_gradient,weight_gradient) in node_avg_weights_gradient.iter_mut().zip(node_weights_gradient.into_iter()) {
                                *avg_weight_gradient += weight_gradient;
                            }
                        }
                    }
                }
            }).unwrap();
            let mut data_length = 0;
            for partition in partitioned_training_data {
                data_length += partition.len()
            }
            for layer_biases in average_biases_gradient.iter_mut() {
                for bias in layer_biases.iter_mut() {
                    *bias /= data_length as f64
                }
            }
            for layer_weights in average_weights_gradient.iter_mut() {
                for node_weights in layer_weights.iter_mut() {
                    for weight in node_weights.iter_mut() {
                        *weight /= data_length as f64
                    }
                }
            }

            (average_biases_gradient,average_weights_gradient)
        }

        pub fn backtracking_line_search_train(
            &mut self, 
            training_data: &Vec<(Vec<f64>,Vec<f64>)>, 
            first_checked_rate: f64, 
            rate_degradation_factor: f64, 
            tolerance_parameter: f64, 
            minimum_learning_rate: f64, 
            max_iterations: u32
        ) {
            let mut previous_test = self.test(training_data);
            println!("{}",previous_test);
            let mut current_learning_rate = first_checked_rate * rate_degradation_factor;
            let mut iteration_num = 0;
            'main_loop: while iteration_num < max_iterations {
                current_learning_rate /= rate_degradation_factor;
                let (biases_gradient,weights_gradient) = self.core_back_propagation(training_data);
                let mut true_local_slope = 0.0;
                for layer in biases_gradient.iter() {
                    for bias in layer {
                        true_local_slope += bias * (bias);
                    }
                }
                for layer in weights_gradient.iter() {
                    for node in layer {
                        for weight in node {
                            true_local_slope += weight * weight;
                        }
                    }
                }
                let tolerable_local_slope = tolerance_parameter * true_local_slope;
                self.apply_gradient(&biases_gradient, &weights_gradient, current_learning_rate);
                let mut new_test = self.test(training_data);
                println!("{}",new_test);
                while previous_test - new_test < current_learning_rate * tolerable_local_slope {
                    println!("Improvement: {}",previous_test-new_test);
                    println!("Required:    {}", current_learning_rate * tolerable_local_slope);
                    current_learning_rate *= rate_degradation_factor;
                    println!("Looping: {}",current_learning_rate);
                    if current_learning_rate < minimum_learning_rate {
                        self.apply_gradient(&biases_gradient, &weights_gradient, current_learning_rate / rate_degradation_factor);
                        break 'main_loop;
                    }
                    self.apply_gradient(&biases_gradient, &weights_gradient, current_learning_rate-(current_learning_rate/rate_degradation_factor));
                    new_test = self.test(training_data);
                    println!("{}",new_test);
                }
                previous_test = new_test;

                iteration_num += 1
            }
        }

        pub fn multithreaded_backtracking_line_search_train(
            &mut self, 
            training_data: &Vec<(Vec<f64>,Vec<f64>)>, 
            num_partitions: f64, 
            first_checked_rate: f64, 
            rate_degradation_factor: f64, 
            tolerance_parameter: f64, 
            minimum_learning_rate: f64, 
            max_iterations: u32
        ) {
            let partitioned_training_data = partition_data(training_data, num_partitions);
            let mut previous_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
            println!("{}",previous_test);
            let mut current_learning_rate = first_checked_rate * rate_degradation_factor;
            let mut iteration_num = 0;
            'main_loop: while iteration_num < max_iterations {
                current_learning_rate /= rate_degradation_factor;
                let (biases_gradient,weights_gradient) = self.core_multithreaded_back_propagation(&partitioned_training_data);
                let mut true_local_slope = 0.0;
                for layer in biases_gradient.iter() {
                    for bias in layer {
                        true_local_slope += bias * (bias);
                    }
                }
                for layer in weights_gradient.iter() {
                    for node in layer {
                        for weight in node {
                            true_local_slope += weight * weight;
                        }
                    }
                }
                let tolerable_local_slope = tolerance_parameter * true_local_slope;
                self.apply_gradient(&biases_gradient, &weights_gradient, current_learning_rate);
                let mut new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                println!("{}",new_test);
                while previous_test - new_test < current_learning_rate * tolerable_local_slope {
                    println!("Improvement: {}",previous_test-new_test);
                    println!("Required:    {}", current_learning_rate * tolerable_local_slope);
                    current_learning_rate *= rate_degradation_factor;
                    println!("Looping: {}",current_learning_rate);
                    if current_learning_rate < minimum_learning_rate {
                        self.apply_gradient(&biases_gradient, &weights_gradient, current_learning_rate / rate_degradation_factor);
                        break 'main_loop;
                    }
                    self.apply_gradient(&biases_gradient, &weights_gradient, current_learning_rate-(current_learning_rate/rate_degradation_factor));
                    new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                    println!("{}",new_test);
                }
                previous_test = new_test;

                iteration_num += 1
            }
        }
        
        pub fn stochastic_multithreaded_backtracking_line_search_train(
            &mut self, 
            training_data: &Vec<(Vec<f64>,Vec<f64>)>, 
            num_partitions: f64, 
            batch_size: usize,
            first_checked_rate: f64, 
            rate_degradation_factor: f64, 
            tolerance_parameter: f64, 
            minimum_learning_rate: f64, 
            max_iterations: u32
        ) {
            let partitioned_training_data = partition_data(training_data, num_partitions);
            let mut previous_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
            println!("{}",previous_test);
            let mut current_learning_rate = first_checked_rate * rate_degradation_factor;
            let mut iteration_num = 0;
            'main_loop: while iteration_num < max_iterations {
                current_learning_rate /= rate_degradation_factor;
                let (biases_gradient,weights_gradient) = self.core_stochastic_multithreaded_back_propagation(&partitioned_training_data,batch_size);
                let mut true_local_slope = 0.0;
                for layer in biases_gradient.iter() {
                    for bias in layer {
                        true_local_slope += bias * (bias);
                    }
                }
                for layer in weights_gradient.iter() {
                    for node in layer {
                        for weight in node {
                            true_local_slope += weight * weight;
                        }
                    }
                }
                let tolerable_local_slope = tolerance_parameter * true_local_slope;
                self.apply_gradient(&biases_gradient, &weights_gradient, current_learning_rate);
                let mut new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                println!("{}",new_test);
                while previous_test - new_test < current_learning_rate * tolerable_local_slope {
                    println!("Improvement: {}",previous_test-new_test);
                    println!("Required:    {}", current_learning_rate * tolerable_local_slope);
                    current_learning_rate *= rate_degradation_factor;
                    println!("Looping: {}",current_learning_rate);
                    if current_learning_rate < minimum_learning_rate {
                        self.apply_gradient(&biases_gradient, &weights_gradient, current_learning_rate / rate_degradation_factor);
                        break 'main_loop;
                    }
                    self.apply_gradient(&biases_gradient, &weights_gradient, current_learning_rate-(current_learning_rate/rate_degradation_factor));
                    new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                    println!("{}",new_test);
                }
                previous_test = new_test;

                iteration_num += 1
            }
        }
    }

    impl Clone for NeuralNetwork {
        fn clone(&self) -> Self {
            NeuralNetwork { biases: self.biases.clone(), weights: self.weights.clone(), node_layout: self.node_layout.clone() }
        }
    }
}

pub mod aiz_unstable {
    use rand::Rng; 
    //RNG, figure it out yourself
    use std::sync::mpsc; //communicating between threads
    use rand::seq::SliceRandom; //stochastic stuf
    use std::collections::VecDeque; //for l_bfgs memory buffer

    //could be a iterator, probably
    //assumes the inner vectors are of equal length
    pub fn transpose_matrix<T>(matrix: &Vec<Vec<T>>) -> Vec<Vec<&T>> {
        let inner_matrix_len = matrix[0].len();
        let mut output = Vec::with_capacity(inner_matrix_len);
        for index in 0..inner_matrix_len {
            let mut output_column = Vec::with_capacity(matrix.len());
            for column in matrix {
                output_column.push(&column[index]);
            }
            output.push(output_column);
        }
        output
    }

    //same as above
    pub fn transpose_owned_matrix<T>(matrix: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let inner_matrix_len = matrix[0].len();
        let mut output_vec: Vec<Vec<T>> = Vec::with_capacity(inner_matrix_len);
        for _ in 0..inner_matrix_len {
            output_vec.push(Vec::with_capacity(matrix.len()))
        }
        for vec in matrix {
            for (val, current_output_vec) in vec.into_iter().zip(output_vec.iter_mut()) {
                current_output_vec.push(val)
            }
        }
        output_vec
    }

    //no randomness is implemented in this partitioning
    //this is because it is only intended to split up work
    //for multithreaded training and testing, so no randomness needed
    //Also: num_partitions is f64 for optimization reasons, 
    //in theory you could use that fact on purpose for cores with dif speeds
    //but it isnt for that
    pub fn partition_data<T>(data: &Vec<T>,num_partitions: f64) -> Vec<Vec<&T>> {
        let exact_partition_size = data.len() as f64 / num_partitions;
        let mut partitioned_data = Vec::new();
        let mut current_partition = Vec::new();
        let mut current_partition_point = exact_partition_size;
        for (example_num,example) in data.iter().enumerate() {
            if (example_num as f64) < current_partition_point {
                current_partition.push(example);
            } else {
                current_partition_point += exact_partition_size;
                partitioned_data.push(current_partition);
                current_partition = Vec::new();
                current_partition.push(example);
            }
        }
        partitioned_data.push(current_partition);
        partitioned_data
    }

    pub fn usize_to_8_be_bytes(n: usize) -> [u8; 8] {
        n.to_be_bytes()
    }

    pub trait FileSupport {
        type ExtraDecodingInfo;
        //is in be bytes
        fn into_bytes(self) -> Vec<u8>;
        fn from_bytes(bytes: Vec<u8>,decoding_info: Self::ExtraDecodingInfo) -> Self;
    }

    pub trait Network: Send + Sync {
        type Gradient: Gradient;

        fn run(&self,inputs: &[f64]) -> Vec<f64>;
        fn subtract_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64);
        fn add_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64);
        fn build_zero_gradient(&self) -> Self::Gradient; //gradient indicating zero movement
        fn one_example_back_propagation(&self,training_in: &[f64],training_out: &[f64]) -> Self::Gradient;
    }

    pub trait Gradient: Send + Sync {
        fn add(&mut self,other: Self);
        fn mult(&mut self,factor: f64);
        fn div(&mut self,divisor: f64);
        fn dot_product(&self,other: &Self) -> f64;
    }

    impl Gradient for (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) { //MLP  
        fn add(&mut self,other: (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>)) {
            for (self_biases_layer,other_biases_layer) in self.0.iter_mut().zip(other.0.into_iter()) {
                for (self_bias,other_bias) in self_biases_layer.iter_mut().zip(other_biases_layer.into_iter()) {
                    *self_bias += other_bias;
                }
            }
            for (self_weights_layer,other_weights_layer) in self.1.iter_mut().zip(other.1.into_iter()) {
                for (self_weights_node,other_weights_node) in self_weights_layer.iter_mut().zip(other_weights_layer.into_iter()) {
                    for (self_weight, other_weight) in self_weights_node.iter_mut().zip(other_weights_node.into_iter()) {
                        *self_weight += other_weight
                    }
                }
            }
        }
        fn mult(&mut self,factor: f64) {
            for biases_layer in &mut self.0 {
                for bias in biases_layer {
                    *bias *= factor;
                }
            }
            for weights_layer in &mut self.1 {
                for weights_node in weights_layer {
                    for weight in weights_node {
                        *weight *= factor;
                    }
                }
            }
        }
        fn div(&mut self,divisor: f64) {
            for biases_layer in &mut self.0 {
                for bias in biases_layer {
                    *bias /= divisor;
                }
            }
            for weights_layer in &mut self.1 {
                for weights_node in weights_layer {
                    for weight in weights_node {
                        *weight /= divisor;
                    }
                }
            }
        }
        fn dot_product(&self,other: &Self) -> f64 {
            let mut output = 0.0;
            for (self_layer,other_layer) in self.0.iter().zip(other.0.iter()) {
                for (self_bias,other_bias) in self_layer.iter().zip(other_layer.iter()) {
                    output += self_bias * other_bias;
                }
            }
            for (self_layer,other_layer) in self.1.iter().zip(other.1.iter()) {
                for (self_node,other_node) in self_layer.iter().zip(other_layer.iter()) {
                    for (self_weight,other_weight) in self_node.iter().zip(other_node.iter()) {
                        output += self_weight * other_weight;
                    }
                }
            }
            output
        }
    }

    pub trait NetworkInherentMethods: Network {
        fn test(&self,test_data: &Vec<(Vec<f64>,Vec<f64>)>) -> f64;
        fn prepartitioned_multithreaded_test(&self,partitioned_test_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>) -> f64;
        fn multithreaded_test(&self,test_data: &Vec<(Vec<f64>,Vec<f64>)>,num_partitions: f64) -> f64;
        fn back_propagation(&self,training_data: &Vec<(Vec<f64>,Vec<f64>)>) -> Self::Gradient;
        fn multithreaded_back_propagation(&self,partitioned_training_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>) -> Self::Gradient;
        fn stochastic_multithreaded_back_propagation(&self,training_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>,batch_size: usize) -> Self::Gradient;
        fn backtracking_line_search_train(
            &mut self, 
            training_data: &Vec<(Vec<f64>,Vec<f64>)>, 
            first_checked_rate: f64, 
            rate_degradation_factor: f64, 
            tolerance_parameter: f64, 
            minimum_learning_rate: f64, 
            max_iterations: u32
        );
        fn multithreaded_backtracking_line_search_train(
            &mut self, 
            training_data: &Vec<(Vec<f64>,Vec<f64>)>, 
            num_partitions: f64,
            first_checked_rate: f64, 
            rate_degradation_factor: f64, 
            tolerance_parameter: f64, 
            minimum_learning_rate: f64, 
            max_iterations: u32
        );
        fn stochastic_multithreaded_backtracking_line_search_train(
            &mut self, 
            training_data: &Vec<(Vec<f64>,Vec<f64>)>, 
            num_partitions: f64,
            batch_size: usize,
            first_checked_rate: f64, 
            rate_degradation_factor: f64, 
            tolerance_parameter: f64, 
            minimum_learning_rate: f64, 
            max_iterations: u32
        );
    }

    impl<T: Network> NetworkInherentMethods for T {
        fn test(&self,test_data: &Vec<(Vec<f64>,Vec<f64>)>) -> f64 {
            let mut output = 0.0;
            for (test_in,test_expected_out) in test_data {
                for (single_real_out,single_expected_out) in self.run(test_in).iter().zip(test_expected_out.iter()) {
                    let dif = single_real_out - single_expected_out;
                    output += dif*dif;
                }
            }
            output / test_data.len() as f64
        }
    
        fn prepartitioned_multithreaded_test(&self,partitioned_test_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>) -> f64 {
            let mut output = 0.0;
            crossbeam::scope(|scope| {
                let (original_transmitter,receiver) = mpsc::channel();
                for partition in partitioned_test_data {
                    let cloned_transmitter = original_transmitter.clone();
                    scope.spawn(move |_| {
                        for (test_in,test_expected_out) in partition {
                            for (single_real_out,single_expected_out) in self.run(test_in).iter().zip(test_expected_out.iter()) {
                                let dif = single_real_out - single_expected_out;
                                cloned_transmitter.send(dif*dif).unwrap();
                            }
                        }
                    });
                }
                drop(original_transmitter);
                for single_example_loss in receiver {
                    output += single_example_loss;
                }
            }).unwrap();
            let mut length = 0;
            for partition in partitioned_test_data {
                length += partition.len();
            }
            output / length as f64
        }

        fn multithreaded_test(&self,test_data: &Vec<(Vec<f64>,Vec<f64>)>,num_partitions: f64) -> f64 {
            self.prepartitioned_multithreaded_test(&partition_data(test_data,num_partitions))
        }

        fn back_propagation(&self,training_data: &Vec<(Vec<f64>,Vec<f64>)>) -> Self::Gradient {
            let mut output_gradient = self.build_zero_gradient();
            for (training_in,training_out) in training_data {
                let current_gradient = self.one_example_back_propagation(training_in, training_out);
                output_gradient.add(current_gradient);
            }
            output_gradient.div(training_data.len() as f64);
            output_gradient
        }

        fn multithreaded_back_propagation(&self,partitioned_training_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>) -> Self::Gradient {
            let mut output_gradient = self.build_zero_gradient();
            crossbeam::scope(|scope| {
                let (original_transmitter,receiver) = mpsc::channel();
                for partition in partitioned_training_data {
                    let cloned_transmitter = original_transmitter.clone();
                    scope.spawn(move |_| {
                        for (training_in,training_out) in partition {
                            let gradient_pair = self.one_example_back_propagation(training_in, training_out);
                            cloned_transmitter.send(gradient_pair).unwrap();
                        }
                    });
                }
                drop(original_transmitter);
                for gradient in receiver {
                    output_gradient.add(gradient)
                }
            }).unwrap();
            let mut data_length = 0;
            for partition in partitioned_training_data {
                data_length += partition.len()
            }
            output_gradient.div(data_length as f64);
            output_gradient
        }

        fn stochastic_multithreaded_back_propagation(&self,partitioned_training_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>,batch_size: usize) -> Self::Gradient {
            let mut output_gradient = self.build_zero_gradient();
            crossbeam::scope(|scope| {
                let (original_transmitter,receiver) = mpsc::channel();
                for partition in partitioned_training_data {
                    let cloned_transmitter = original_transmitter.clone();
                    scope.spawn(move |_| {
                        let mut rng = rand::thread_rng();
                        for (training_in,training_out) in partition.choose_multiple(&mut rng, batch_size) {
                            let gradient_pair = self.one_example_back_propagation(training_in, training_out);
                            cloned_transmitter.send(gradient_pair).unwrap();
                        }
                    });
                }
                drop(original_transmitter);
                for gradient in receiver {
                    output_gradient.add(gradient)
                }
            }).unwrap();
            let mut data_length = 0;
            for partition in partitioned_training_data {
                data_length += partition.len()
            }
            output_gradient.div(data_length as f64);
            output_gradient
        }
    
        fn backtracking_line_search_train(
            &mut self,
            training_data: &Vec<(Vec<f64>,Vec<f64>)>,
            first_checked_rate: f64,
            rate_degradation_factor: f64,
            tolerance_parameter: f64,
            minimum_learning_rate: f64,
            max_iterations: u32
        ) {
            let mut previous_test = self.test(training_data);
            println!("{}",previous_test);
            let mut current_learning_rate = first_checked_rate * rate_degradation_factor;
            let mut iteration_num = 0;
            'main_loop: while iteration_num < max_iterations {
                current_learning_rate /= rate_degradation_factor;
                let gradient = self.back_propagation(training_data);
                let tolerable_local_slope = tolerance_parameter * gradient.dot_product(&gradient);
                self.subtract_gradient(&gradient,current_learning_rate);
                let mut new_test = self.test(training_data);
                println!("{}",new_test);
                while previous_test - new_test < current_learning_rate * tolerable_local_slope {
                    println!("Improvement: {}",previous_test-new_test);
                    println!("Required:    {}", current_learning_rate * tolerable_local_slope);
                    current_learning_rate *= rate_degradation_factor;
                    println!("Looping: {}",current_learning_rate);
                    if current_learning_rate < minimum_learning_rate {
                        self.subtract_gradient(&gradient, current_learning_rate / rate_degradation_factor);
                        break 'main_loop;
                    }
                    self.subtract_gradient(&gradient, current_learning_rate-(current_learning_rate/rate_degradation_factor));
                    new_test = self.test(training_data);
                    println!("{}",new_test);
                }
                previous_test = new_test;
                iteration_num += 1;
            }
        }

        fn multithreaded_backtracking_line_search_train(
            &mut self,
            training_data: &Vec<(Vec<f64>,Vec<f64>)>,
            num_partitions: f64,
            first_checked_rate: f64,
            rate_degradation_factor: f64,
            tolerance_parameter: f64,
            minimum_learning_rate: f64,
            max_iterations: u32
        ) {
            let partitioned_training_data = partition_data(training_data, num_partitions);
            let mut previous_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
            println!("{}",previous_test);
            let mut current_learning_rate = first_checked_rate * rate_degradation_factor;
            let mut iteration_num = 0;
            'main_loop: while iteration_num < max_iterations {
                current_learning_rate /= rate_degradation_factor;
                let gradient = self.multithreaded_back_propagation(&partitioned_training_data);
                let tolerable_local_slope = tolerance_parameter * gradient.dot_product(&gradient);
                self.subtract_gradient(&gradient,current_learning_rate);
                let mut new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                println!("{}",new_test);
                while previous_test - new_test < current_learning_rate * tolerable_local_slope {
                    println!("Improvement: {}",previous_test-new_test);
                    println!("Required:    {}", current_learning_rate * tolerable_local_slope);
                    current_learning_rate *= rate_degradation_factor;
                    println!("Looping: {}",current_learning_rate);
                    if current_learning_rate < minimum_learning_rate {
                        self.subtract_gradient(&gradient, current_learning_rate / rate_degradation_factor);
                        break 'main_loop;
                    }
                    self.subtract_gradient(&gradient, current_learning_rate-(current_learning_rate/rate_degradation_factor));
                    new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                    println!("{}",new_test);
                }
                previous_test = new_test;
                iteration_num += 1;
            }
        }
    
        fn stochastic_multithreaded_backtracking_line_search_train(
            &mut self,
            training_data: &Vec<(Vec<f64>,Vec<f64>)>,
            num_partitions: f64,
            batch_size: usize,
            first_checked_rate: f64,
            rate_degradation_factor: f64,
            tolerance_parameter: f64,
            minimum_learning_rate: f64,
            max_iterations: u32
        ) {
            let partitioned_training_data = partition_data(training_data, num_partitions);
            let mut previous_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
            println!("{}",previous_test);
            let mut current_learning_rate = first_checked_rate * rate_degradation_factor;
            let mut iteration_num = 0;
            'main_loop: while iteration_num < max_iterations {
                current_learning_rate /= rate_degradation_factor;
                let gradient = self.stochastic_multithreaded_back_propagation(&partitioned_training_data,batch_size);
                let tolerable_local_slope = tolerance_parameter * gradient.dot_product(&gradient);
                self.subtract_gradient(&gradient,current_learning_rate);
                let mut new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                println!("{}",new_test);
                while previous_test - new_test < current_learning_rate * tolerable_local_slope {
                    println!("Improvement: {}",previous_test-new_test);
                    println!("Required:    {}", current_learning_rate * tolerable_local_slope);
                    current_learning_rate *= rate_degradation_factor;
                    println!("Looping: {}",current_learning_rate);
                    if current_learning_rate < minimum_learning_rate {
                        self.subtract_gradient(&gradient, current_learning_rate / rate_degradation_factor);
                        break 'main_loop;
                    }
                    self.subtract_gradient(&gradient, current_learning_rate-(current_learning_rate/rate_degradation_factor));
                    new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                    println!("{}",new_test);
                }
                previous_test = new_test;
                iteration_num += 1;
            }
        }
    }

    pub trait FlattenableGradient: Network {
        fn flatten_gradient(gradient: Self::Gradient) -> Vec<f64>;
        fn flatten_ref_gradient(gradient: &Self::Gradient) -> Vec<&f64>;
        fn subtract_flat_gradient(&mut self,flat_gradient: &Vec<f64>,learning_rate: f64);
        fn add_flat_gradient(&mut self,flat_gradient: &Vec<f64>,learning_rate: f64);
        fn build_zero_flat_gradient(&self) -> Vec<f64>;
        fn get_num_params(&self) -> usize;
    }

    pub trait FlattenableGradientInherentMethods: FlattenableGradient {
        fn l_bfgs_train(
            &mut self,
            training_data: &Vec<(Vec<f64>,Vec<f64>)>,
            iteration_memory: usize,
            first_checked_rate: f64,
            rate_degradation_factor: f64,
            tolerance_parameter: f64,
            minimum_learning_rate: f64,
            max_iterations: u32
        );
    }

    impl<T: FlattenableGradient> FlattenableGradientInherentMethods for T {
        fn l_bfgs_train(
            &mut self,
            training_data: &Vec<(Vec<f64>,Vec<f64>)>,
            iteration_memory: usize,
            first_checked_rate: f64,
            rate_degradation_factor: f64,
            tolerance_parameter: f64,
            minimum_learning_rate: f64,
            max_iterations: u32
        ) {
            let mut applied_gradient_mem: VecDeque<Vec<f64>> = VecDeque::with_capacity(iteration_memory);
            let mut gradient_change_mem = VecDeque::with_capacity(iteration_memory);
            let mut p_mem = VecDeque::with_capacity(iteration_memory); //-10% know what to call this var
            
            let mut previous_test = self.test(training_data);
            println!("{}",previous_test);

            let gradient = self.back_propagation(training_data);
            let tolerable_local_slope = tolerance_parameter * gradient.dot_product(&gradient);
            let mut current_learning_rate = first_checked_rate;
            self.subtract_gradient(&gradient,current_learning_rate);
            let mut new_test = self.test(training_data);
            println!("{}",new_test);
            while previous_test - new_test < current_learning_rate * tolerable_local_slope {
                println!("Improvement: {}",previous_test-new_test);
                println!("Required:    {}", current_learning_rate * tolerable_local_slope);
                current_learning_rate *= rate_degradation_factor;
                println!("Looping: {}",current_learning_rate);
                if current_learning_rate < minimum_learning_rate {
                    self.subtract_gradient(&gradient, current_learning_rate / rate_degradation_factor);
                    return ();
                }
                self.subtract_gradient(&gradient, current_learning_rate-(current_learning_rate/rate_degradation_factor));
                new_test = self.test(training_data);
                println!("{}",new_test);
            }
            previous_test = new_test;

            let mut applied_gradient = Vec::new();
            for param in <Self>::flatten_ref_gradient(&gradient) {
                applied_gradient.push(param*current_learning_rate);
            }

            applied_gradient_mem.push_back(applied_gradient);
            
            let mut previous_gradient: Vec<f64> = <Self>::flatten_gradient(gradient); 
            'main_loop: for _ in 0..max_iterations-1 {
                let current_gradient = <Self>::flatten_gradient(self.back_propagation(training_data));
                let mut gradient_change_vec = Vec::with_capacity(current_gradient.len());
                for (current_param_der,previous_param_der) in current_gradient.iter().zip(previous_gradient.into_iter()) {
                    gradient_change_vec.push(current_param_der-previous_param_der);
                }
                previous_gradient = current_gradient.clone(); //expensive, not sure if there is another way
                let mut applied_n_change_gradient_dot = 0.0; 
                for (param_der,param_der_change) in applied_gradient_mem[applied_gradient_mem.len()-1].iter().zip(gradient_change_vec.iter()) {
                    applied_n_change_gradient_dot += param_der * param_der_change;
                }
                if p_mem.len() == 10 {
                    p_mem.pop_front();
                    gradient_change_mem.pop_front();
                }
                p_mem.push_back(1.0/applied_n_change_gradient_dot);
                gradient_change_mem.push_back(gradient_change_vec);
                //after this point many var names taken directly from math as i dont know what they mean
                let mut q = current_gradient;
                let mut alpha_vec = Vec::new();
                for (p,(applied_gradient,gradient_change)) in (p_mem.iter().zip(applied_gradient_mem.iter().zip(gradient_change_mem.iter()))).rev() {
                    let mut alpha = 0.0;
                    for (param_der,param_q) in applied_gradient.iter().zip(q.iter()) {
                        alpha += param_der*param_q;
                    }
                    alpha *= p;
                    for (param_q,gradient_param_change) in q.iter_mut().zip(gradient_change.iter()) {
                        *param_q -= alpha * gradient_param_change;
                    }
                    alpha_vec.push(alpha);
                }
                let mut mult_numerator = 0.0;
                let mut mult_denominator = 0.0;
                for (applied_param_der,gradient_param_change) in applied_gradient_mem[applied_gradient_mem.len()-1].iter().zip(gradient_change_mem[gradient_change_mem.len()-1].iter()) {
                    mult_numerator += applied_param_der*gradient_param_change;
                    mult_denominator += gradient_param_change*gradient_param_change;
                }
                let mult = mult_numerator/mult_denominator;
                let mut z  = Vec::new();
                for param in q {
                    z.push(param*mult);
                }
                for (p,(applied_gradient,(gradient_change,alpha))) in p_mem.iter().zip(applied_gradient_mem.iter().zip(gradient_change_mem.iter().zip(alpha_vec.into_iter()))) {
                    let mut beta = 0.0;
                    for (gradient_param_change,param_z) in gradient_change.iter().zip(z.iter()) {
                        beta += gradient_param_change * param_z;
                    }
                    beta *= p;
                    let alpha_beta_dif = alpha-beta;
                    for (param_z,applied_param_der) in z.iter_mut().zip(applied_gradient.iter()) {
                        *param_z += applied_param_der*alpha_beta_dif;
                    }
                }
                for param_z in z.iter_mut() {
                    *param_z *= -1.0;
                }

                let mut tolerable_local_slope = 0.0;
                for (param_der,param_z) in previous_gradient.iter().zip(z.iter()) {
                    tolerable_local_slope += param_der * param_z;
                }
                tolerable_local_slope *= tolerance_parameter;
                let mut current_learning_rate = first_checked_rate;
                self.subtract_flat_gradient(&z,current_learning_rate);
                let mut new_test = self.test(training_data);
                println!("{}",new_test);
                while previous_test - new_test < current_learning_rate * tolerable_local_slope {
                    println!("Improvement: {}",previous_test-new_test);
                    println!("Required:    {}", current_learning_rate * tolerable_local_slope);
                    current_learning_rate *= rate_degradation_factor;
                    println!("Looping: {}",current_learning_rate);
                    if current_learning_rate < minimum_learning_rate {
                        self.subtract_flat_gradient(&z, current_learning_rate / rate_degradation_factor);
                        break 'main_loop;
                    }
                    self.subtract_flat_gradient(&z, current_learning_rate-(current_learning_rate/rate_degradation_factor));
                    new_test = self.test(training_data);
                    println!("{}",new_test);
                }
                previous_test = new_test;
                if applied_gradient_mem.len() == 10 {
                    applied_gradient_mem.pop_front();
                }

                for param_z in z.iter_mut() {
                    *param_z *= current_learning_rate;
                }
                applied_gradient_mem.push_back(z);
            }

            

        }
    }

    pub struct ActivationFn(pub fn(f64) -> f64,pub fn(f64) -> f64);

    pub mod activation_fn_backend {
        pub fn sigmoid_call(x: f64) -> f64 {
            1.0/(1.0+(-x).exp())
        }
        pub fn sigmoid_call_der(x: f64) -> f64 {
            let temp_val = 1.0/(1.0+(-x).exp());
            temp_val*(1.0*temp_val)
        }
    
        pub fn linear_call(x: f64) -> f64 {
            x
        }
        pub fn linear_call_der(_: f64) -> f64 {
            1.0
        }

        pub fn relu_call(x: f64) -> f64 {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }
        pub fn relu_call_der(x: f64) -> f64 {
            if x > 0.0 {
                1.0
            } else if x == 0.0{
                0.5 //really undefined
            } else {
                0.0
            }
        }
    
        pub fn binary_step_call(x: f64) -> f64 {
            if x > 0.0 {1.0} else {0.0}
        }
        pub fn binary_step_call_der(_: f64) -> f64 {
            0.0
        }

        pub fn tanh_call(x: f64) -> f64 {
            let e_to_x = x.exp();
            let e_to_neg_x = 1.0/(e_to_x);
            (e_to_x-e_to_neg_x)/(e_to_x+e_to_neg_x)
        }
        pub fn tanh_call_der(x: f64) -> f64 {
            1.0 - tanh_call(x)
        }

        pub fn leaky_relu_call(x: f64) -> f64 {
            if x > 0.0 {x} else {0.01*x}
        }
        pub fn leaky_relu_call_der(x: f64) -> f64 {
            if x > 0.0 {1.0} else if x == 0.0 {0.505} else {0.01} //2nd is technically undefined
        }
    
        pub fn silu_call(x: f64) -> f64 {
            x/(1.0+(-x).exp())
        }
        pub fn silu_call_der(x: f64) -> f64 {
            let e_to_neg_x = (-x).exp();
            let one_plus_e_to_neg_x = 1.0+e_to_neg_x;
            (one_plus_e_to_neg_x+x*e_to_neg_x)/(one_plus_e_to_neg_x*one_plus_e_to_neg_x)
        }

        pub fn gaussian_call(x: f64) -> f64 {
            (-x*x).exp()
        }
        pub fn gaussian_call_der(x: f64) -> f64 {
            -2.0 * x * gaussian_call(x)
        }
    }

    pub const SIGMOID: ActivationFn = ActivationFn(activation_fn_backend::sigmoid_call,activation_fn_backend::sigmoid_call_der);
    pub const LINEAR: ActivationFn = ActivationFn(activation_fn_backend::linear_call,activation_fn_backend::linear_call_der);
    pub const RELU: ActivationFn =  ActivationFn(activation_fn_backend::relu_call,activation_fn_backend::relu_call_der);
    pub const BINARY_STEP: ActivationFn = ActivationFn(activation_fn_backend::binary_step_call,activation_fn_backend::binary_step_call_der);
    pub const TANH: ActivationFn = ActivationFn(activation_fn_backend::tanh_call,activation_fn_backend::tanh_call_der);
    pub const LEAKY_RELU: ActivationFn = ActivationFn(activation_fn_backend::leaky_relu_call,activation_fn_backend::leaky_relu_call_der);
    pub const SILU: ActivationFn = ActivationFn(activation_fn_backend::silu_call,activation_fn_backend::silu_call_der);
    pub const GAUSSIAN: ActivationFn = ActivationFn(activation_fn_backend::gaussian_call,activation_fn_backend::gaussian_call_der);
    //pub const : ActivationFn = ActivationFn(activation_fn_backend::_call,activation_fn_backend::_call_der);

    //here be actual structs

    pub struct MultiLayerPerceptron {
        biases: Vec<Vec<f64>>,
        weights: Vec<Vec<Vec<f64>>>,
        node_layout: Vec<usize>, //May be better to use lens from biases, hardly would matter though, probably
        activation_fn: fn(f64) -> f64,
        derivative_activation_fn: fn(f64) -> f64
    }

    impl FileSupport for MultiLayerPerceptron {
        type ExtraDecodingInfo = ActivationFn;

        fn into_bytes(self) -> Vec<u8> {
            /* turns a NeuralNetwork into the following file format:
            All nums are as BE bytes

            Offset 0: usize in 8 bytes denoting the length of the network; let this be x
            Offset 8: usize in 8 bytes denoting the first num in nodeLayout
            Offset 16: usize in 8 bytes denoting the second num in nodeLayout
            ...
            Offset 8x: usize in 8 bytes denoting the last num in nodeLayout
            Offset 8x+8: u8 corresponding to the activation_fn 

            The follwing bytes are then all f64's of self.biases and self.weights in that order
            they are ordered as the result of iterating through all the Vecs in them recursively
            The exact layout of how these correspond to the actual biases and weights can be found
            through nodelayout
            */
            let mut output_bytes = Vec::new();
            for byte in usize_to_8_be_bytes(self.node_layout.len()) {
                output_bytes.push(byte);
            }
            for layer in self.node_layout {
                for byte in usize_to_8_be_bytes(layer) {
                    output_bytes.push(byte);
                }
            }
            for layer in self.biases {
                for bias in layer {
                    for byte in bias.to_be_bytes() {
                        output_bytes.push(byte);
                    }
                }
            }
            for layer in self.weights {
                for node in layer {
                    for weight in node {
                        for byte in weight.to_be_bytes() {
                            output_bytes.push(byte);
                        }
                    }
                }
            }
            output_bytes
        }
        fn from_bytes(bytes: Vec<u8>,activation_fn: Self::ExtraDecodingInfo) -> Self {
            let mut true_byte_num = 0;
            let mut eight_byte_buffer = [0; 8];
            let mut network_length = 0;
            let mut node_layout = Vec::new();
            let mut last_bias_byte_num = 0;
            let mut biases_buffer = Vec::new();
            let mut weights_buffer = Vec::new();
            for byte in bytes {
                match true_byte_num {
                    0..=6 => {
                        eight_byte_buffer[true_byte_num] = byte;
                        true_byte_num += 1;
                    }
                    7 => {
                        eight_byte_buffer[true_byte_num] = byte;
                        network_length = usize::from_be_bytes(eight_byte_buffer);
                        network_length *= 8; //JANK
                        network_length += 7; //JANK
                        true_byte_num += 1;
                    }
                    8.. if true_byte_num <= network_length => {
                        eight_byte_buffer[true_byte_num % 8] = byte;
                        if true_byte_num % 8 == 7 {
                            node_layout.push(usize::from_be_bytes(eight_byte_buffer));
                        }
                        if true_byte_num == network_length {
                            for layer in node_layout[1..node_layout.len()].iter() {
                                last_bias_byte_num += layer;
                            }
                            last_bias_byte_num *= 8; //Jank
                            last_bias_byte_num += network_length;
                        }
                        true_byte_num += 1;
                    }
                    8.. if (true_byte_num > network_length) && (true_byte_num <= last_bias_byte_num) => { //should still work but annoying
                        eight_byte_buffer[true_byte_num % 8] = byte;
                        if true_byte_num % 8 == 7 {
                            biases_buffer.push(f64::from_be_bytes(eight_byte_buffer));
                        }
                        true_byte_num += 1;
                    }
                    8.. if (true_byte_num > network_length) && (true_byte_num > last_bias_byte_num) => {
                        eight_byte_buffer[true_byte_num % 8] = byte;
                        if true_byte_num % 8 == 7 {
                            weights_buffer.push(f64::from_be_bytes(eight_byte_buffer));
                        }
                        true_byte_num += 1;
                    }
                    _ => {
                        println!("length: {}\nlast_bias_num: {}",network_length,last_bias_byte_num);
                        panic!("Some byte failed to match")
                    }
                }
            }
            let mut biases = Vec::new();
            let mut layer_biases = Vec::new();
            let mut current_layer_num = 1;
            for num in biases_buffer {
                layer_biases.push(num);
                if layer_biases.len() == node_layout[current_layer_num] {
                    biases.push(layer_biases);
                    layer_biases = Vec::new();
                    current_layer_num += 1;
                }
            }
            let mut weights = Vec::new();
            let mut layer_weights = Vec::new();
            let mut node_weights = Vec::new();
            let mut current_layer_num = 1;
            for num in weights_buffer {
                node_weights.push(num);
                if node_weights.len() == node_layout[current_layer_num-1] {
                    layer_weights.push(node_weights);
                    node_weights = Vec::new();
                    if layer_weights.len() == node_layout[current_layer_num] {
                        weights.push(layer_weights);
                        layer_weights = Vec::new();
                        current_layer_num += 1;
                    }
                }
            }
            MultiLayerPerceptron {
                biases: biases,
                weights: weights,
                node_layout: node_layout,
                activation_fn: activation_fn.0,
                derivative_activation_fn: activation_fn.1
            }
        }
    }

    impl MultiLayerPerceptron {
        pub fn new(node_layout: Vec<usize>,activation_fn: ActivationFn,bias_bounds: f64,weight_bounds: f64) -> Self {
            let mut rng = rand::thread_rng();
            let mut biases = Vec::with_capacity(node_layout.len());
            let mut weights = Vec::with_capacity(node_layout.len());
            for (layer, previous_layer) in (&node_layout[1..node_layout.len()]).iter().zip((&node_layout[0..node_layout.len()-1]).iter()) { //first iteration is on the 2nd layer, layer num is 0 and refers to 1st layer making the previous layer
                let mut layer_biases = Vec::with_capacity(*layer);
                let mut layer_weights = Vec::with_capacity(*layer);
                for _ in 0..*layer {
                    layer_biases.push(2.0*bias_bounds*rng.gen::<f64>()-bias_bounds);
                    let mut node_weights = Vec::with_capacity(*previous_layer);
                    for _ in 0..*previous_layer {
                        node_weights.push(2.0*weight_bounds*rng.gen::<f64>()-weight_bounds);
                    }
                    layer_weights.push(node_weights);
                }
                biases.push(layer_biases);
                weights.push(layer_weights);
            }
            MultiLayerPerceptron {
                biases: biases,
                weights: weights,
                node_layout: node_layout,
                activation_fn: activation_fn.0,
                derivative_activation_fn: activation_fn.1
            }
        }
    }

    impl Network for MultiLayerPerceptron {
        type Gradient = (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>);

        fn run(&self, inputs: &[f64]) -> Vec<f64> {
            let mut layer_activations = Vec::with_capacity(self.node_layout[1]);
            for (node_bias,node_weights) in self.biases[0].iter().zip(self.weights[0].iter()) {
                let mut node_val_before_activation_fn = *node_bias;
                for (weight,previous_node_activation) in node_weights.iter().zip(inputs.iter()) {
                    node_val_before_activation_fn += weight*previous_node_activation;
                }
                layer_activations.push((self.activation_fn)(node_val_before_activation_fn));
            }
            for (layer_biases,layer_weights) in self.biases[1..self.biases.len()].iter().zip(self.weights[1..self.weights.len()].iter()) {
                let mut new_layer_activations = Vec::with_capacity(layer_biases.len());
                for (node_bias,node_weights) in layer_biases.iter().zip(layer_weights.iter()) {
                    let mut node_val_before_activation_fn = *node_bias;
                    for (weight,previous_node_activation) in node_weights.iter().zip(layer_activations.iter()) {
                        node_val_before_activation_fn += weight*previous_node_activation;
                    }
                    new_layer_activations.push((self.activation_fn)(node_val_before_activation_fn));
                }
                layer_activations = new_layer_activations;
            }
            layer_activations
        }

        fn subtract_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64) {
            for (layer_biases,layer_biases_gradient) in self.biases.iter_mut().zip(gradient.0.iter()) {
                for (bias, bias_gradient) in layer_biases.iter_mut().zip(layer_biases_gradient.iter()) {
                    *bias -= bias_gradient*learning_rate;
                }
            }
            for (layer_weights,layer_weights_gradient) in self.weights.iter_mut().zip(gradient.1.iter()) {
                for (node_weights,node_weights_gradient) in layer_weights.iter_mut().zip(layer_weights_gradient.iter()) {
                    for (weight, weight_gradient) in node_weights.iter_mut().zip(node_weights_gradient.iter()) {
                        *weight -= weight_gradient*learning_rate
                    }
                }
            }
        }

        fn add_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64) {
            for (layer_biases,layer_biases_gradient) in self.biases.iter_mut().zip(gradient.0.iter()) {
                for (bias, bias_gradient) in layer_biases.iter_mut().zip(layer_biases_gradient.iter()) {
                    *bias += bias_gradient*learning_rate;
                }
            }
            for (layer_weights,layer_weights_gradient) in self.weights.iter_mut().zip(gradient.1.iter()) {
                for (node_weights,node_weights_gradient) in layer_weights.iter_mut().zip(layer_weights_gradient.iter()) {
                    for (weight, weight_gradient) in node_weights.iter_mut().zip(node_weights_gradient.iter()) {
                        *weight += weight_gradient*learning_rate
                    }
                }
            }
        }

        fn build_zero_gradient(&self) -> Self::Gradient {
            let mut biases = Vec::with_capacity(self.node_layout.len());
            let mut weights = Vec::with_capacity(self.node_layout.len());
            for (layer, previous_layer) in (&self.node_layout[1..self.node_layout.len()]).iter().zip((&self.node_layout[0..self.node_layout.len()-1]).iter()) { //first iteration is on the 2nd layer, layer num is 0 and refers to 1st layer making the previous layer
                let mut layer_biases = Vec::with_capacity(*layer);
                let mut layer_weights = Vec::with_capacity(*layer);
                for _ in 0..*layer {
                    layer_biases.push(0.0);
                    let mut node_weights = Vec::with_capacity(*previous_layer);
                    for _ in 0..*previous_layer {
                        node_weights.push(0.0);
                    }
                    layer_weights.push(node_weights);
                }
                biases.push(layer_biases);
                weights.push(layer_weights);
            }
            (biases,weights)
        }

        //TO DO: Check with_capacity's
        fn one_example_back_propagation(&self,training_in: &[f64],training_out: &[f64]) -> Self::Gradient {
            let mut biases_gradient = Vec::with_capacity(self.node_layout.len());
            let mut weights_gradient = Vec::with_capacity(self.node_layout.len());
            
            let (network_activations,network_pre_activations) = {
                let mut network_pre_activations = Vec::with_capacity(self.node_layout.len()-1);
                let mut network_activations = Vec::with_capacity(self.node_layout.len());
                network_activations.push(training_in.to_vec()); //remove this is possible
                let mut layer_activations = Vec::with_capacity(self.node_layout[1]);
                let mut layer_pre_activations = Vec::with_capacity(self.node_layout[1]);
                for (node_bias,node_weights) in self.biases[0].iter().zip(self.weights[0].iter()) {
                    let mut node_val_before_activation_fn = *node_bias;
                    for (weight,previous_node_activation) in node_weights.iter().zip(training_in.iter()) {
                        node_val_before_activation_fn += weight*previous_node_activation;
                    }
                    layer_pre_activations.push(node_val_before_activation_fn);
                    layer_activations.push((self.activation_fn)(node_val_before_activation_fn));
                }
                for (layer_biases,layer_weights) in self.biases[1..self.biases.len()].iter().zip(self.weights[1..self.weights.len()].iter()) {
                    let mut new_layer_activations = Vec::with_capacity(layer_biases.len());
                    let mut new_layer_pre_activations = Vec::with_capacity(layer_biases.len());
                    for (node_bias,node_weights) in layer_biases.iter().zip(layer_weights.iter()) {
                        let mut node_val_before_activation_fn = *node_bias;
                        for (weight,previous_node_activation) in node_weights.iter().zip(layer_activations.iter()) {
                            node_val_before_activation_fn += weight*previous_node_activation;
                        }
                        new_layer_pre_activations.push(node_val_before_activation_fn);
                        new_layer_activations.push((self.activation_fn)(node_val_before_activation_fn));
                    }
                    network_pre_activations.push(layer_pre_activations);
                    layer_pre_activations = new_layer_pre_activations;
                    network_activations.push(layer_activations);
                    layer_activations = new_layer_activations;
                }
                network_pre_activations.push(layer_pre_activations);
                network_activations.push(layer_activations);
    
                (network_activations,network_pre_activations)
            }; //special run  but inlined

            //last layer specific calculations
            let mut last_layer_biases_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-1]);
            let mut last_layer_weights_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-1]);
            for (pre_activation,(expected_val,real_val)) in 
            network_pre_activations[network_pre_activations.len()-1].iter().zip(
            training_out.iter().zip(
            network_activations[network_activations.len()-1].iter())) {
                let current_node_derivative = 2.0*(real_val-expected_val)*(self.derivative_activation_fn)(*pre_activation);
                last_layer_biases_gradient.push(current_node_derivative);
                let mut node_weights_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-2]);
                for previous_node_activation in network_activations[network_activations.len()-2].iter() {
                    node_weights_gradient.push(previous_node_activation*current_node_derivative);
                }
                last_layer_weights_gradient.push(node_weights_gradient);
            }
            biases_gradient.push(last_layer_biases_gradient);
            weights_gradient.push(last_layer_weights_gradient);
            //All other layer calculations
            for (forward_layer_weights,(current_pre_activations,previous_activations)) in 
            self.weights.iter().rev().zip(
            network_pre_activations[0..network_pre_activations.len()-1].iter().rev().zip(
            network_activations[0..network_activations.len()-2].iter().rev())) {
                let mut layer_biases_gradient = Vec::with_capacity(current_pre_activations.len());
                let mut layer_weights_gradient = Vec::with_capacity(current_pre_activations.len());
                for (input_node_weights,node_pre_activation) in transpose_matrix(forward_layer_weights).iter().zip(current_pre_activations.iter()) {
                    let mut new_node_derivative = 0.0;
                    for (weight,forward_derivative) in input_node_weights.iter().zip(biases_gradient[biases_gradient.len()-1].iter()) {//layer_node_derivatives.iter()) { //seems to work
                        new_node_derivative += *weight * forward_derivative;
                    }
                    new_node_derivative *= (self.derivative_activation_fn)(*node_pre_activation);
                    layer_biases_gradient.push(new_node_derivative);
                    let mut node_weights_gradients = Vec::with_capacity(previous_activations.len());
                    for previous_node_activation in previous_activations {
                        node_weights_gradients.push(previous_node_activation*new_node_derivative)
                    }
                    layer_weights_gradient.push(node_weights_gradients);
                }
                biases_gradient.push(layer_biases_gradient);
                weights_gradient.push(layer_weights_gradient);
            }
            biases_gradient = biases_gradient.into_iter().rev().collect();
            weights_gradient = weights_gradient.into_iter().rev().collect();

            (biases_gradient,weights_gradient)
        }
    
    }
    
    impl FlattenableGradient for MultiLayerPerceptron{
        fn flatten_gradient(gradient: Self::Gradient) -> Vec<f64> {
            let mut output = Vec::new();
            for layer_biases in gradient.0 {
                for bias in layer_biases {
                    output.push(bias);
                }
            }
            for layer_weights in gradient.1 {
                for node_weights in layer_weights {
                    for weight in node_weights {
                        output.push(weight);
                    }
                }
            }
            output
        }

        fn flatten_ref_gradient(gradient: &Self::Gradient) -> Vec<&f64> {
            let mut output = Vec::new();
            for layer_biases in &gradient.0 {
                for bias in layer_biases {
                    output.push(bias);
                }
            }
            for layer_weights in &gradient.1 {
                for node_weights in layer_weights {
                    for weight in node_weights {
                        output.push(weight);
                    }
                }
            }
            output
        }

        fn subtract_flat_gradient(&mut self,flat_gradient: &Vec<f64>,learning_rate: f64) {
            let mut flat_gradient_iter = flat_gradient.iter();
            for layer_biases in &mut self.biases {
                for bias in layer_biases {
                    *bias -= match flat_gradient_iter.next() {Some(x) => x, None => {panic!("flattened gradient doesn't match the network")}} * learning_rate;
                }
            }
            for layer_weights in &mut self.weights {
                for node_weights in layer_weights {
                    for weight in node_weights {
                        *weight -= match flat_gradient_iter.next() {Some(x) => x, None => {panic!("flattened gradient doesn't match the network")}} * learning_rate;
                    }
                }
            }
        }

        fn add_flat_gradient(&mut self,flat_gradient: &Vec<f64>,learning_rate: f64) {
            let mut flat_gradient_iter = flat_gradient.iter();
            for layer_biases in &mut self.biases {
                for bias in layer_biases {
                    *bias += match flat_gradient_iter.next() {Some(x) => x, None => {panic!("flattened gradient doesn't match the network")}} * learning_rate;
                }
            }
            for layer_weights in &mut self.weights {
                for node_weights in layer_weights {
                    for weight in node_weights {
                        *weight += match flat_gradient_iter.next() {Some(x) => x, None => {panic!("flattened gradient doesn't match the network")}} * learning_rate;
                    }
                }
            }
        }

        fn build_zero_flat_gradient(&self) -> Vec<f64> {
            let mut output = Vec::new();
            for _ in 0..self.get_num_params() {
                output.push(0.0);
            }
            output
        }

        //May be better to just store this in a variable, though its like no performance loss
        fn get_num_params(&self) -> usize {
            let mut num_params = 0;
            for (previous_layer,current_layer) in self.node_layout[0..self.node_layout.len()-1].iter().zip(self.node_layout[1..self.node_layout.len()].iter()) {
                num_params += previous_layer * current_layer + current_layer;
            }
            num_params
        }


    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_into_and_from_bytes() {
        let nn = aiz::NeuralNetwork::new(vec![10,5,20,16],1.0,1.0);
        assert_eq!(nn,aiz::NeuralNetwork::from_bytes(nn.clone().into_bytes()));
    }
}