
pub mod aiz {
    use rand::Rng;
    use rand::seq::SliceRandom;
    use crossbeam;
    use std::sync::mpsc;

    //should be optimizable
    //may be possible to have it do output Vec<Vec<&T>> and avoid needing Copy and possibly being faster
    //will need to check
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

    pub fn flip_owned_matrix<T>(matrix: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let mut output_vec: Vec<Vec<T>> = Vec::new();
        for _ in 0..matrix[0].len() {
            output_vec.push(Vec::new())
        }
        for vec in matrix {
            for (val, current_output_vec) in vec.into_iter().zip(output_vec.iter_mut()) {
                current_output_vec.push(val)
            }
        }
        output_vec
    }

    fn find_greatest_movement_in_gradients(biases_gradient: &Vec<Vec<f64>>,weights_gradient: &Vec<Vec<Vec<f64>>>) -> f64{
        let mut greatest_movement = 0.0;
        for column in biases_gradient.iter() {
            for bias in column {
                if greatest_movement < bias.abs() {
                    greatest_movement = bias.abs();
                }
            }
        }
        for layer in weights_gradient.iter() {
            for in_node in layer {
                for weight in in_node {
                    if greatest_movement < weight.abs() {
                        greatest_movement = weight.abs();
                    }
                }
            }
        }

        greatest_movement
    }

    pub fn partition_data<T>(data: &Vec<T>, num_partitions: f64) -> Vec<Vec<&T>> {
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

    pub struct NeuralNetwork {
        biases: Vec<Vec<f64>>,
        weights: Vec<Vec<Vec<f64>>>,
        node_layout: Vec<usize>
    }

    impl NeuralNetwork {
        pub fn new(node_layout: Vec<usize>) -> Self {
            let mut rng = rand::thread_rng();
            let length = node_layout.len();
            let mut biases = Vec::with_capacity(length-1);
            for layer_size in node_layout[1..length].iter() {
                let mut layer_biases = Vec::with_capacity(*layer_size);
                for _ in 0..*layer_size {
                    layer_biases.push(2.0*rng.gen::<f64>()-1.0); //see if rand doc has a fn for this kind of thing
                }
                biases.push(layer_biases);
            }
            let mut weights = Vec::with_capacity(length-1);
            for (previous_layer_size, layer_size) in node_layout[0..length-1].iter().zip(node_layout[1..length].iter()) {
                let mut layer_weights = Vec::with_capacity(*previous_layer_size);
                for _ in 0..*previous_layer_size {
                    let mut node_weights = Vec::with_capacity(*layer_size);
                    for _ in 0..*layer_size {
                        node_weights.push(2.0*rng.gen::<f64>()-1.0);
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

        pub fn run(&self,inputs: &Vec<f64>) -> Vec<f64> {
            let mut current_layer = Vec::new();
            for (row_weights,row_bias) in flip_matrix(&self.weights[0]).iter().zip(&self.biases[0]) {
                current_layer.push(self.activation_function(row_weights.iter()
                                                        .zip(inputs
                                                        .iter())
                                                        .map(|(weight,activation)| weight*activation)
                                                        .sum::<f64>()+row_bias));
            }
            let mut layer_num = 1;
            for (layer_weights, layer_biases) in self.weights[1..self.weights.len()].iter().zip(self.biases[1..self.weights.len()].iter()) {
                layer_num += 1;
                let mut new_current_layer = Vec::with_capacity(self.node_layout[layer_num]);
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
        
        pub fn special_run(&self,inputs: &Vec<f64>) -> Vec<Vec<f64>> {
            let mut output = Vec::new();
            output.push(inputs.clone());
            let mut current_layer = Vec::new();
            for (row_weights,row_bias) in flip_matrix(&self.weights[0]).iter().zip(&self.biases[0]) {
                current_layer.push(self.activation_function(row_weights.iter()
                                                        .zip(inputs
                                                        .iter())
                                                        .map(|(weight,activation)| weight*activation)
                                                        .sum::<f64>()+row_bias));
            }
            let mut layer_num = 1;
            for (layer_weights, layer_biases) in self.weights[1..self.weights.len()].iter().zip(self.biases[1..self.weights.len()].iter()) {
                layer_num += 1;
                let mut new_current_layer = Vec::with_capacity(self.node_layout[layer_num]);
                for (row_weights,row_bias) in flip_matrix(layer_weights).iter().zip(layer_biases.iter()) {
                    new_current_layer.push(self.activation_function(row_weights.iter()
                                                            .zip(current_layer
                                                            .iter())
                                                            .map(|(weight,activation)| weight*activation)
                                                            .sum::<f64>()+row_bias));
                }
                output.push(current_layer);
                current_layer = new_current_layer;
            }
            output.push(current_layer);
            output
        }

        //I dont see a reason why an end-user would need these 2
        fn activation_function(&self, x: f64) -> f64 {
            1.0/(1.0+(-x).exp())
        }

        fn derivative_activation_function(&self, x: f64) -> f64 {
            let intermedite_num = self.activation_function(x);
            intermedite_num*(1.0-intermedite_num)
            //see if using a variable is faster than not using it
        }

        pub fn test(&self,test_data: &Vec<(Vec<f64>,Vec<f64>)>) -> f64 {
            let mut all_costs = Vec::new();
            for (inputs,expected_outputs) in test_data {
                let experimental_outputs = self.run(inputs);
                all_costs.push(
                    expected_outputs.iter()
                    .zip(experimental_outputs.iter())
                    .map(|(single_experimental_out,single_expected_out)| {
                        let dif = single_experimental_out-single_expected_out;
                        dif*dif})
                    .sum::<f64>());
            }
            all_costs.iter().sum::<f64>() / all_costs.len() as f64 //there may be a way to do this without converting to f64, albeit this is a small performance hit, 
        }

        pub fn multithreaded_test(&self,test_data: &Vec<(Vec<f64>,Vec<f64>)>,num_test_data_partitions: f64) -> f64 {
            self.prepartitioned_multithreaded_test(&partition_data(test_data, num_test_data_partitions))
        }

        pub fn prepartitioned_multithreaded_test(&self, test_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>) -> f64 {
            let mut total_cost = 0.0;
            crossbeam::scope(|scope| {
                let (original_transmitter,receiver) = mpsc::channel();
                for partition in test_data {
                    let thread_transmitter = original_transmitter.clone();
                    scope.spawn(move |_| {
                        for (test_in,test_expected_out) in partition {
                            let test_experimental_out = self.run(test_in);
                            let cost = test_expected_out.iter()
                                .zip(test_experimental_out.iter())
                                .map(|(single_experimental_out,single_expected_out)| {
                                    let dif = single_experimental_out-single_expected_out;
                                    dif*dif
                                })
                                .sum::<f64>();
                            
                            thread_transmitter.send(cost).unwrap();
                        }
                    });
                };
                drop(original_transmitter); //so it doesn't block the reciever

                for cost in receiver {
                    total_cost += cost;
                }
            }).unwrap();
            let mut num_examples = 0;
            for partition in test_data {
                num_examples += partition.len();
            }
            total_cost/num_examples as f64
        }

        fn apply_gradient(&mut self,biases_gradient: &Vec<Vec<f64>>,weights_gradient: &Vec<Vec<Vec<f64>>>,multiplier: f64) {
            for (column_biases, column_gradient) in self.biases.iter_mut().zip(biases_gradient.iter()) {
                for (bias, movement) in column_biases.iter_mut().zip(column_gradient.iter()) {
                    *bias -= movement*multiplier;
                }
            }
            for (layer_weights, layer_gradient) in self.weights.iter_mut().zip(weights_gradient.iter()) {
                for (in_node_weights,in_node_gradient) in layer_weights.iter_mut().zip(layer_gradient.iter()) {
                    for (weight, movement) in in_node_weights.iter_mut().zip(in_node_gradient.iter()) {
                        *weight -= movement*multiplier;
                    }
                }
            }
        }

        fn one_example_back_propagation(&self,training_ex_in: &Vec<f64>,training_ex_out: &Vec<f64>) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            let mut biases_gradient = Vec::new();
            let mut weights_gradient = Vec::new();
            let layer_vals = self.special_run(training_ex_in);
            let mut layer_node_multipliers = Vec::new(); //there may be a specific func for this
            for _ in 0..layer_vals[layer_vals.len()-1].len() {
                layer_node_multipliers.push(1.0);
            }
            let temp_layer_vals: Vec<&Vec<f64>> = layer_vals[0..layer_vals.len()].iter().rev().collect(); //JANK, originally to avoid lazyness in python
            let mut temp_layer_node_multpliers: Vec<f64>;
            for (current_layer_vals,(layer_weights,(layer_biases,(layer_num,previous_layer_vals)))) in 
            layer_vals.iter().rev().zip(self.weights.iter().rev().zip(self.biases.iter().rev().zip((0..layer_vals.len()).into_iter().zip(temp_layer_vals.iter())))) {
                let mut layer_biases_gradient = Vec::new();
                let mut layer_weights_gradient = Vec::new();
                temp_layer_node_multpliers = Vec::new();
                if layer_num == 0 {
                    for (out_node_val,(out_node_back_weights,(out_node_bias,out_node_expected_val))) in
                    current_layer_vals.iter().zip(flip_matrix(layer_weights).iter().zip(layer_biases.iter().zip(training_ex_out.iter()))) {
                        let out_node_multiplier = self.derivative_activation_function(previous_layer_vals.iter()
                                                                                            .zip(out_node_back_weights.iter())
                                                                                            .map(|(previous_activation,weight)| previous_activation*weight)
                                                                                            .sum::<f64>() 
                                                                                            + out_node_bias) * 2.0 * (out_node_val-out_node_expected_val);
                        layer_biases_gradient.push(out_node_multiplier);
                        layer_weights_gradient.push(previous_layer_vals.iter().map(|in_node_activation| in_node_activation*out_node_multiplier).collect::<Vec<f64>>());
                        temp_layer_node_multpliers.push(out_node_multiplier);
                    }
                } else {
                    for (out_node_back_weights,(out_node_bias,out_node_multiplier)) in 
                    flip_matrix(layer_weights).iter().zip(layer_biases.iter().zip(layer_node_multipliers.iter())) {
                        let out_node_multiplier = self.derivative_activation_function(previous_layer_vals.iter()
                                                                                            .zip(out_node_back_weights.iter())
                                                                                            .map(|(previous_activation,weight)| previous_activation*weight)
                                                                                            .sum::<f64>() 
                                                                                            + out_node_bias) * out_node_multiplier;
                        layer_biases_gradient.push(out_node_multiplier);
                        layer_weights_gradient.push(previous_layer_vals.iter().map(|in_node_activation| in_node_activation*out_node_multiplier).collect::<Vec<f64>>());
                        temp_layer_node_multpliers.push(out_node_multiplier);
                    }
                }
                biases_gradient.push(layer_biases_gradient);
                weights_gradient.push(layer_weights_gradient);
                let mut new_layer_node_multipliers = Vec::new();
                for in_node in layer_weights {
                    new_layer_node_multipliers.push(in_node.iter().zip(temp_layer_node_multpliers.iter()).map(|(weight,multiplier)| weight*multiplier).sum::<f64>());
                }
                layer_node_multipliers = new_layer_node_multipliers;
            }
            (biases_gradient,weights_gradient)
        }

        fn core_back_propagation(&self,training_data: &Vec<(Vec<f64>,Vec<f64>)>) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            let length = self.node_layout.len();
            let mut final_biases_gradient = Vec::with_capacity(length-1);
            for layer_size in self.node_layout[1..length].iter() {
                let mut layer_biases = Vec::with_capacity(*layer_size);
                for _ in 0..*layer_size {
                    layer_biases.push(0.0); 
                }
                final_biases_gradient.push(layer_biases);
            }
            let mut final_weights_gradient = Vec::with_capacity(length-1);
            for (previous_layer_size, layer_size) in self.node_layout[0..length-1].iter().zip(self.node_layout[1..length].iter()) {
                let mut layer_weights = Vec::with_capacity(*previous_layer_size);
                for _ in 0..*previous_layer_size {
                    let mut node_weights = Vec::with_capacity(*layer_size);
                    for _ in 0..*layer_size {
                        node_weights.push(0.0);
                    }
                    layer_weights.push(node_weights);
                }
                final_weights_gradient.push(layer_weights);
            }
            for (training_ex_in,training_ex_out) in training_data {
                /*
                let mut biases_gradient = Vec::new();
                let mut weights_gradient = Vec::new();
                let layer_vals = self.special_run(training_ex_in);
                let mut layer_node_multipliers = Vec::new(); //there may be a specific func for this
                for _ in 0..layer_vals[layer_vals.len()-1].len() {
                    layer_node_multipliers.push(1.0);
                }
                let temp_layer_vals: Vec<&Vec<f64>> = layer_vals[0..layer_vals.len()].iter().rev().collect(); //JANK, originally to avoid lazyness in python
                let mut temp_layer_node_multpliers: Vec<f64>;
                for (current_layer_vals,(layer_weights,(layer_biases,(layer_num,previous_layer_vals)))) in 
                layer_vals.iter().rev().zip(self.weights.iter().rev().zip(self.biases.iter().rev().zip((0..layer_vals.len()).into_iter().zip(temp_layer_vals.iter())))) {
                    let mut layer_biases_gradient = Vec::new();
                    let mut layer_weights_gradient = Vec::new();
                    temp_layer_node_multpliers = Vec::new();
                    if layer_num == 0 {
                        for (out_node_val,(out_node_back_weights,(out_node_bias,out_node_expected_val))) in
                        current_layer_vals.iter().zip(flip_matrix(layer_weights).iter().zip(layer_biases.iter().zip(training_ex_out.iter()))) {
                            let out_node_multiplier = self.derivative_activation_function(previous_layer_vals.iter()
                                                                                                .zip(out_node_back_weights.iter())
                                                                                                .map(|(previous_activation,weight)| previous_activation*weight)
                                                                                                .sum::<f64>() 
                                                                                                + out_node_bias) * 2.0 * (out_node_val-out_node_expected_val);
                            layer_biases_gradient.push(out_node_multiplier);
                            layer_weights_gradient.push(previous_layer_vals.iter().map(|in_node_activation| in_node_activation*out_node_multiplier).collect::<Vec<f64>>());
                            temp_layer_node_multpliers.push(out_node_multiplier);
                        }
                    } else {
                        for (out_node_back_weights,(out_node_bias,out_node_multiplier)) in 
                        flip_matrix(layer_weights).iter().zip(layer_biases.iter().zip(layer_node_multipliers.iter())) {
                            let out_node_multiplier = self.derivative_activation_function(previous_layer_vals.iter()
                                                                                                .zip(out_node_back_weights.iter())
                                                                                                .map(|(previous_activation,weight)| previous_activation*weight)
                                                                                                .sum::<f64>() 
                                                                                                + out_node_bias) * out_node_multiplier;
                            layer_biases_gradient.push(out_node_multiplier);
                            layer_weights_gradient.push(previous_layer_vals.iter().map(|in_node_activation| in_node_activation*out_node_multiplier).collect::<Vec<f64>>());
                            temp_layer_node_multpliers.push(out_node_multiplier);
                        }
                    }
                    biases_gradient.push(layer_biases_gradient);
                    weights_gradient.push(layer_weights_gradient);
                    let mut new_layer_node_multipliers = Vec::new();
                    for in_node in layer_weights {
                        new_layer_node_multipliers.push(in_node.iter().zip(temp_layer_node_multpliers.iter()).map(|(weight,multiplier)| weight*multiplier).sum::<f64>());
                    }
                    layer_node_multipliers = new_layer_node_multipliers;
                }
                */
                let (biases_gradient,weights_gradient) = self.one_example_back_propagation(training_ex_in, training_ex_out);
                for (column_bias_gradient,final_column_bias_gradient) in biases_gradient.into_iter().rev().zip(final_biases_gradient.iter_mut()) {
                    for (bias_gradient,final_bias_gradient) in column_bias_gradient.into_iter().zip(final_column_bias_gradient.iter_mut()) {
                        *final_bias_gradient += bias_gradient;
                    }
                }
                for (layer_weights_gradient,final_layer_weights_gradient) in weights_gradient.into_iter().rev().zip(final_weights_gradient.iter_mut()) {
                    for (in_node_weights_gradient, final_in_node_weights_gradient) in layer_weights_gradient.into_iter().zip(final_layer_weights_gradient.iter_mut()) {
                        for (weight_gradient,final_weight_gradient) in in_node_weights_gradient.into_iter().zip(final_in_node_weights_gradient.iter_mut()) {
                            *final_weight_gradient += weight_gradient;
                        }
                    }
                }
            }
            let num_examples = training_data.len() as f64;
            for column in final_biases_gradient.iter_mut() {
                for bias in column.iter_mut() {
                    *bias /= num_examples;
                }
            }
            for layer in final_weights_gradient.iter_mut() {
                for in_node in layer.iter_mut() {
                    for weight in in_node.iter_mut() {
                        *weight /= num_examples;
                    }
                }
            }

            (final_biases_gradient,final_weights_gradient)
        }

        fn core_stochastic_back_propagation(&self,training_data: &Vec<(Vec<f64>,Vec<f64>)>,num_considered: usize) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            let length = self.node_layout.len();
            let mut final_biases_gradient = Vec::with_capacity(length-1);
            for layer_size in self.node_layout[1..length].iter() {
                let mut layer_biases = Vec::with_capacity(*layer_size);
                for _ in 0..*layer_size {
                    layer_biases.push(0.0); 
                }
                final_biases_gradient.push(layer_biases);
            }
            let mut final_weights_gradient = Vec::with_capacity(length-1);
            for (previous_layer_size, layer_size) in self.node_layout[0..length-1].iter().zip(self.node_layout[1..length].iter()) {
                let mut layer_weights = Vec::with_capacity(*previous_layer_size);
                for _ in 0..*previous_layer_size {
                    let mut node_weights = Vec::with_capacity(*layer_size);
                    for _ in 0..*layer_size {
                        node_weights.push(0.0);
                    }
                    layer_weights.push(node_weights);
                }
                final_weights_gradient.push(layer_weights);
            }
            for (training_ex_in,training_ex_out) in training_data.choose_multiple(&mut rand::thread_rng(),num_considered) {
                /*
                let mut biases_gradient = Vec::new();
                let mut weights_gradient = Vec::new();
                let layer_vals = self.special_run(training_ex_in);
                let mut layer_node_multipliers = Vec::new(); //there may be a specific func for this
                for _ in 0..layer_vals[layer_vals.len()-1].len() {
                    layer_node_multipliers.push(1.0);
                }
                let temp_layer_vals: Vec<&Vec<f64>> = layer_vals[0..layer_vals.len()].iter().rev().collect(); //JANK, originally to avoid lazyness in python
                let mut temp_layer_node_multpliers: Vec<f64>;
                for (current_layer_vals,(layer_weights,(layer_biases,(layer_num,previous_layer_vals)))) in 
                layer_vals.iter().rev().zip(self.weights.iter().rev().zip(self.biases.iter().rev().zip((0..layer_vals.len()).into_iter().zip(temp_layer_vals.iter())))) {
                    let mut layer_biases_gradient = Vec::new();
                    let mut layer_weights_gradient = Vec::new();
                    temp_layer_node_multpliers = Vec::new();
                    if layer_num == 0 {
                        for (out_node_val,(out_node_back_weights,(out_node_bias,out_node_expected_val))) in
                        current_layer_vals.iter().zip(flip_matrix(layer_weights).iter().zip(layer_biases.iter().zip(training_ex_out.iter()))) {
                            let out_node_multiplier = self.derivative_activation_function(previous_layer_vals.iter()
                                                                                                .zip(out_node_back_weights.iter())
                                                                                                .map(|(previous_activation,weight)| previous_activation*weight)
                                                                                                .sum::<f64>() 
                                                                                                + out_node_bias) * 2.0 * (out_node_val-out_node_expected_val);
                            layer_biases_gradient.push(out_node_multiplier);
                            layer_weights_gradient.push(previous_layer_vals.iter().map(|in_node_activation| in_node_activation*out_node_multiplier).collect::<Vec<f64>>());
                            temp_layer_node_multpliers.push(out_node_multiplier);
                        }
                    } else {
                        for (out_node_back_weights,(out_node_bias,out_node_multiplier)) in 
                        flip_matrix(layer_weights).iter().zip(layer_biases.iter().zip(layer_node_multipliers.iter())) {
                            let out_node_multiplier = self.derivative_activation_function(previous_layer_vals.iter()
                                                                                                .zip(out_node_back_weights.iter())
                                                                                                .map(|(previous_activation,weight)| previous_activation*weight)
                                                                                                .sum::<f64>() 
                                                                                                + out_node_bias) * out_node_multiplier;
                            layer_biases_gradient.push(out_node_multiplier);
                            layer_weights_gradient.push(previous_layer_vals.iter().map(|in_node_activation| in_node_activation*out_node_multiplier).collect::<Vec<f64>>());
                            temp_layer_node_multpliers.push(out_node_multiplier);
                        }
                    }
                    biases_gradient.push(layer_biases_gradient);
                    weights_gradient.push(layer_weights_gradient);
                    let mut new_layer_node_multipliers = Vec::new();
                    for in_node in layer_weights {
                        new_layer_node_multipliers.push(in_node.iter().zip(temp_layer_node_multpliers.iter()).map(|(weight,multiplier)| weight*multiplier).sum::<f64>());
                    }
                    layer_node_multipliers = new_layer_node_multipliers;
                }
                */
                let (biases_gradient,weights_gradient) = self.one_example_back_propagation(training_ex_in, training_ex_out);
                for (column_bias_gradient,final_column_bias_gradient) in biases_gradient.into_iter().rev().zip(final_biases_gradient.iter_mut()) {
                    for (bias_gradient,final_bias_gradient) in column_bias_gradient.into_iter().zip(final_column_bias_gradient.iter_mut()) {
                        *final_bias_gradient += bias_gradient;
                    }
                }
                for (layer_weights_gradient,final_layer_weights_gradient) in weights_gradient.into_iter().rev().zip(final_weights_gradient.iter_mut()) {
                    for (in_node_weights_gradient, final_in_node_weights_gradient) in layer_weights_gradient.into_iter().zip(final_layer_weights_gradient.iter_mut()) {
                        for (weight_gradient,final_weight_gradient) in in_node_weights_gradient.into_iter().zip(final_in_node_weights_gradient.iter_mut()) {
                            *final_weight_gradient += weight_gradient;
                        }
                    }
                }
            }
            let num_examples = num_considered as f64;
            for column in final_biases_gradient.iter_mut() {
                for bias in column.iter_mut() {
                    *bias /= num_examples;
                }
            }
            for layer in final_weights_gradient.iter_mut() {
                for in_node in layer.iter_mut() {
                    for weight in in_node.iter_mut() {
                        *weight /= num_examples;
                    }
                }
            }

            (final_biases_gradient,final_weights_gradient)
        }
        
        fn core_multithreaded_back_propagation(&self,partitioned_training_data: &Vec<Vec<&(Vec<f64>,Vec<f64>)>>) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            let length = self.node_layout.len();
            let mut final_biases_gradient = Vec::with_capacity(length-1);
            for layer_size in self.node_layout[1..length].iter() {
                let mut layer_biases = Vec::with_capacity(*layer_size);
                for _ in 0..*layer_size {
                    layer_biases.push(0.0); 
                }
                final_biases_gradient.push(layer_biases);
            }
            let mut final_weights_gradient = Vec::with_capacity(length-1);
            for (previous_layer_size, layer_size) in self.node_layout[0..length-1].iter().zip(self.node_layout[1..length].iter()) {
                let mut layer_weights = Vec::with_capacity(*previous_layer_size);
                for _ in 0..*previous_layer_size {
                    let mut node_weights = Vec::with_capacity(*layer_size);
                    for _ in 0..*layer_size {
                        node_weights.push(0.0);
                    }
                    layer_weights.push(node_weights);
                }
                final_weights_gradient.push(layer_weights);
            }

            //multithreading specific stuff
            crossbeam::scope(|scope| {
                let (original_transmitter,reciever) = mpsc::channel();
                for partition in partitioned_training_data {
                    let transmitter = original_transmitter.clone();
                    scope.spawn(move |_| {
                        for (training_ex_in,training_ex_out) in partition {
                            /*
                            let mut biases_gradient = Vec::new();
                            let mut weights_gradient = Vec::new();
                            let layer_vals = self.special_run(training_ex_in);
                            let mut layer_node_multipliers = Vec::new(); //there may be a specific func for this
                            for _ in 0..layer_vals[layer_vals.len()-1].len() {
                                layer_node_multipliers.push(1.0);
                            }
                            let temp_layer_vals: Vec<&Vec<f64>> = layer_vals[0..layer_vals.len()].iter().rev().collect(); //JANK, originally to avoid lazyness in python
                            let mut temp_layer_node_multpliers: Vec<f64>;
                            for (current_layer_vals,(layer_weights,(layer_biases,(layer_num,previous_layer_vals)))) in 
                            layer_vals.iter().rev().zip(self.weights.iter().rev().zip(self.biases.iter().rev().zip((0..layer_vals.len()).into_iter().zip(temp_layer_vals.iter())))) {
                                let mut layer_biases_gradient = Vec::new();
                                let mut layer_weights_gradient = Vec::new();
                                temp_layer_node_multpliers = Vec::new();
                                if layer_num == 0 {
                                    for (out_node_val,(out_node_back_weights,(out_node_bias,out_node_expected_val))) in
                                    current_layer_vals.iter().zip(flip_matrix(layer_weights).iter().zip(layer_biases.iter().zip(training_ex_out.iter()))) {
                                        let out_node_multiplier = self.derivative_activation_function(previous_layer_vals.iter()
                                                                                                            .zip(out_node_back_weights.iter())
                                                                                                            .map(|(previous_activation,weight)| previous_activation*weight)
                                                                                                            .sum::<f64>() 
                                                                                                            + out_node_bias) * 2.0 * (out_node_val-out_node_expected_val);
                                        layer_biases_gradient.push(out_node_multiplier);
                                        layer_weights_gradient.push(previous_layer_vals.iter().map(|in_node_activation| in_node_activation*out_node_multiplier).collect::<Vec<f64>>());
                                        temp_layer_node_multpliers.push(out_node_multiplier);
                                    }
                                } else {
                                    for (out_node_back_weights,(out_node_bias,out_node_multiplier)) in 
                                    flip_matrix(layer_weights).iter().zip(layer_biases.iter().zip(layer_node_multipliers.iter())) {
                                        let out_node_multiplier = self.derivative_activation_function(previous_layer_vals.iter()
                                                                                                            .zip(out_node_back_weights.iter())
                                                                                                            .map(|(previous_activation,weight)| previous_activation*weight)
                                                                                                            .sum::<f64>() 
                                                                                                            + out_node_bias) * out_node_multiplier;
                                        layer_biases_gradient.push(out_node_multiplier);
                                        layer_weights_gradient.push(previous_layer_vals.iter().map(|in_node_activation| in_node_activation*out_node_multiplier).collect::<Vec<f64>>());
                                        temp_layer_node_multpliers.push(out_node_multiplier);
                                    }
                                }
                                biases_gradient.push(layer_biases_gradient);
                                weights_gradient.push(layer_weights_gradient);
                                let mut new_layer_node_multipliers = Vec::new();
                                for in_node in layer_weights {
                                    new_layer_node_multipliers.push(in_node.iter().zip(temp_layer_node_multpliers.iter()).map(|(weight,multiplier)| weight*multiplier).sum::<f64>());
                                }
                                layer_node_multipliers = new_layer_node_multipliers;
                            }
                            */
                            transmitter.send(self.one_example_back_propagation(training_ex_in, training_ex_out)).unwrap();//(biases_gradient,weights_gradient)).unwrap();
                        }
                    });
                }
                drop(original_transmitter);

                for (biases_gradient,weights_gradient) in reciever {
                    for (column_bias_gradient,final_column_bias_gradient) in biases_gradient.into_iter().rev().zip(final_biases_gradient.iter_mut()) {
                        for (bias_gradient,final_bias_gradient) in column_bias_gradient.into_iter().zip(final_column_bias_gradient.iter_mut()) {
                            *final_bias_gradient += bias_gradient;
                        }
                    }
                    for (layer_weights_gradient,final_layer_weights_gradient) in weights_gradient.into_iter().rev().zip(final_weights_gradient.iter_mut()) {
                        for (in_node_weights_gradient, final_in_node_weights_gradient) in layer_weights_gradient.into_iter().zip(final_layer_weights_gradient.iter_mut()) {
                            for (weight_gradient,final_weight_gradient) in in_node_weights_gradient.into_iter().zip(final_in_node_weights_gradient.iter_mut()) {
                                *final_weight_gradient += weight_gradient;
                            }
                        }
                    }
                }
            }).unwrap();
            let mut num_examples = 0;
            for partition in partitioned_training_data {
                num_examples += partition.len();
            }
            let num_examples = num_examples as f64;
            for column in final_biases_gradient.iter_mut() {
                for bias in column.iter_mut() {
                    *bias /= num_examples;
                }
            }
            for layer in final_weights_gradient.iter_mut() {
                for in_node in layer.iter_mut() {
                    for weight in in_node.iter_mut() {
                        *weight /= num_examples;
                    }
                }
            }

            (final_biases_gradient,final_weights_gradient)
        }

        //these fns can and should have better granularity adjustment (or as its actually called: learning rate adjustment)
        //right now it is basic: keep it where it is until the result is worse, if so half it
        pub fn back_propagation(&mut self, training_data: &Vec<(Vec<f64>,Vec<f64>)>, test_data: &Vec<(Vec<f64>,Vec<f64>)>, min_granularity: f64, is_silent: bool) {
            let mut current_granularity = 1.0;
            //double testing done here, would be nice to avoid it
            let mut previous_test = self.test(test_data);
            if !is_silent {
                println!("Test: {}",previous_test);
            }
            //not using previous_biases and previous_weights here to avoid cloning, needs testing to see if better
            //theoretically has worse precision
            'main_loop: loop {
                let (biases_gradient,weights_gradient) = self.core_back_propagation(training_data);
                let mut multiplier = current_granularity/find_greatest_movement_in_gradients(&biases_gradient, &weights_gradient);
                self.apply_gradient(&biases_gradient, &weights_gradient, multiplier);
                let mut new_test = self.test(test_data);
                if !is_silent {
                    println!("Test: {}",new_test);
                }
                while new_test > previous_test {
                    current_granularity /= 2.0;
                    if !is_silent {
                        println!("Granularity: {}",current_granularity);
                        println!("Min Granularity: {}",min_granularity);
                    }
                    if current_granularity < min_granularity {
                        self.apply_gradient(&biases_gradient, &weights_gradient, multiplier*-1.0);
                        break 'main_loop;
                    }
                    multiplier /= 2.0;
                    self.apply_gradient(&biases_gradient, &weights_gradient, multiplier*-1.0);
                    new_test = self.test(test_data);
                    if !is_silent {
                        println!("Test: {}",new_test);
                    }
                }
                previous_test = new_test;
            }
        }

        pub fn stochastic_back_propagation(&mut self, training_data: &Vec<(Vec<f64>,Vec<f64>)>, test_data: &Vec<(Vec<f64>,Vec<f64>)>, min_granularity: f64, num_considered_in_iteration: usize, is_silent: bool) {
            let mut current_granularity = 1.0;
            //double testing done here, would be nice to avoid it
            let mut previous_test = self.test(test_data);
            if !is_silent {
                println!("Test: {}",previous_test);
            }
            //not using previous_biases and previous_weights here to avoid cloning, needs testing to see if better
            //theoretically has worse precision
            'main_loop: loop {
                let (biases_gradient,weights_gradient) = self.core_stochastic_back_propagation(training_data,num_considered_in_iteration);
                let mut multiplier = current_granularity/find_greatest_movement_in_gradients(&biases_gradient, &weights_gradient);
                self.apply_gradient(&biases_gradient, &weights_gradient, multiplier);
                let mut new_test = self.test(test_data);
                if !is_silent {
                    println!("Test: {}",new_test);
                }
                while new_test > previous_test {
                    current_granularity /= 2.0;
                    if !is_silent {
                        println!("Granularity: {}",current_granularity);
                        println!("Min granularity: {}",min_granularity);
                    }
                    if current_granularity < min_granularity {
                        self.apply_gradient(&biases_gradient, &weights_gradient, multiplier*-1.0);
                        break 'main_loop;
                    }
                    multiplier /= 2.0;
                    self.apply_gradient(&biases_gradient, &weights_gradient, multiplier*-1.0);
                    new_test = self.test(test_data);
                    if !is_silent {
                        println!("Test: {}",new_test);
                    }
                }
                previous_test = new_test;
            }
        }
    
        pub fn multithreaded_back_propagation(&mut self, training_data: &Vec<(Vec<f64>,Vec<f64>)>, test_data: &Vec<(Vec<f64>,Vec<f64>)>, min_granularity: f64, num_data_partitions: f64, is_silent: bool) {
            let mut current_granularity = 1.0;
            //partitioning the data
            let partitioned_training_data = partition_data(training_data, num_data_partitions);
            let partitioned_test_data = partition_data(test_data, num_data_partitions);
            //double running done here, one pass in back prop, the other in test, would be nice to avoid it
            let mut previous_test = self.prepartitioned_multithreaded_test(&partitioned_test_data);
            if !is_silent {
                println!("Test: {}",previous_test);
            }
            'main_loop: loop {
                let (biases_gradient,weights_gradient) = self.core_multithreaded_back_propagation(&partitioned_training_data);
                let mut multiplier = current_granularity/find_greatest_movement_in_gradients(&biases_gradient, &weights_gradient);
                self.apply_gradient(&biases_gradient, &weights_gradient, multiplier);
                let mut new_test = self.prepartitioned_multithreaded_test(&partitioned_test_data);
                if !is_silent {
                    println!("Test: {}",new_test);
                }
                while new_test > previous_test {
                    current_granularity /= 2.0;
                    if !is_silent {
                        println!("Granularity: {}",current_granularity);
                        println!("Min Granularity: {}",min_granularity);
                    }
                    if current_granularity < min_granularity {
                        self.apply_gradient(&biases_gradient, &weights_gradient, multiplier*-1.0);
                        break 'main_loop;
                    }
                    multiplier /= 2.0;
                    self.apply_gradient(&biases_gradient,&weights_gradient,multiplier*-1.0);
                    new_test = self.prepartitioned_multithreaded_test(&partitioned_test_data);
                    if !is_silent {
                        println!("Test: {}",new_test);
                    }
                }
                previous_test = new_test;
            }
        }
  
        pub fn backtracking_line_search_train(&mut self, training_data: &Vec<(Vec<f64>,Vec<f64>)>, first_checked_rate: f64, rate_degradation_factor: f64, tolerance_parameter: f64, minimum_learning_rate: f64) {
            let partitioned_training_data = partition_data(training_data, 4.0);
            let mut previous_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
            println!("{}",previous_test);
            let mut current_learning_rate = first_checked_rate;
            'main_loop: loop {
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
                    self.apply_gradient(&biases_gradient, &weights_gradient, 1.0-current_learning_rate);
                    new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                    println!("{}",new_test);
                }
                previous_test = new_test;
            }
        }
    }
}

pub mod aiz_unstable {
    //An unstable branch of the main aiz module

    use rand::Rng; //To inititalize networks randomly
    //use rand::seq::SliceRandom; //to sample random groups of examples from the data for stochastic training
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

        fn activation_fn(&self, x: f64) -> f64 {
            1.0/(1.0+(-x).exp())
        }

        //see if could be &f64 input
        fn derivative_activation_fn(&self, x: f64) -> f64 {
            let intermedite_num = self.activation_fn(x);
            intermedite_num*(1.0-intermedite_num)
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

        //TO DO: swap Vec::new's with Vec::with_capacity's
        fn one_example_back_propagation(&self, training_in: &Vec<f64>, training_out: &Vec<f64>) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            let mut biases_gradient = Vec::new();
            let mut weights_gradient = Vec::new();
            
            let (network_activations,network_pre_activations) = self.special_run(training_in);

            //last layer specific calculations
            let mut layer_node_derivatives = Vec::new();
            let mut last_layer_biases_gradient = Vec::new();
            let mut last_layer_weights_gradient = Vec::new();
            for (pre_activation,(expected_val,real_val)) in 
            network_pre_activations[network_pre_activations.len()-1].iter().zip(
            training_out.iter().zip(
            network_activations[network_activations.len()-1].iter())) {
                let current_node_derivative = 2.0*(real_val-expected_val)*self.derivative_activation_fn(*pre_activation);
                last_layer_biases_gradient.push(current_node_derivative);
                let mut node_weights_gradient = Vec::new();
                for previous_node_activation in network_activations[network_activations.len()-2].iter() {
                    node_weights_gradient.push(previous_node_activation*current_node_derivative);
                }
                last_layer_weights_gradient.push(node_weights_gradient);
                layer_node_derivatives.push(current_node_derivative);
            }
            biases_gradient.push(last_layer_biases_gradient);
            weights_gradient.push(last_layer_weights_gradient);
            //All other layer calculations
            for (forward_layer_weights,(current_pre_activations,previous_activations)) in 
            self.weights.iter().rev().zip(
            network_pre_activations[0..network_pre_activations.len()-1].iter().rev().zip(
            network_activations[0..network_activations.len()-2].iter().rev())) {
                let mut new_layer_node_derivatives = Vec::new();
                let mut layer_biases_gradient = Vec::new();
                let mut layer_weights_gradient = Vec::new();
                for (input_node_weights,node_pre_activation) in transpose_matrix(forward_layer_weights).iter().zip(current_pre_activations.iter()) {
                    let mut new_node_derivative = 0.0;
                    for (weight,forward_derivative) in input_node_weights.iter().zip(layer_node_derivatives.iter()) {
                        new_node_derivative += *weight * forward_derivative;
                    }
                    new_node_derivative *= self.derivative_activation_fn(*node_pre_activation);
                    layer_biases_gradient.push(new_node_derivative);
                    let mut node_weights_gradients = Vec::new();
                    for previous_node_activation in previous_activations {
                        node_weights_gradients.push(previous_node_activation*new_node_derivative)
                    }
                    layer_weights_gradient.push(node_weights_gradients);
                    new_layer_node_derivatives.push(new_node_derivative);
                }
                biases_gradient.push(layer_biases_gradient);
                weights_gradient.push(layer_weights_gradient);
                layer_node_derivatives = new_layer_node_derivatives;
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

        pub fn backtracking_line_search_train(&mut self, training_data: &Vec<(Vec<f64>,Vec<f64>)>, first_checked_rate: f64, rate_degradation_factor: f64, tolerance_parameter: f64, minimum_learning_rate: f64, max_iterations: u32) {
            let partitioned_training_data = partition_data(training_data, 4.0);
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
                    self.apply_gradient(&biases_gradient, &weights_gradient, 1.0-current_learning_rate);
                    new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                    println!("{}",new_test);
                }
                previous_test = new_test;

                iteration_num += 1
            }
        }
    
    }
}