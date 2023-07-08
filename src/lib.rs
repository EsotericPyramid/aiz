
pub mod aiz {
    use rand::Rng;
    use rand::seq::SliceRandom;
    use crossbeam;
    use std::sync::mpsc;

    //should be optimizable
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

        pub fn test(&self,test_data: &Vec<(Vec<f64>,Vec<f64>)>) -> f64{
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

        fn apply_gradient(&mut self,biases_gradient: Vec<Vec<f64>>,weights_gradient: Vec<Vec<Vec<f64>>>,multiplier: f64) {
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

        fn core_back_propagation(&self,training_data: &Vec<(Vec<f64>,Vec<f64>)>) -> (Vec<Vec<f64>>,Vec<Vec<Vec<f64>>>) {
            //a lot of with_capacity optimizations to be made here
            /*
            let mut biases_gradients = Vec::new();
            let mut weights_gradients = Vec::new();
            */
            //testing out a running sum
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
                /*
                biases_gradients.push(biases_gradient);
                weights_gradients.push(weights_gradient);
                */
                //for the running sum
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
            //It would be a good idea to avoid doing this as ot requires a ton of memory, a better option would probably be to have a running sum and then divide em later
            /*
            let mut final_biases_gradient = Vec::new();
            for column in flip_owned_matrix(biases_gradients) {
                let mut final_biases_column_gradient = Vec::new();
                for bias in flip_owned_matrix(column) {
                    final_biases_column_gradient.push(bias.iter().sum::<f64>()/bias.len() as f64);
                }
                final_biases_gradient.push(final_biases_column_gradient);
            }    
            let mut final_weights_gradient = Vec::new();
            for layer in flip_owned_matrix(weights_gradients) {
                let mut final_weights_layer_gradient = Vec::new();
                for column in flip_owned_matrix(layer) {
                    let mut final_weights_column_gradient = Vec::new();
                    for weight in flip_owned_matrix(column) {
                        final_weights_column_gradient.push(weight.iter().sum::<f64>()/weight.len() as f64);
                    }
                    final_weights_layer_gradient.push(final_weights_column_gradient);
                }
                final_weights_gradient.push(final_weights_layer_gradient);
            }
            (final_biases_gradient,final_weights_gradient)
            */
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
            //a lot of with_capacity optimizations to be made here
            /* 
            let mut biases_gradients = Vec::new();
            let mut weights_gradients = Vec::new();
            */
            //testing out a running sum
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
                /*
                biases_gradients.push(biases_gradient);
                weights_gradients.push(weights_gradient);
                */
                //For the running sum
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
            //It would be a good idea to avoid doing this as ot requires a ton of memory, a better option would probably be to have a running sum and then divide em later
            /*
            let mut final_biases_gradient = Vec::new();
            for column in flip_owned_matrix(biases_gradients) {
                let mut final_biases_column_gradient = Vec::new();
                for bias in flip_owned_matrix(column) {
                    final_biases_column_gradient.push(bias.iter().sum::<f64>()/bias.len() as f64);
                }
                final_biases_gradient.push(final_biases_column_gradient);
            }    
            let mut final_weights_gradient = Vec::new();
            for layer in flip_owned_matrix(weights_gradients) {
                let mut final_weights_layer_gradient = Vec::new();
                for column in flip_owned_matrix(layer) {
                    let mut final_weights_column_gradient = Vec::new();
                    for weight in flip_owned_matrix(column) {
                        final_weights_column_gradient.push(weight.iter().sum::<f64>()/weight.len() as f64);
                    }
                    final_weights_layer_gradient.push(final_weights_column_gradient);
                }
                final_weights_gradient.push(final_weights_layer_gradient);
            }
            (final_biases_gradient,final_weights_gradient)
            */
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
        
        //works but not as well as I would like
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
                            transmitter.send((biases_gradient,weights_gradient)).unwrap();
                        }
                    });
                }
                //to drop the original transmitter, although pretty jank
                {
                    original_transmitter
                };

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
                let mut greatest_movement = 0.0;
                for column in biases_gradient.iter() {
                    for bias in column {
                        if &greatest_movement < bias {
                            greatest_movement = *bias;
                        }
                    }
                }
                for layer in weights_gradient.iter() {
                    for in_node in layer {
                        for weight in in_node {
                            if &greatest_movement < weight {
                                greatest_movement = *weight;
                            }
                        }
                    }
                }
                let mut multiplier = current_granularity/greatest_movement;
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
                    multiplier /= 2.0 ;
                    if current_granularity < min_granularity {
                        for (column_biases, column_gradient) in self.biases.iter_mut().zip(biases_gradient.iter()) {
                            for (bias, movement) in column_biases.iter_mut().zip(column_gradient.iter()) {
                                *bias += movement*multiplier*2.0;
                            }
                        }
                        for (layer_weights, layer_gradient) in self.weights.iter_mut().zip(weights_gradient.iter()) {
                            for (in_node_weights,in_node_gradient) in layer_weights.iter_mut().zip(layer_gradient.iter()) {
                                for (weight, movement) in in_node_weights.iter_mut().zip(in_node_gradient.iter()) {
                                    *weight += movement*multiplier*2.0;
                                }
                            }
                        }
                        break 'main_loop;
                    }
                    for (column_biases, column_gradient) in self.biases.iter_mut().zip(biases_gradient.iter()) {
                        for (bias, movement) in column_biases.iter_mut().zip(column_gradient.iter()) {
                            *bias += movement*multiplier;
                        }
                    }
                    for (layer_weights, layer_gradient) in self.weights.iter_mut().zip(weights_gradient.iter()) {
                        for (in_node_weights,in_node_gradient) in layer_weights.iter_mut().zip(layer_gradient.iter()) {
                            for (weight, movement) in in_node_weights.iter_mut().zip(in_node_gradient.iter()) {
                                *weight += movement*multiplier;
                            }
                        }
                    }
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
                let mut greatest_movement = 0.0;
                for column in biases_gradient.iter() {
                    for bias in column {
                        if &greatest_movement < bias {
                            greatest_movement = *bias;
                        }
                    }
                }
                for layer in weights_gradient.iter() {
                    for in_node in layer {
                        for weight in in_node {
                            if &greatest_movement < weight {
                                greatest_movement = *weight;
                            }
                        }
                    }
                }
                let mut multiplier = current_granularity/greatest_movement;
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
                    multiplier /= 2.0 ;
                    if current_granularity < min_granularity {
                        for (column_biases, column_gradient) in self.biases.iter_mut().zip(biases_gradient.iter()) {
                            for (bias, movement) in column_biases.iter_mut().zip(column_gradient.iter()) {
                                *bias += movement*multiplier*2.0;
                            }
                        }
                        for (layer_weights, layer_gradient) in self.weights.iter_mut().zip(weights_gradient.iter()) {
                            for (in_node_weights,in_node_gradient) in layer_weights.iter_mut().zip(layer_gradient.iter()) {
                                for (weight, movement) in in_node_weights.iter_mut().zip(in_node_gradient.iter()) {
                                    *weight += movement*multiplier*2.0;
                                }
                            }
                        }
                        break 'main_loop;
                    }
                    for (column_biases, column_gradient) in self.biases.iter_mut().zip(biases_gradient.iter()) {
                        for (bias, movement) in column_biases.iter_mut().zip(column_gradient.iter()) {
                            *bias += movement*multiplier;
                        }
                    }
                    for (layer_weights, layer_gradient) in self.weights.iter_mut().zip(weights_gradient.iter()) {
                        for (in_node_weights,in_node_gradient) in layer_weights.iter_mut().zip(layer_gradient.iter()) {
                            for (weight, movement) in in_node_weights.iter_mut().zip(in_node_gradient.iter()) {
                                *weight += movement*multiplier;
                            }
                        }
                    }
                    new_test = self.test(test_data);
                    if !is_silent {
                        println!("Test: {}",new_test);
                    }
                }
                previous_test = new_test;
            }
        }
    
        pub fn multithreaded_back_propagation(&mut self, training_data: &Vec<(Vec<f64>,Vec<f64>)>, test_data: &Vec<(Vec<f64>,Vec<f64>)>, min_granularity: f64, num_training_data_partitions: f64, is_silent: bool) {
            let mut current_granularity = 1.0;
            //double testing done here, would be nice to avoid it
            let mut previous_test = self.test(test_data);
            if !is_silent {
                println!("Test: {}",previous_test);
            }
            //partitioning the data
            let data_length = training_data.len();
            let exact_partition_size = data_length as f64 / num_training_data_partitions;
            let mut partitioned_training_data = Vec::new();
            let mut current_partition = Vec::new();
            let mut current_partition_point = exact_partition_size;
            for (example_num,example) in training_data.iter().enumerate() {
                if (example_num as f64) < current_partition_point {
                    current_partition.push(example);
                } else {
                    current_partition_point += exact_partition_size;
                    partitioned_training_data.push(current_partition);
                    current_partition = Vec::new();
                    current_partition.push(example);
                }
            }
            partitioned_training_data.push(current_partition);
            //not using previous_biases and previous_weights here to avoid cloning, needs testing to see if better
            //theoretically has worse precision
            'main_loop: loop {
                let (biases_gradient,weights_gradient) = self.core_multithreaded_back_propagation(&partitioned_training_data);
                let mut greatest_movement = 0.0;
                for column in biases_gradient.iter() {
                    for bias in column {
                        if &greatest_movement < bias {
                            greatest_movement = *bias;
                        }
                    }
                }
                for layer in weights_gradient.iter() {
                    for in_node in layer {
                        for weight in in_node {
                            if &greatest_movement < weight {
                                greatest_movement = *weight;
                            }
                        }
                    }
                }
                let mut multiplier = current_granularity/greatest_movement;
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
                    multiplier /= 2.0 ;
                    if current_granularity < min_granularity {
                        for (column_biases, column_gradient) in self.biases.iter_mut().zip(biases_gradient.iter()) {
                            for (bias, movement) in column_biases.iter_mut().zip(column_gradient.iter()) {
                                *bias += movement*multiplier*2.0;
                            }
                        }
                        for (layer_weights, layer_gradient) in self.weights.iter_mut().zip(weights_gradient.iter()) {
                            for (in_node_weights,in_node_gradient) in layer_weights.iter_mut().zip(layer_gradient.iter()) {
                                for (weight, movement) in in_node_weights.iter_mut().zip(in_node_gradient.iter()) {
                                    *weight += movement*multiplier*2.0;
                                }
                            }
                        }
                        break 'main_loop;
                    }
                    for (column_biases, column_gradient) in self.biases.iter_mut().zip(biases_gradient.iter()) {
                        for (bias, movement) in column_biases.iter_mut().zip(column_gradient.iter()) {
                            *bias += movement*multiplier;
                        }
                    }
                    for (layer_weights, layer_gradient) in self.weights.iter_mut().zip(weights_gradient.iter()) {
                        for (in_node_weights,in_node_gradient) in layer_weights.iter_mut().zip(layer_gradient.iter()) {
                            for (weight, movement) in in_node_weights.iter_mut().zip(in_node_gradient.iter()) {
                                *weight += movement*multiplier;
                            }
                        }
                    }
                    new_test = self.test(test_data);
                    if !is_silent {
                        println!("Test: {}",new_test);
                    }
                }
                previous_test = new_test;
            }
        }
    }
}