#![allow(clippy::too_many_arguments)] //its a given with this

use rand::Rng; 
//RNG, figure it out yourself
use std::sync::mpsc; //communicating between threads
use std::collections::VecDeque; //for l_bfgs memory buffer
use std::thread; //multithreading

//assumes all the inner vectors are the same length
pub fn transpose_matrix<T>(matrix: &Vec<Vec<T>>) -> TransposedMatrix<'_,T> {
    TransposedMatrix { matrix, row_index: 0 , matrix_dimension: matrix[0].len()}
}

pub struct TransposedMatrix<'a,T>{
    matrix: &'a Vec<Vec<T>>,
    row_index: usize,
    matrix_dimension: usize,
}

impl<'a,T> Iterator for TransposedMatrix<'a,T> {
    type Item = Vec<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_index == self.matrix_dimension {
            None
        } else {
            let mut output = Vec::with_capacity(self.matrix.len());
            for vector in self.matrix {
                output.push(&vector[self.row_index])
            }
            self.row_index += 1;
            Some(output)
        }
    }
}

//no randomness is implemented in this partitioning
//this is because it is only intended to split up work
//for multithreaded training and testing, so no randomness needed
//Also: num_partitions is f64 for optimization reasons, 
//in theory you could use that fact on purpose for cores with dif speeds
//but it isnt for that
pub fn partition_data<T>(data: &[T],num_partitions: f64) -> Vec<Vec<&T>> {
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
            current_partition = Vec::with_capacity(exact_partition_size.ceil() as usize);
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
    fn add(&mut self,other: Self) where Self: Sized;
    fn mult(&mut self,factor: f64) where Self: Sized;
    fn div(&mut self,divisor: f64) where Self: Sized;
    fn dot_product(&self,other: &Self) -> f64 where Self: Sized;
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
    fn test(&self,test_data: &[(Vec<f64>,Vec<f64>)]) -> f64;
    fn prepartitioned_multithreaded_test(&self,partitioned_test_data: &[Vec<&(Vec<f64>,Vec<f64>)>]) -> f64;
    fn multithreaded_test(&self,test_data: &[(Vec<f64>,Vec<f64>)],num_partitions: f64) -> f64;
    fn back_propagation(&self,training_data: &[(Vec<f64>,Vec<f64>)]) -> Self::Gradient;
    fn multithreaded_back_propagation(&self,partitioned_training_data: &[Vec<&(Vec<f64>,Vec<f64>)>]) -> Self::Gradient;
    fn backtracking_line_search_train(
        &mut self, 
        training_data: &[(Vec<f64>,Vec<f64>)], 
        first_checked_rate: f64, 
        rate_degradation_factor: f64, 
        tolerance_parameter: f64, 
        minimum_learning_rate: f64, 
        max_iterations: u32
    );
    fn multithreaded_backtracking_line_search_train(
        &mut self, 
        training_data: &[(Vec<f64>,Vec<f64>)], 
        num_partitions: f64,
        first_checked_rate: f64, 
        rate_degradation_factor: f64, 
        tolerance_parameter: f64, 
        minimum_learning_rate: f64, 
        max_iterations: u32
    );
}

impl<T: Network> NetworkInherentMethods for T {
    fn test(&self,test_data: &[(Vec<f64>,Vec<f64>)]) -> f64 {
        let mut output = 0.0;
        for (test_in,test_expected_out) in test_data {
            for (single_real_out,single_expected_out) in self.run(test_in).iter().zip(test_expected_out.iter()) {
                let dif = single_real_out - single_expected_out;
                output += dif*dif;
            }
        }
        output / test_data.len() as f64
    }

    fn prepartitioned_multithreaded_test(&self,partitioned_test_data: &[Vec<&(Vec<f64>,Vec<f64>)>]) -> f64 {
        let mut output = 0.0;
        thread::scope(|scope| {
            let (original_transmitter,receiver) = mpsc::channel();
            for partition in partitioned_test_data {
                let cloned_transmitter = original_transmitter.clone();
                scope.spawn(move || {
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
        });
        let mut length = 0;
        for partition in partitioned_test_data {
            length += partition.len();
        }
        output / length as f64
        
    }

    fn multithreaded_test(&self,test_data: &[(Vec<f64>,Vec<f64>)],num_partitions: f64) -> f64 {
        self.prepartitioned_multithreaded_test(&partition_data(test_data,num_partitions))
    }

    fn back_propagation(&self,training_data: &[(Vec<f64>,Vec<f64>)]) -> Self::Gradient {
        let mut output_gradient = self.build_zero_gradient();
        for (training_in,training_out) in training_data {
            let current_gradient = self.one_example_back_propagation(training_in, training_out);
            output_gradient.add(current_gradient);
        }
        output_gradient.div(training_data.len() as f64);
        output_gradient
    }

    fn multithreaded_back_propagation(&self,partitioned_training_data: &[Vec<&(Vec<f64>,Vec<f64>)>]) -> Self::Gradient {
        let mut output_gradient = self.build_zero_gradient();
        thread::scope(|scope| {
            let (original_transmitter,receiver) = mpsc::channel();
            for partition in partitioned_training_data {
                let cloned_transmitter = original_transmitter.clone();
                scope.spawn(move || {
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
        });
        let mut data_length = 0;
        for partition in partitioned_training_data {
            data_length += partition.len()
        }
        output_gradient.div(data_length as f64);
        output_gradient
    }

    fn backtracking_line_search_train(
        &mut self,
        training_data: &[(Vec<f64>,Vec<f64>)],
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
        training_data: &[(Vec<f64>,Vec<f64>)],
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
}

pub trait FlattenableGradient: Network {
    fn flatten_gradient(gradient: Self::Gradient) -> Vec<f64>;
    fn flatten_ref_gradient(gradient: &Self::Gradient) -> Vec<&f64>;
    fn subtract_flat_gradient(&mut self,flat_gradient: &[f64],learning_rate: f64);
    fn add_flat_gradient(&mut self,flat_gradient: &[f64],learning_rate: f64);
    fn build_zero_flat_gradient(&self) -> Vec<f64>;
    fn get_num_params(&self) -> usize;
}

pub trait FlattenableGradientInherentMethods: FlattenableGradient {
    fn l_bfgs_train(
        &mut self,
        training_data: &[(Vec<f64>,Vec<f64>)],
        iteration_memory: usize,
        first_checked_rate: f64,
        improvement_vs_curvature_bias: f64,
        slope_tolerance_parameter: f64,
        curvature_tolerance_parameter: f64,
        max_line_search_iterations: u32,
        max_iterations: u32
    );

    fn multithreaded_l_bfgs_train(
        &mut self,
        training_data: &[(Vec<f64>,Vec<f64>)],
        num_partitions: f64,
        iteration_memory: usize,
        first_checked_rate: f64,
        improvement_vs_curvature_bias: f64,
        slope_tolerance_parameter: f64,
        curvature_tolerance_parameter: f64,
        max_line_search_iterations: u32,
        max_iterations: u32
    );
}

impl<T: FlattenableGradient> FlattenableGradientInherentMethods for T {
    fn l_bfgs_train(
        &mut self,
        training_data: &[(Vec<f64>,Vec<f64>)],
        iteration_memory: usize,
        first_checked_rate: f64,
        improvement_vs_curvature_bias: f64,
        slope_tolerance_parameter: f64,
        curvature_tolerance_parameter: f64,
        max_line_search_iterations: u32,
        max_iterations: u32
    ) {
        let mut network_change_mem: VecDeque<Vec<f64>> = VecDeque::with_capacity(iteration_memory);
        let mut gradient_change_mem: VecDeque<Vec<f64>> = VecDeque::with_capacity(iteration_memory);
        let mut p_mem = VecDeque::with_capacity(iteration_memory);

        let mut previous_test = self.test(training_data);
        let current_gradient = <Self>::flatten_gradient(self.back_propagation(training_data));
        let mut search_direction = Vec::with_capacity(current_gradient.len());
        let mut next_gradient = Vec::new();
        for param_der in current_gradient.iter() {
            search_direction.push(-param_der);
        }
        let mut applied_lr = 0.0;
        let mut highest_curvature_failure = 0.0;
        let mut lowest_improvement_failure = first_checked_rate/improvement_vs_curvature_bias;
        let mut new_test = 0.0;
        let mut local_slope = 0.0;
        for (param_search_amount,param_der) in search_direction.iter().zip(current_gradient.iter()) {
            local_slope += param_search_amount * param_der;
        }
        let tolerable_local_slope = slope_tolerance_parameter * local_slope;
        let tolerable_non_local_slope = curvature_tolerance_parameter * local_slope.abs();

        let mut lr_of_last_gradient_calculation = -1.0;
        for _ in 0..max_line_search_iterations {
            let next_lr = (1.0-improvement_vs_curvature_bias) * highest_curvature_failure + improvement_vs_curvature_bias * lowest_improvement_failure;
            self.add_flat_gradient(&search_direction, next_lr-applied_lr);
            applied_lr = next_lr;
            new_test = self.test(training_data);
            println!("LR: {} \tTest: {}",applied_lr,new_test);
            if new_test > previous_test + applied_lr * tolerable_local_slope {
                lowest_improvement_failure = applied_lr;
                continue
            }
            next_gradient = <Self>::flatten_gradient(self.back_propagation(training_data));
            lr_of_last_gradient_calculation = applied_lr;
            let mut new_local_slope = 0.0;
            for (param_search_amount,new_param_der) in search_direction.iter().zip(next_gradient.iter()) {
                new_local_slope += param_search_amount * new_param_der;
            }
            if new_local_slope.abs() > tolerable_non_local_slope {
                highest_curvature_failure = applied_lr;
                continue
            }
            break
        }
        previous_test = new_test;

        if applied_lr != lr_of_last_gradient_calculation {
            next_gradient = <Self>::flatten_gradient(self.back_propagation(training_data));
        }

        for param in search_direction.iter_mut() {
            *param *= applied_lr;
        }
        let mut gradient_change = Vec::new();
        for (param_der,next_param_der) in current_gradient.iter().zip(next_gradient.iter()) {
            gradient_change.push(next_param_der-param_der);
        }
        let mut reciprocal_p = 0.0;
        for (param_search_amount,param_der_change) in search_direction.iter().zip(gradient_change.iter()) {
            reciprocal_p += param_search_amount * param_der_change;
        }
        network_change_mem.push_back(search_direction);
        gradient_change_mem.push_back(gradient_change);
        p_mem.push_back(1.0/reciprocal_p);
        
        for _ in 1..max_iterations {
            let current_gradient: Vec<f64> = next_gradient;
            next_gradient = Vec::with_capacity(0);//cause rust doesnt want to skip the reinitialization
            let mut search_direction = current_gradient.clone();
            
            let mut alpha_mem = Vec::new();
            for (network_change,(gradient_change,p)) in (
            network_change_mem.iter().zip(
            gradient_change_mem.iter().zip(
            p_mem.iter()))
            ).rev() {
                let mut alpha = 0.0;
                for (param_change,param_search_amount) in network_change.iter().zip(search_direction.iter()) {
                    alpha += param_change * param_search_amount;
                }
                alpha *= p;
                for (param_search_amount,param_der_change) in search_direction.iter_mut().zip(gradient_change.iter()) {
                    *param_search_amount -= alpha * param_der_change;
                }
                alpha_mem.push(alpha);
            }

            let mut mult_numerator = 0.0;
            let mut mult_denominator = 0.0;
            for (param_change,param_der_change) in network_change_mem[network_change_mem.len()-1].iter().zip(gradient_change_mem[gradient_change_mem.len()-1].iter()) {
                mult_numerator += param_change * param_der_change;
                mult_denominator += param_der_change * param_der_change;
            }
            let mult = mult_numerator/mult_denominator;
            for param_search_amount in search_direction.iter_mut() {
                *param_search_amount *= mult;
            }

            for (network_change,(gradient_change,(p,alpha))) in 
            network_change_mem.iter().zip(
            gradient_change_mem.iter().zip(
            p_mem.iter().zip(
            alpha_mem.iter().rev()
            ))) {
                let mut beta = 0.0;
                for (param_der_change,param_search_amount) in gradient_change.iter().zip(search_direction.iter()) {
                    beta += param_der_change * param_search_amount;
                }
                let alpha_beta_dif = alpha - beta * p;
                for (param_search_amount,param_change) in search_direction.iter_mut().zip(network_change.iter()) {
                    *param_search_amount += param_change * alpha_beta_dif;
                }
            }
            for param in search_direction.iter_mut() {
                *param *= -1.0;
            }

            let mut applied_lr = 0.0;
            let mut highest_curvature_failure = 0.0;
            let mut lowest_improvement_failure = first_checked_rate/improvement_vs_curvature_bias;
            let mut new_test = 0.0;
            let mut local_slope = 0.0;
            for (param_search_amount,param_der) in search_direction.iter().zip(current_gradient.iter()) {
                local_slope += param_search_amount * param_der;
            }
            let tolerable_local_slope = slope_tolerance_parameter * local_slope;
            let tolerable_non_local_slope = curvature_tolerance_parameter * local_slope.abs();

            let mut lr_of_last_gradient_calculation = -1.0;
            for _ in 0..max_line_search_iterations {
                let next_lr = (1.0-improvement_vs_curvature_bias) * highest_curvature_failure + improvement_vs_curvature_bias * lowest_improvement_failure;
                self.add_flat_gradient(&search_direction, next_lr-applied_lr);
                applied_lr = next_lr;
                new_test = self.test(training_data);
                println!("LR: {} \tTest: {}",applied_lr,new_test);
                if new_test > previous_test + applied_lr * tolerable_local_slope {
                    lowest_improvement_failure = applied_lr;
                    continue
                }
                next_gradient = <Self>::flatten_gradient(self.back_propagation(training_data));
                lr_of_last_gradient_calculation = applied_lr;
                let mut new_local_slope = 0.0;
                for (param_search_amount,new_param_der) in search_direction.iter().zip(next_gradient.iter()) {
                    new_local_slope += param_search_amount * new_param_der;
                }
                if new_local_slope.abs() > tolerable_non_local_slope {
                    highest_curvature_failure = applied_lr;
                    continue
                }
                break
            }
            previous_test = new_test;

            if applied_lr != lr_of_last_gradient_calculation {
                next_gradient = <Self>::flatten_gradient(self.back_propagation(training_data));
            }

            if p_mem.len() == iteration_memory {
                network_change_mem.pop_front();
                gradient_change_mem.pop_front();
                p_mem.pop_front();
            }
            for param in search_direction.iter_mut() {
                *param *= applied_lr;
            }
            let mut gradient_change = Vec::new();
            for (param_der,next_param_der) in current_gradient.iter().zip(next_gradient.iter()) {
                gradient_change.push(next_param_der-param_der);
            }
            let mut reciprocal_p = 0.0;
            for (param_search_amount,param_der_change) in search_direction.iter().zip(gradient_change.iter()) {
                reciprocal_p += param_search_amount * param_der_change;
            }
            network_change_mem.push_back(search_direction);
            gradient_change_mem.push_back(gradient_change);
            p_mem.push_back(1.0/reciprocal_p);
        }
    }

    fn multithreaded_l_bfgs_train(
        &mut self,
        training_data: &[(Vec<f64>,Vec<f64>)],
        num_partitions: f64,
        iteration_memory: usize,
        first_checked_rate: f64,
        improvement_vs_curvature_bias: f64,
        slope_tolerance_parameter: f64,
        curvature_tolerance_parameter: f64,
        max_line_search_iterations: u32,
        max_iterations: u32
    ) { 
        let partitioned_training_data = partition_data(training_data, num_partitions);

        let mut network_change_mem: VecDeque<Vec<f64>> = VecDeque::with_capacity(iteration_memory);
        let mut gradient_change_mem: VecDeque<Vec<f64>> = VecDeque::with_capacity(iteration_memory);
        let mut p_mem = VecDeque::with_capacity(iteration_memory);

        let mut previous_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
        let current_gradient = <Self>::flatten_gradient(self.multithreaded_back_propagation(&partitioned_training_data));
        let mut search_direction = Vec::with_capacity(current_gradient.len());
        let mut next_gradient = Vec::with_capacity(0);
        for param_der in current_gradient.iter() {
            search_direction.push(-param_der);
        }
        let mut applied_lr = 0.0;
        let mut highest_curvature_failure = 0.0;
        let mut lowest_improvement_failure = first_checked_rate/improvement_vs_curvature_bias;
        let mut new_test = 0.0;
        let mut local_slope = 0.0;
        for (param_search_amount,param_der) in search_direction.iter().zip(current_gradient.iter()) {
            local_slope += param_search_amount * param_der;
        }
        let tolerable_local_slope = slope_tolerance_parameter * local_slope;
        let tolerable_non_local_slope = curvature_tolerance_parameter * local_slope.abs();

        let mut lr_of_last_gradient_calculation = -1.0;
        for _ in 0..max_line_search_iterations {
            let next_lr = (1.0-improvement_vs_curvature_bias) * highest_curvature_failure + improvement_vs_curvature_bias * lowest_improvement_failure;
            self.add_flat_gradient(&search_direction, next_lr-applied_lr);
            applied_lr = next_lr;
            new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
            println!("LR: {} \tTest: {}",applied_lr,new_test);
            if new_test > previous_test + applied_lr * tolerable_local_slope {
                lowest_improvement_failure = applied_lr;
                continue
            }
            next_gradient = <Self>::flatten_gradient(self.multithreaded_back_propagation(&partitioned_training_data));
            lr_of_last_gradient_calculation = applied_lr;
            let mut new_local_slope = 0.0;
            for (param_search_amount,new_param_der) in search_direction.iter().zip(next_gradient.iter()) {
                new_local_slope += param_search_amount * new_param_der;
            }
            if new_local_slope.abs() > tolerable_non_local_slope {
                highest_curvature_failure = applied_lr;
                continue
            }
            break
        }
        previous_test = new_test;

        if applied_lr != lr_of_last_gradient_calculation {
            next_gradient = <Self>::flatten_gradient(self.multithreaded_back_propagation(&partitioned_training_data));
        }

        for param in search_direction.iter_mut() {
            *param *= applied_lr;
        }
        let mut gradient_change = Vec::new();
        for (param_der,next_param_der) in current_gradient.iter().zip(next_gradient.iter()) {
            gradient_change.push(next_param_der-param_der);
        }
        let mut reciprocal_p = 0.0;
        for (param_search_amount,param_der_change) in search_direction.iter().zip(gradient_change.iter()) {
            reciprocal_p += param_search_amount * param_der_change;
        }
        network_change_mem.push_back(search_direction);
        gradient_change_mem.push_back(gradient_change);
        p_mem.push_back(1.0/reciprocal_p);
        
        for _ in 1..max_iterations {
            let current_gradient: Vec<f64> = next_gradient;
            next_gradient = Vec::with_capacity(0);//cause rust doesnt want to skip the reinitialization
            let mut search_direction = current_gradient.clone();
            
            let mut alpha_mem = Vec::new();
            for (network_change,(gradient_change,p)) in (
            network_change_mem.iter().zip(
            gradient_change_mem.iter().zip(
            p_mem.iter()))
            ).rev() {
                let mut alpha = 0.0;
                for (param_change,param_search_amount) in network_change.iter().zip(search_direction.iter()) {
                    alpha += param_change * param_search_amount;
                }
                alpha *= p;
                for (param_search_amount,param_der_change) in search_direction.iter_mut().zip(gradient_change.iter()) {
                    *param_search_amount -= alpha * param_der_change;
                }
                alpha_mem.push(alpha);
            }

            let mut mult_numerator = 0.0;
            let mut mult_denominator = 0.0;
            for (param_change,param_der_change) in network_change_mem[network_change_mem.len()-1].iter().zip(gradient_change_mem[gradient_change_mem.len()-1].iter()) {
                mult_numerator += param_change * param_der_change;
                mult_denominator += param_der_change * param_der_change;
            }
            let mult = mult_numerator/mult_denominator;
            for param_search_amount in search_direction.iter_mut() {
                *param_search_amount *= mult;
            }

            for (network_change,(gradient_change,(p,alpha))) in 
            network_change_mem.iter().zip(
            gradient_change_mem.iter().zip(
            p_mem.iter().zip(
            alpha_mem.iter().rev()
            ))) {
                let mut beta = 0.0;
                for (param_der_change,param_search_amount) in gradient_change.iter().zip(search_direction.iter()) {
                    beta += param_der_change * param_search_amount;
                }
                let alpha_beta_dif = alpha - beta * p;
                for (param_search_amount,param_change) in search_direction.iter_mut().zip(network_change.iter()) {
                    *param_search_amount += param_change * alpha_beta_dif;
                }
            }
            for param in search_direction.iter_mut() {
                *param *= -1.0;
            }
            
            let mut applied_lr = 0.0;
            let mut highest_curvature_failure = 0.0;
            let mut lowest_improvement_failure = first_checked_rate/improvement_vs_curvature_bias;
            let mut new_test = 0.0;
            let mut local_slope = 0.0;
            for (param_search_amount,param_der) in search_direction.iter().zip(current_gradient.iter()) {
                local_slope += param_search_amount * param_der;
            }
            let tolerable_local_slope = slope_tolerance_parameter * local_slope;
            let tolerable_non_local_slope = curvature_tolerance_parameter * local_slope.abs();

            let mut lr_of_last_gradient_calculation = -1.0;
            for _ in 0..max_line_search_iterations {
                let next_lr = (1.0-improvement_vs_curvature_bias) * highest_curvature_failure + improvement_vs_curvature_bias * lowest_improvement_failure;
                self.add_flat_gradient(&search_direction, next_lr-applied_lr);
                applied_lr = next_lr;
                new_test = self.prepartitioned_multithreaded_test(&partitioned_training_data);
                println!("LR: {} \tTest: {}",applied_lr,new_test);
                if new_test > previous_test + applied_lr * tolerable_local_slope {
                    lowest_improvement_failure = applied_lr;
                    continue
                }
                next_gradient = <Self>::flatten_gradient(self.multithreaded_back_propagation(&partitioned_training_data));
                lr_of_last_gradient_calculation = applied_lr;
                let mut new_local_slope = 0.0;
                for (param_search_amount,new_param_der) in search_direction.iter().zip(next_gradient.iter()) {
                    new_local_slope += param_search_amount * new_param_der;
                }
                if new_local_slope.abs() > tolerable_non_local_slope {
                    highest_curvature_failure = applied_lr;
                    continue
                }
                break
            }

            if applied_lr != lr_of_last_gradient_calculation {
                next_gradient = <Self>::flatten_gradient(self.multithreaded_back_propagation(&partitioned_training_data));
            }

            previous_test = new_test;
            if p_mem.len() == iteration_memory {
                network_change_mem.pop_front();
                gradient_change_mem.pop_front();
                p_mem.pop_front();
            }
            for param in search_direction.iter_mut() {
                *param *= applied_lr;
            }
            let mut gradient_change = Vec::new();
            for (param_der,next_param_der) in current_gradient.iter().zip(next_gradient.iter()) {
                gradient_change.push(next_param_der-param_der);
            }
            let mut reciprocal_p = 0.0;
            for (param_search_amount,param_der_change) in search_direction.iter().zip(gradient_change.iter()) {
                reciprocal_p += param_search_amount * param_der_change;
            }
            network_change_mem.push_back(search_direction);
            gradient_change_mem.push_back(gradient_change);
            p_mem.push_back(1.0/reciprocal_p);
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
        temp_val*(1.0-temp_val)
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
#[derive(Clone)]
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
            biases,
            weights,
            node_layout,
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
        for (layer, previous_layer) in (node_layout[1..node_layout.len()]).iter().zip((node_layout[0..node_layout.len()-1]).iter()) { //first iteration is on the 2nd layer, layer num is 0 and refers to 1st layer making the previous layer
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
            biases,
            weights,
            node_layout,
            activation_fn: activation_fn.0,
            derivative_activation_fn: activation_fn.1
        }
    }

    pub fn get_biases(&self) -> &Vec<Vec<f64>> {
        &self.biases
    }
    pub fn get_weights(&self) -> &Vec<Vec<Vec<f64>>> {
        &self.weights
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
        for (layer, previous_layer) in (self.node_layout[1..self.node_layout.len()]).iter().zip((self.node_layout[0..self.node_layout.len()-1]).iter()) { //first iteration is on the 2nd layer, layer num is 0 and refers to 1st layer making the previous layer
            let layer_biases = vec![0.0; *layer];
            let mut layer_weights = Vec::with_capacity(*layer);
            for _ in 0..*layer {
                //layer_biases.push(0.0);
                let node_weights = vec![0.0;*previous_layer];
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
            for (input_node_weights,node_pre_activation) in transpose_matrix(forward_layer_weights).zip(current_pre_activations.iter()) {
                let mut new_node_derivative = 0.0;
                for (weight,forward_derivative) in input_node_weights.into_iter().zip(biases_gradient[biases_gradient.len()-1].iter()) {//layer_node_derivatives.iter()) { //seems to work
                    new_node_derivative += weight * forward_derivative;
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

    fn subtract_flat_gradient(&mut self,flat_gradient: &[f64],learning_rate: f64) {
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

    fn add_flat_gradient(&mut self,flat_gradient: &[f64],learning_rate: f64) {
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
        vec![0.0;self.get_num_params()]
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

pub mod splice{
    use super::*;

    pub trait SpliceableNetwork: Network {
        type SRunInfo: Send + Sync;

        fn special_run(&self, inputs: &[f64]) -> (Vec<f64>,Self::SRunInfo);
        fn full_der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> (Vec<f64>, Self::Gradient);
        fn der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> Self::Gradient;
        fn get_inputs(&self) -> usize;
        fn get_outputs(&self) -> usize;

        fn chain_run(&self, input_iterator: &mut ChainOutput) -> ChainOutput {
            ChainOutput::Tail(self.run(&input_iterator.take(self.get_inputs()).collect::<Vec<f64>>()).into_iter())
        }
        fn chain_special_run(&self, input_iterator: &mut ChainOutput) -> (ChainOutput,Self::SRunInfo) {
            let (output,info) = self.special_run(&input_iterator.take(self.get_inputs()).collect::<Vec<f64>>());
            (ChainOutput::Tail(output.into_iter()),info)
        }
        fn chain_full_der_run(&self, output_node_der_iterator: &mut ChainOutput, s_run_info: Self::SRunInfo) -> (ChainOutput,Self::Gradient) {
            let (output, gradient) = self.full_der_run(&output_node_der_iterator.take(self.get_outputs()).collect::<Vec<f64>>(),s_run_info);
            (ChainOutput::Tail(output.into_iter()),gradient)
        }
        fn chain_der_run(&self,output_node_der_iterator: &mut ChainOutput,s_run_info: Self::SRunInfo) -> Self::Gradient {
            self.der_run(&output_node_der_iterator.take(self.get_outputs()).collect::<Vec<f64>>(),s_run_info)
        }

        fn chain_out_only_run(&self, inputs: &[f64]) -> ChainOutput {
            ChainOutput::Tail(self.run(inputs).into_iter())
        }
        fn chain_out_only_special_run(&self, inputs: &[f64]) -> (ChainOutput,Self::SRunInfo) {
            let (output,info) = self.special_run(inputs);
            (ChainOutput::Tail(output.into_iter()),info)
        }
        fn chain_out_only_full_der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> (ChainOutput,Self::Gradient) {
            let (output, gradient) = self.full_der_run(output_node_ders,s_run_info);
            (ChainOutput::Tail(output.into_iter()),gradient)
        }
    }

    impl SpliceableNetwork for MultiLayerPerceptron {
        type SRunInfo = (Vec<Vec<f64>>,Vec<Vec<f64>>);

        fn special_run(&self, inputs: &[f64]) -> (Vec<f64>,Self::SRunInfo) {
            let mut network_pre_activations = Vec::with_capacity(self.node_layout.len()-1);
            let mut network_activations = Vec::with_capacity(self.node_layout.len());
            network_activations.push(inputs.to_vec()); //remove this is possible
            let mut layer_activations = Vec::with_capacity(self.node_layout[1]);
            let mut layer_pre_activations = Vec::with_capacity(self.node_layout[1]);
            for (node_bias,node_weights) in self.biases[0].iter().zip(self.weights[0].iter()) {
                let mut node_val_before_activation_fn = *node_bias;
                for (weight,previous_node_activation) in node_weights.iter().zip(inputs.iter()) {
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
            //last activation layer skipped so it is usable as output

            (layer_activations,(network_activations,network_pre_activations))
        }

        fn full_der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> (Vec<f64>, Self::Gradient) {
            let mut biases_gradient = Vec::with_capacity(self.node_layout.len());
            let mut weights_gradient = Vec::with_capacity(self.node_layout.len());
            
            let (network_activations,network_pre_activations) = s_run_info;
            //last layer specific calculations
            let mut last_layer_biases_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-1]);
            let mut last_layer_weights_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-1]);
            for (pre_activation,output_node_der) in 
            network_pre_activations[network_pre_activations.len()-1].iter().zip(
            output_node_ders.iter()) {
                let current_node_derivative = output_node_der*(self.derivative_activation_fn)(*pre_activation);
                last_layer_biases_gradient.push(current_node_derivative);
                let mut node_weights_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-2]);
                for previous_node_activation in network_activations[network_activations.len()-1].iter() {
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
            network_activations[0..network_activations.len()-1].iter().rev())) {
                let mut layer_biases_gradient = Vec::with_capacity(current_pre_activations.len());
                let mut layer_weights_gradient = Vec::with_capacity(current_pre_activations.len());
                for (input_node_weights,node_pre_activation) in transpose_matrix(forward_layer_weights).zip(current_pre_activations.iter()) {
                    let mut new_node_derivative = 0.0;
                    for (weight,forward_derivative) in input_node_weights.into_iter().zip(biases_gradient[biases_gradient.len()-1].iter()) {//layer_node_derivatives.iter()) { //seems to work
                        new_node_derivative += weight * forward_derivative;
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

            let mut input_node_ders = Vec::with_capacity(self.node_layout[0]);
            for input_node_weights in transpose_matrix(&self.weights[0]) {
                let mut new_node_der = 0.0;
                for (weight, forward_derivative) in input_node_weights.into_iter().zip(biases_gradient[biases_gradient.len()-1].iter()) {
                    new_node_der += weight * forward_derivative;
                }
                input_node_ders.push(new_node_der);
            }

            biases_gradient = biases_gradient.into_iter().rev().collect();
            weights_gradient = weights_gradient.into_iter().rev().collect();

            (input_node_ders,(biases_gradient,weights_gradient))
        }

        fn der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> Self::Gradient {
            let mut biases_gradient = Vec::with_capacity(self.node_layout.len());
            let mut weights_gradient = Vec::with_capacity(self.node_layout.len());
            
            let (network_activations,network_pre_activations) = s_run_info;
            //last layer specific calculations
            let mut last_layer_biases_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-1]);
            let mut last_layer_weights_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-1]);
            for (pre_activation,output_node_der) in 
            network_pre_activations[network_pre_activations.len()-1].iter().zip(
            output_node_ders.iter()) {
                let current_node_derivative = output_node_der*(self.derivative_activation_fn)(*pre_activation);
                last_layer_biases_gradient.push(current_node_derivative);
                let mut node_weights_gradient = Vec::with_capacity(self.node_layout[self.node_layout.len()-2]);
                for previous_node_activation in network_activations[network_activations.len()-1].iter() {
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
            network_activations[0..network_activations.len()-1].iter().rev())) {
                let mut layer_biases_gradient = Vec::with_capacity(current_pre_activations.len());
                let mut layer_weights_gradient = Vec::with_capacity(current_pre_activations.len());
                for (input_node_weights,node_pre_activation) in transpose_matrix(forward_layer_weights).zip(current_pre_activations.iter()) {
                    let mut new_node_derivative = 0.0;
                    for (weight,forward_derivative) in input_node_weights.into_iter().zip(biases_gradient[biases_gradient.len()-1].iter()) {//layer_node_derivatives.iter()) { //seems to work
                        new_node_derivative += weight * forward_derivative;
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

        fn get_inputs(&self) -> usize {
            self.node_layout[0]
        }
        fn get_outputs(&self) -> usize {
            self.node_layout[self.node_layout.len()-1]
        }
    }

    pub trait ObjectSafeSpliceableNetwork: Send + Sync {
        fn os_run(&self, inputs: &[f64]) -> Vec<f64>;
        fn os_subtract_gradient(&mut self,gradient: &[f64],learning_rate: f64);
        fn os_add_gradient(&mut self,gradient: &[f64],learning_rate: f64);
        fn os_build_zero_gradient(&self) -> Vec<f64>;
        fn os_get_inputs(&self) -> usize;
        fn os_get_outputs(&self) -> usize;
        fn get_wrapper(&self) -> Box<dyn ObjectSafeWrappedSpliceableNetwork + '_>;
    }

    impl<T: SpliceableNetwork + FlattenableGradient> ObjectSafeSpliceableNetwork for T {
        fn os_run(&self, inputs: &[f64]) -> Vec<f64> {
            self.run(inputs)
        }
        fn os_subtract_gradient(&mut self,gradient: &[f64],learning_rate: f64) {
            self.subtract_flat_gradient(gradient, learning_rate);
        }
        fn os_add_gradient(&mut self,gradient: &[f64],learning_rate: f64) {
            self.add_flat_gradient(gradient, learning_rate);
        }
        fn os_build_zero_gradient(&self) -> Vec<f64> {
            self.build_zero_flat_gradient()
        }
        fn os_get_inputs(&self) -> usize {
            self.get_inputs()
        }
        fn os_get_outputs(&self) -> usize {
            self.get_outputs()
        }
        fn get_wrapper(&self) -> Box<dyn ObjectSafeWrappedSpliceableNetwork + '_> {
            Box::new(ObjectSafeSpliceableNetworkWrapper {
                network: self,
                stored_s_run_infos: Vec::new(),
                gradient: None
            })
        }
    }

    pub struct ObjectSafeSpliceableNetworkWrapper<'a,T: SpliceableNetwork + FlattenableGradient> {
        network: &'a T,
        stored_s_run_infos: Vec<T::SRunInfo>,
        gradient: Option<T::Gradient>
    }

    pub trait ObjectSafeWrappedSpliceableNetwork: Send + Sync {
        fn run(&self, inputs: &[f64]) -> Vec<f64>;
        fn special_run(&mut self, inputs: &[f64]) -> Vec<f64>;
        fn der_run(&mut self,output_node_ders: &[f64]);
        fn full_der_run(&mut self,output_node_ders: &[f64]) -> Vec<f64>;
        fn flush_gradient(&mut self) -> Vec<f64>;
        fn build_zero_gradient(&self) -> Vec<f64>;
        fn get_inputs(&self) -> usize;
        fn get_outputs(&self) -> usize;
    }

    impl<T: SpliceableNetwork + FlattenableGradient> ObjectSafeWrappedSpliceableNetwork for ObjectSafeSpliceableNetworkWrapper<'_,T> {
        fn run(&self, inputs: &[f64]) -> Vec<f64> {
            self.network.run(inputs)
        }
        fn special_run(&mut self, inputs: &[f64]) -> Vec<f64> {
            let (outputs,info) = self.network.special_run(inputs);
            self.stored_s_run_infos.push(info);
            outputs
        }
        fn der_run(&mut self,output_node_ders: &[f64]) {
            let s_run_info = self.stored_s_run_infos.pop().unwrap();
            let gradient = self.network.der_run(output_node_ders,s_run_info);
            match &mut self.gradient {
                Some(current_gradient) => {current_gradient.add(gradient)}
                None => {self.gradient = Some(gradient)}
            }
        }
        fn full_der_run(&mut self,output_node_ders: &[f64]) -> Vec<f64> {
            let s_run_info = self.stored_s_run_infos.pop().unwrap();
            let (input_node_ders,gradient) = self.network.full_der_run(output_node_ders,s_run_info);
            match &mut self.gradient {
                Some(current_gradient) => {current_gradient.add(gradient)}
                None => {self.gradient = Some(gradient)}
            }
            input_node_ders
        }
        fn flush_gradient(&mut self) -> Vec<f64> {
            <T>::flatten_gradient(std::mem::replace(&mut self.gradient, None).unwrap_or_else(|| {panic!("No gradient to Flush")}))
        }
        fn build_zero_gradient(&self) -> Vec<f64> {
            self.network.build_zero_flat_gradient()
        }
        fn get_inputs(&self) -> usize {
            self.network.get_inputs()
        }
        fn get_outputs(&self) -> usize {
            self.network.get_outputs()
        }
    }

    impl Gradient for Vec<Vec<f64>> {
        fn add(&mut self,other: Self) where Self: Sized {
            for (self_gradient,other_gradient) in self.iter_mut().zip(other.into_iter()) {
                for (self_param,other_param) in self_gradient.iter_mut().zip(other_gradient.into_iter()) {
                    *self_param += other_param;
                }
            }
        }
        fn div(&mut self,divisor: f64) where Self: Sized {
            for gradient in self.iter_mut() {
                for param in gradient {
                    *param /= divisor;
                }
            }
        }
        fn mult(&mut self,factor: f64) where Self: Sized {
            for gradient in self.iter_mut() {
                for param in gradient {
                    *param *= factor;
                }
            }
        }
        fn dot_product(&self,other: &Self) -> f64 where Self: Sized {
            let mut output = 0.0;
            for (self_gradient,other_gradient) in self.iter().zip(other.iter()) {
                for (self_param,other_param) in self_gradient.iter().zip(other_gradient.iter()) {
                    output += self_param * other_param;
                }
            }
            output
        }
    }

    pub struct CompositeNetwork {
        network_layout: Vec<Vec<usize>>,
        networks: Vec<Box<dyn ObjectSafeSpliceableNetwork>>
    }

    impl CompositeNetwork {
        pub fn new(network_layout: Vec<Vec<usize>>,networks: Vec<Box<dyn ObjectSafeSpliceableNetwork>>) -> Self {
            Self{
                network_layout,
                networks
            }
        }
    }

    impl Network for CompositeNetwork {
        type Gradient = Vec<Vec<f64>>;

        fn run(&self,inputs: &[f64]) -> Vec<f64> {
            let mut layer_outputs = Vec::new();
            let mut current_first_input_index = 0;
            for network_index in self.network_layout[0].iter() {
                let current_net = &self.networks[*network_index];
                for output in current_net.os_run(&inputs[current_first_input_index..current_first_input_index+current_net.os_get_inputs()]) {
                    layer_outputs.push(output);
                }
                current_first_input_index += current_net.os_get_inputs();
            }
            for layer_networks in self.network_layout.iter() {
                let mut next_layer_outputs = Vec::new();
                let mut current_first_input_index = 0;
                for network_index in layer_networks {
                    let current_net = &self.networks[*network_index];
                    for output in current_net.os_run(&layer_outputs[current_first_input_index..current_first_input_index+current_net.os_get_inputs()]) {
                        next_layer_outputs.push(output);
                    }
                    current_first_input_index += current_net.os_get_inputs();
                }
                layer_outputs = next_layer_outputs;
            }
            layer_outputs
        }
        fn subtract_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64) {
            for (network, network_gradient) in self.networks.iter_mut().zip(gradient.iter()) {
                network.os_subtract_gradient(network_gradient, learning_rate);
            }
        }
        fn add_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64) {
            for (network, network_gradient) in self.networks.iter_mut().zip(gradient.iter()) {
                network.os_add_gradient(network_gradient, learning_rate);
            }
        }
        fn build_zero_gradient(&self) -> Self::Gradient {
            let mut output = Vec::new();
            for network in self.networks.iter() {
                output.push(network.os_build_zero_gradient());
            }
            output
        }
        fn one_example_back_propagation(&self,training_in: &[f64],training_out: &[f64]) -> Self::Gradient {
            let mut network_wrappers = Vec::new();
            for network in self.networks.iter() {
                network_wrappers.push(network.get_wrapper());
            }
            let mut layer_outputs = Vec::new();
            let mut current_first_input_index = 0;
            for network_index in self.network_layout[0].iter() {
                let current_net = &mut network_wrappers[*network_index];
                for output in current_net.special_run(&training_in[current_first_input_index..current_first_input_index+current_net.get_inputs()]) {
                    layer_outputs.push(output);
                }
                current_first_input_index += current_net.get_inputs();
            }
            for layer_networks in self.network_layout[1..self.network_layout.len()].iter() {
                let mut next_layer_outputs = Vec::new();
                let mut current_first_input_index = 0;
                for network_index in layer_networks {
                    let current_net = &mut network_wrappers[*network_index];
                    for output in current_net.special_run(&layer_outputs[current_first_input_index..current_first_input_index+current_net.get_inputs()]) {
                        next_layer_outputs.push(output);
                    }
                    current_first_input_index += current_net.get_inputs();
                }
                layer_outputs = next_layer_outputs;
            }
            for (real_val,expected_val) in layer_outputs.iter_mut().zip(training_out.iter()) {
                *real_val = 2.0*(*real_val-expected_val)
            }
            for layer_networks in self.network_layout[1..self.network_layout.len()].iter().rev() {
                let mut next_layer_node_ders = Vec::new();
                let mut current_last_der_index = layer_outputs.len(); //techinacally plus one
                for network_index in layer_networks.iter().rev() {
                    let current_net = &mut network_wrappers[*network_index];
                    next_layer_node_ders.push(current_net.full_der_run(&layer_outputs[current_last_der_index-current_net.get_outputs()..current_last_der_index]));
                    current_last_der_index -= current_net.get_outputs();
                }
                layer_outputs = Vec::new();
                for network_input_ders in next_layer_node_ders.into_iter().rev() {
                    for input_node_der in network_input_ders {
                        layer_outputs.push(input_node_der);
                    }
                }
            }
            let mut current_last_der_index = layer_outputs.len(); //see above
            for network_index in self.network_layout[0].iter().rev() {
                let current_net = &mut network_wrappers[*network_index];
                current_net.der_run(&layer_outputs[current_last_der_index-current_net.get_outputs()..current_last_der_index]);
                current_last_der_index -= current_net.get_outputs();
            }
            let mut output_gradient = Vec::new();
            for mut wrapper in network_wrappers {
                output_gradient.push(wrapper.flush_gradient());
            }
            output_gradient

        }
    }

    pub struct DualSequentialNetwork<F: SpliceableNetwork,G: SpliceableNetwork> {
        pub left_network: F,
        pub right_network: G,
    }
    
    impl<LG: Gradient, RG: Gradient> Gradient for (LG,RG) {
        fn add(&mut self,other: Self) {
            self.0.add(other.0);
            self.1.add(other.1);
        }
        fn div(&mut self,divisor: f64) {
            self.0.div(divisor);
            self.1.div(divisor);
        }
        fn mult(&mut self,factor: f64) {
            self.0.mult(factor);
            self.1.mult(factor);
        }
        fn dot_product(&self,other: &Self) -> f64 {
            self.0.dot_product(&other.0) + self.1.dot_product(&other.1)
        }
    }

    impl<F: SpliceableNetwork,G: SpliceableNetwork> Network for DualSequentialNetwork<F,G> {
        type Gradient = (F::Gradient,G::Gradient);

        fn run(&self,inputs: &[f64]) -> Vec<f64> {
            self.right_network.chain_run(&mut self.left_network.chain_out_only_run(inputs)).collect()
        }
        fn subtract_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64) {
            self.left_network.subtract_gradient(&gradient.0, learning_rate);
            self.right_network.subtract_gradient(&gradient.1, learning_rate);
        }
        fn add_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64) {
            self.left_network.add_gradient(&gradient.0, learning_rate);
            self.right_network.add_gradient(&gradient.1, learning_rate);
        }
        fn build_zero_gradient(&self) -> Self::Gradient {
            (self.left_network.build_zero_gradient(),self.right_network.build_zero_gradient())
        }
        fn one_example_back_propagation(&self,training_in: &[f64],training_out: &[f64]) -> Self::Gradient {
            let (mut mid_outputs,left_info) = self.left_network.chain_out_only_special_run(training_in);
            let (outputs,right_info) = self.right_network.chain_special_run(&mut mid_outputs);
            let mut output_node_ders = Vec::new();
            for (real_val,expected_val) in outputs.zip(training_out.iter()) {
                output_node_ders.push(2.0*(real_val-expected_val));
            }
            let (mut mid_node_ders,right_gradient) = self.right_network.chain_out_only_full_der_run(&output_node_ders, right_info);
            let left_gradient = self.left_network.chain_der_run(&mut mid_node_ders, left_info);
            (left_gradient,right_gradient)

        }
    }

    impl<F: SpliceableNetwork,G: SpliceableNetwork> SpliceableNetwork for DualSequentialNetwork<F,G>{
        type SRunInfo = (F::SRunInfo,G::SRunInfo);

        fn special_run(&self, inputs: &[f64]) -> (Vec<f64>,Self::SRunInfo) {
            let (mid_outputs, left_info) = self.left_network.special_run(inputs);
            let (outputs, right_info) = self.right_network.special_run(&mid_outputs);
            (outputs,(left_info,right_info))
        }
        fn full_der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> (Vec<f64>, Self::Gradient) {
            let (mid_node_ders, right_gradient) = self.right_network.full_der_run(output_node_ders, s_run_info.1);
            let (input_node_ders, left_gradient) = self.left_network.full_der_run(&mid_node_ders, s_run_info.0);
            (input_node_ders,(left_gradient,right_gradient))
        }
        fn der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> Self::Gradient {
            let (mid_node_ders,right_gradient) = self.right_network.full_der_run(output_node_ders, s_run_info.1);
            (self.left_network.der_run(&mid_node_ders, s_run_info.0),right_gradient)
        }
        fn get_inputs(&self) -> usize {
            self.left_network.get_inputs()
        }
        fn get_outputs(&self) -> usize {
            self.right_network.get_outputs()
        }
    }

    impl<F: SpliceableNetwork+FlattenableGradient,G: SpliceableNetwork+FlattenableGradient> FlattenableGradient for DualSequentialNetwork<F,G> {
        fn flatten_gradient(gradient: Self::Gradient) -> Vec<f64> {
            let mut output = <F>::flatten_gradient(gradient.0);
            output.extend(<G>::flatten_gradient(gradient.1).into_iter());
            output
        }
        fn flatten_ref_gradient(gradient: &Self::Gradient) -> Vec<&f64> {
            let mut output = <F>::flatten_ref_gradient(&gradient.0);
            output.extend(<G>::flatten_ref_gradient(&gradient.1).into_iter());
            output
        }
        fn add_flat_gradient(&mut self,flat_gradient: &[f64],learning_rate: f64) {
            self.left_network.add_flat_gradient(&flat_gradient[0..self.left_network.get_num_params()], learning_rate);
            self.right_network.add_flat_gradient(&flat_gradient[self.left_network.get_num_params()..flat_gradient.len()], learning_rate);
        }
        fn subtract_flat_gradient(&mut self,flat_gradient: &[f64],learning_rate: f64) {
            self.left_network.subtract_flat_gradient(&flat_gradient[0..self.left_network.get_num_params()], learning_rate);
            self.right_network.subtract_flat_gradient(&flat_gradient[self.left_network.get_num_params()..flat_gradient.len()], learning_rate);
        }
        fn build_zero_flat_gradient(&self) -> Vec<f64> {
            let mut output = self.left_network.build_zero_flat_gradient();
            output.extend(self.right_network.build_zero_flat_gradient());
            output
        }
        fn get_num_params(&self) -> usize {
            self.left_network.get_num_params() + self.right_network.get_num_params()
        }
    }

    pub enum ChainOutput {
        Chain(Box<std::iter::Chain<ChainOutput,ChainOutput>>),
        Tail(std::vec::IntoIter<f64>)
    }

    impl Iterator for ChainOutput {
        type Item = f64;

        fn next(&mut self) -> Option<Self::Item> {
            match self {
                Self::Chain(inner_iter)  => {inner_iter.next()}
                Self::Tail(inner_iter) => {inner_iter.next()}
            }
        }
    }

    pub struct DualParallelNetwork<F: SpliceableNetwork,G: SpliceableNetwork> {
        pub top_network: F,
        pub bottom_network: G
    }

    impl<F: SpliceableNetwork,G: SpliceableNetwork> Network for DualParallelNetwork<F,G> {
        type Gradient = (F::Gradient,G::Gradient);

        fn run(&self,inputs: &[f64]) -> Vec<f64> {
            let top_chain_output = self.top_network.chain_out_only_run(&inputs[0..self.top_network.get_inputs()]);
            let bottom_chain_output = self.bottom_network.chain_out_only_run(&inputs[self.top_network.get_inputs()..inputs.len()]);
            top_chain_output.chain(bottom_chain_output).collect()
        }
        fn subtract_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64) {
            self.top_network.subtract_gradient(&gradient.0, learning_rate);
            self.bottom_network.subtract_gradient(&gradient.1, learning_rate);
        }
        fn add_gradient(&mut self,gradient: &Self::Gradient,learning_rate: f64) {
            self.top_network.add_gradient(&gradient.0, learning_rate);
            self.bottom_network.add_gradient(&gradient.1, learning_rate);
        }
        fn build_zero_gradient(&self) -> Self::Gradient {
            let top_gradient = self.top_network.build_zero_gradient();
            let bottom_gradient = self.bottom_network.build_zero_gradient();
            (top_gradient,bottom_gradient)
        }
        fn one_example_back_propagation(&self,training_in: &[f64],training_out: &[f64]) -> Self::Gradient {
            let (top_chain_output,top_info) = self.top_network.chain_out_only_special_run(&training_in[0..self.top_network.get_inputs()]);
            let (bottom_chain_output,bottom_info) = self.bottom_network.chain_out_only_special_run(&training_in[self.top_network.get_inputs()..training_in.len()]);
            let mut output_node_ders = Vec::new();
            for (real_val,expected_val) in top_chain_output.chain(bottom_chain_output).zip(training_out.iter()) {
                output_node_ders.push(2.0*(real_val-expected_val));
            }
            let top_gradient = self.top_network.der_run(&output_node_ders[0..self.top_network.get_outputs()], top_info);
            let bottom_gradient = self.bottom_network.der_run(&output_node_ders[self.top_network.get_outputs()..output_node_ders.len()], bottom_info);
            (top_gradient,bottom_gradient)
        }
    }

    impl<F: SpliceableNetwork,G: SpliceableNetwork> SpliceableNetwork for DualParallelNetwork<F,G> {
        type SRunInfo = (F::SRunInfo,G::SRunInfo);

        fn special_run(&self, inputs: &[f64]) -> (Vec<f64>,Self::SRunInfo) {
            let (top_chain_output,top_info) = self.top_network.chain_out_only_special_run(&inputs[0..self.top_network.get_inputs()]);
            let (bottom_chain_output, bottom_info) = self.bottom_network.chain_out_only_special_run(&inputs[self.top_network.get_inputs()..inputs.len()]);
            (top_chain_output.chain(bottom_chain_output).collect(),(top_info,bottom_info))
        }
        fn full_der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> (Vec<f64>, Self::Gradient) {
            let (top_chain_ders,top_gradient) = self.top_network.chain_out_only_full_der_run(&output_node_ders[0..self.top_network.get_outputs()], s_run_info.0);
            let (bottom_chain_ders, bottom_gradient) = self.bottom_network.chain_out_only_full_der_run(&output_node_ders[self.top_network.get_outputs()..output_node_ders.len()], s_run_info.1);
            (top_chain_ders.chain(bottom_chain_ders).collect(),(top_gradient,bottom_gradient))
        }
        fn der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> Self::Gradient {
            let top_gradient = self.top_network.der_run(&output_node_ders[0..self.top_network.get_inputs()], s_run_info.0);
            let bottom_gradient = self.bottom_network.der_run(&output_node_ders[self.top_network.get_inputs()..output_node_ders.len()],s_run_info.1);
            (top_gradient,bottom_gradient)
        }
        fn get_inputs(&self) -> usize {
            self.top_network.get_inputs() + self.bottom_network.get_inputs()
        }
        fn get_outputs(&self) -> usize {
            self.top_network.get_outputs() + self.bottom_network.get_outputs()
        }
    
        fn chain_run(&self, input_iterator: &mut ChainOutput) -> ChainOutput {
            let top_chain_output = self.top_network.chain_run(input_iterator);
            let bottom_chain_output = self.bottom_network.chain_run(input_iterator);
            ChainOutput::Chain(Box::new(top_chain_output.chain(bottom_chain_output)))
        }
        fn chain_special_run(&self, input_iterator: &mut ChainOutput) -> (ChainOutput,Self::SRunInfo) {
            let (top_chain_output,top_info) = self.top_network.chain_special_run(input_iterator);
            let (bottom_chain_output,bottom_info) = self.bottom_network.chain_special_run(input_iterator);
            (ChainOutput::Chain(Box::new(top_chain_output.chain(bottom_chain_output))),(top_info,bottom_info))
        }
        fn chain_full_der_run(&self, output_node_der_iterator: &mut ChainOutput, s_run_info: Self::SRunInfo) -> (ChainOutput,Self::Gradient) {
            let (top_chain_ders,top_gradient) = self.top_network.chain_full_der_run(output_node_der_iterator, s_run_info.0);
            let (bottom_chain_output,bottom_info) = self.bottom_network.chain_full_der_run(output_node_der_iterator, s_run_info.1);
            (ChainOutput::Chain(Box::new(top_chain_ders.chain(bottom_chain_output))),(top_gradient,bottom_info))
        }
        fn chain_der_run(&self,output_node_der_iterator: &mut ChainOutput,s_run_info: Self::SRunInfo) -> Self::Gradient {
            let top_gradient = self.top_network.chain_der_run(output_node_der_iterator, s_run_info.0);
            let bottom_gradient = self.bottom_network.chain_der_run(output_node_der_iterator, s_run_info.1);
            (top_gradient,bottom_gradient)
        }

        fn chain_out_only_run(&self, inputs: &[f64]) -> ChainOutput {
            let top_chain_output = self.top_network.chain_out_only_run(&inputs[0..self.top_network.get_inputs()]);
            let bottom_chain_output = self.bottom_network.chain_out_only_run(&inputs[self.top_network.get_inputs()..inputs.len()]);
            ChainOutput::Chain(Box::new(top_chain_output.chain(bottom_chain_output)))
        }
        fn chain_out_only_special_run(&self, inputs: &[f64]) -> (ChainOutput,Self::SRunInfo) {
            let (top_chain_output,top_info) = self.top_network.chain_out_only_special_run(&inputs[0..self.top_network.get_inputs()]);
            let (bottom_chain_output,bottom_info) = self.bottom_network.chain_out_only_special_run(&inputs[self.top_network.get_inputs()..inputs.len()]);
            (ChainOutput::Chain(Box::new(top_chain_output.chain(bottom_chain_output))),(top_info,bottom_info))
        }
        fn chain_out_only_full_der_run(&self, output_node_ders: &[f64], s_run_info: Self::SRunInfo) -> (ChainOutput,Self::Gradient) {
            let (top_chain_ders,top_gradient) = self.top_network.chain_out_only_full_der_run(&output_node_ders[0..self.top_network.get_outputs()], s_run_info.0);
            let (bottom_chain_output,bottom_info) = self.bottom_network.chain_out_only_full_der_run(&output_node_ders[self.top_network.get_outputs()..output_node_ders.len()], s_run_info.1);
            (ChainOutput::Chain(Box::new(top_chain_ders.chain(bottom_chain_output))),(top_gradient,bottom_info))
        }
    }

    impl<F: SpliceableNetwork+FlattenableGradient,G: SpliceableNetwork+FlattenableGradient> FlattenableGradient for DualParallelNetwork<F,G> {
        fn flatten_gradient(gradient: Self::Gradient) -> Vec<f64> {
            let mut output = <F>::flatten_gradient(gradient.0);
            output.extend(<G>::flatten_gradient(gradient.1));
            output
        }
        fn flatten_ref_gradient(gradient: &Self::Gradient) -> Vec<&f64> {
            let mut output = <F>::flatten_ref_gradient(&gradient.0);
            output.extend(<G>::flatten_ref_gradient(&gradient.1));
            output
        }
        fn add_flat_gradient(&mut self,flat_gradient: &[f64],learning_rate: f64) {
            self.top_network.add_flat_gradient(&flat_gradient[0..self.top_network.get_num_params()], learning_rate);
            self.bottom_network.add_flat_gradient(&flat_gradient[self.top_network.get_num_params()..flat_gradient.len()], learning_rate);
        }
        fn subtract_flat_gradient(&mut self,flat_gradient: &[f64],learning_rate: f64) {
            self.top_network.subtract_flat_gradient(&flat_gradient[0..self.top_network.get_num_params()], learning_rate);
            self.bottom_network.subtract_flat_gradient(&flat_gradient[self.top_network.get_num_params()..flat_gradient.len()], learning_rate);
        }
        fn build_zero_flat_gradient(&self) -> Vec<f64> {
            let mut output = self.top_network.build_zero_flat_gradient();
            output.extend(self.bottom_network.build_zero_flat_gradient());
            output
        }
        fn get_num_params(&self) -> usize {
            self.top_network.get_num_params() + self.bottom_network.get_num_params()
        }
    }
}

#[cfg(test)]

mod tests {
    use super::splice;
    use super::SIGMOID;
    use super::MultiLayerPerceptron;
    use super::NetworkInherentMethods;

    #[test]
    //looks like it generally works
    fn dual_nets_trial_by_fire() {
        let mut network = splice::DualSequentialNetwork{
            left_network: splice::DualParallelNetwork{
                top_network: splice::DualParallelNetwork{
                    top_network: MultiLayerPerceptron::new(vec![1,10,1],SIGMOID,1.0,1.0),
                    bottom_network: MultiLayerPerceptron::new(vec![1,8,1],SIGMOID,1.0,1.0)
                },
                bottom_network: MultiLayerPerceptron::new(vec![1,12,1],SIGMOID,1.0,1.0)
            },
            right_network: splice::DualSequentialNetwork{
                left_network: MultiLayerPerceptron::new(vec![3,20,10],SIGMOID,1.0,1.0),
                right_network: MultiLayerPerceptron::new(vec![10,2,1],SIGMOID,1.0,1.0)
            }
        };
        println!("TEST");
        network.backtracking_line_search_train(&vec![(vec![1.0,0.0,0.5],vec![0.69]),(vec![0.75,0.5,0.25],vec![0.420])], 1.0, 0.5, 0.5, 0.0125, 100)
    }

    #[test]
    //seems to work
    fn composite_network_trial_by_fire() {
        let networks: Vec<Box<dyn splice::ObjectSafeSpliceableNetwork>> = vec![
            Box::new(MultiLayerPerceptron::new(vec![1,10,1],SIGMOID,1.0,1.0)),
            Box::new(MultiLayerPerceptron::new(vec![1,8,2],SIGMOID,1.0,1.0)),
            Box::new(MultiLayerPerceptron::new(vec![1,12,1],SIGMOID,1.0,1.0)),
            Box::new(MultiLayerPerceptron::new(vec![2,20,10],SIGMOID,1.0,1.0)),
            Box::new(MultiLayerPerceptron::new(vec![2,20,10],SIGMOID,1.0,1.0)),
            Box::new(MultiLayerPerceptron::new(vec![20,2,1],SIGMOID,1.0,1.0))
        ];
        let mut network = splice::CompositeNetwork::new(vec![vec![0,1,2],vec![3,4],vec![5]],networks);
        network.backtracking_line_search_train(&vec![(vec![1.0,0.0,0.5],vec![0.69]),(vec![0.75,0.5,0.25],vec![0.420])], 1.0, 0.5, 0.5, 0.0125, 100)
    }

}