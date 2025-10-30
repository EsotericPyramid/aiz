use std::fs;
use std::process::Command;

use aiz::*;

fn label_to_vec(label: u8) -> Vec<f64> {
    match label {
        0 => vec![1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        1 => vec![0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        2 => vec![0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        3 => vec![0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
        4 => vec![0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
        5 => vec![0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
        6 => vec![0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],
        7 => vec![0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
        8 => vec![0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
        9 => vec![0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
        _ => panic!("invalid label"),
    }
}

fn vec_to_label(vec: Vec<f64>) -> u8 {
    let mut current_num = 0;
    let mut highest_num = 11;
    let mut highest_num_value = 0.0;
    for current_num_value in vec {
        if current_num_value > highest_num_value {
            highest_num_value = current_num_value.clone();
            highest_num = current_num;
        }
        current_num += 1;
    }
    highest_num
}

fn read_n_parse_dataset() -> (Vec<(Vec<f64>,Vec<f64>)>,Vec<(Vec<f64>,Vec<f64>)>){
    //training labels
    let raw_training_label_file = fs::read("train-labels-idx1-ubyte").expect("failed to read training labels");
    let mut training_labels = Vec::new();
    let mut current_byte_num = 0;
    for label in raw_training_label_file.into_iter() {
        if current_byte_num >= 8 {
            training_labels.push(label_to_vec(label));
        }
        current_byte_num += 1;
    }
    //training images
    let raw_training_image_file = fs::read("train-images-idx3-ubyte").expect("unable to read training images");
    let mut training_images = Vec::new();
    let mut current_image = Vec::new();
    let mut current_byte_num = 0;
    let mut current_image_position = 0;
    for pixel in raw_training_image_file.into_iter() {
        if current_byte_num >= 16 {
            current_image.push(pixel as f64/256.0f64);
            current_image_position += 1;
            if current_image_position == 784 {
                current_image_position = 0;
                training_images.push(current_image);
                current_image = Vec::new();
            }
        }
        current_byte_num += 1;
    }
    let training_data = training_images.into_iter().zip(training_labels.into_iter()).collect::<Vec<(Vec<f64>,Vec<f64>)>>();
    //test labels
    let raw_test_label_file = fs::read("t10k-labels-idx1-ubyte").expect("failed to read test labels");
    let mut test_labels = Vec::new();
    let mut current_byte_num = 0;
    for label in raw_test_label_file.into_iter() {
        if current_byte_num >= 8 {
            test_labels.push(label_to_vec(label));
        }
        current_byte_num += 1;
    }
    //test images
    let raw_test_image_file = fs::read("t10k-images-idx3-ubyte").expect("unable to read test images");
    let mut test_images = Vec::new();
    let mut current_image = Vec::new();
    let mut current_byte_num = 0;
    let mut current_image_position = 0;
    for pixel in raw_test_image_file.into_iter() {
        if current_byte_num >= 16 {
            current_image.push(pixel as f64/256.0f64);
            current_image_position += 1;
            if current_image_position == 784 {
                current_image_position = 0;
                test_images.push(current_image);
                current_image = Vec::new();
            }
        }
        current_byte_num += 1;
    }
    let test_data = test_images.into_iter().zip(test_labels.into_iter()).collect::<Vec<(Vec<f64>,Vec<f64>)>>();
    (training_data,test_data)

}

fn pixel_brightness_to_ascii_char(brightness: &f64) -> char {
    let char_array = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`\'. ".chars().rev().collect::<Vec<char>>();
    let adj_brightness = (brightness * 70.0).floor();
    let output: char;
    if adj_brightness ==  70.0 {
        output = ' ';
    } else {
        output = char_array[adj_brightness as usize];
    }
    output
}

fn print_out_image_in_ascii(flattened_image: &Vec<f64>) {
    let mut column_num = 0;
    let mut output = String::new();
    for pixel in flattened_image {
        output.push(pixel_brightness_to_ascii_char(pixel));
        output.push(' ');
        column_num += 1;
        if column_num % 28 == 0 {
            output.push('\n');
        }
    }
    print!("{}",output);
}

fn main() {
    let mut anti_sleep_thread = Command::new("caffeinate").spawn().expect("Failed to run 'caffeinate'");
    let (training_data,test_data) = read_n_parse_dataset();
    println!("all data processed");

    for i in 0..10 {
        print_out_image_in_ascii(&training_data[i].0);
    }

    let mut nn = MultiLayerPerceptron::new(vec![784,16,16,10], SIGMOID,1.0, 1.0);
    
    nn.multithreaded_backtracking_line_search_train(
        &training_data, 
        8.0, 
        1.0,
        0.5, 
        0.5, 
        0.000001, 
        200
    );
    println!("FINAL TEST: {}", nn.test(&test_data));
    let mut training_count = 0;
    for (train_in,train_out) in training_data.into_iter() {
        if vec_to_label(nn.run(&train_in)) == vec_to_label(train_out) {
            training_count += 1;
        }
    }
    println!("Train Right Classifications: {}",training_count);
    let mut test_count = 0;
    for (test_in,test_out) in test_data.into_iter() {
        if vec_to_label(nn.run(&test_in)) == vec_to_label(test_out) {
            test_count += 1;
        }
    }
    println!("Test Right Classifications: {}",test_count);
    //let bytes = nn.into_bytes();
    //fs::write("MNIST_nn_v3.aiz",&bytes[..]).expect("Failed to write to file");
    anti_sleep_thread.kill().expect("Failed to kill 'caffeinate'");
}
