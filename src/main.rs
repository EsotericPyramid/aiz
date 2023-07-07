use aiz::aiz; //change the structure so I dont need to do this dumb stuff
use std::fs;

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
            current_image.push(pixel as f64/ 256.0f64);
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
            current_image.push(pixel as f64/ 256.0f64);
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

fn main() {
    let (training_data,test_data) = read_n_parse_dataset();
    println!("all data processed");
    let mut best_nn = aiz::NeuralNetwork::new(vec![1,1]);
    let mut best_nn_score = 10000.0;
    for _ in 0..20 {
        let mut nn = aiz::NeuralNetwork::new(vec![784,12,12,10]);
        nn.back_propagation(&training_data, &test_data, (-3.0_f64).exp2(), true);
        let nn_score = nn.test(&test_data);
        if nn_score < best_nn_score {
            println!("{}",nn_score);
            best_nn = nn;
            best_nn_score = nn_score;
        }
    }
    best_nn.back_propagation(&training_data, &test_data, (-20.0_f64).exp2(), false);
    let mut num_correct = 0;
    for example in training_data {
        if vec_to_label(best_nn.run(&example.0)) == vec_to_label(example.1) {
            num_correct += 1;
        }
    }
    println!("{}",num_correct);
    let mut num_correct = 0;
    for example in test_data {
        if vec_to_label(best_nn.run(&example.0)) == vec_to_label(example.1) {
            num_correct += 1;
        }
    }
    println!("{}",num_correct);
}