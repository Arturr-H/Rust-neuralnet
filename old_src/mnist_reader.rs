// THANKS ALOT https://github.com/busyboredom/rust-mnist

use std::convert::TryFrom;
use std::fs;
use std::io;
use std::io::Read;

use rand::seq::SliceRandom;
use rand::thread_rng;

// Filenames
const TRAIN_DATA_FILENAME: &str = "train-images.idx3-ubyte";
const TEST_DATA_FILENAME: &str = "t10k-images.idx3-ubyte";
const TRAIN_LABEL_FILENAME: &str = "train-labels.idx1-ubyte";
const TEST_LABEL_FILENAME: &str = "t10k-labels.idx1-ubyte";

// Constants relating to the MNIST dataset. All usize for array/vec indexing.
const IMAGES_MAGIC_NUMBER: usize = 2051;
const LABELS_MAGIC_NUMBER: usize = 2049;
const NUM_TRAIN_IMAGES: usize = 60_000;
const NUM_TEST_IMAGES: usize = 10_000;
const IMAGE_ROWS: usize = 28;
const IMAGE_COLUMNS: usize = 28;

pub struct Mnist {
    // Arrays of images.
    pub train_data: Vec<[u8; IMAGE_ROWS * IMAGE_COLUMNS]>,
    pub test_data: Vec<[u8; IMAGE_ROWS * IMAGE_COLUMNS]>,

    // Arrays of labels.
    pub train_labels: Vec<u8>,
    pub test_labels: Vec<u8>,
}

impl Mnist {
    /// Load MNIST dataset.
    ///
    /// # Panics
    ///
    /// Panics if the MNIST dataset is not present at the specified path, or if the dataset is
    /// malformed.
    #[must_use]
    pub fn new(mnist_path: &str) -> Mnist {
        // Get Training Data.
        let train_data = parse_images(&[mnist_path, TRAIN_DATA_FILENAME].concat()).expect(
            &format!(
                "Training data file \"{}{}\" not found; did you \
                     remember to download and extract it?",
                mnist_path, TRAIN_DATA_FILENAME,
            )[..],
        );

        // Assert that numbers extracted from the file were as expected.
        assert_eq!(
            train_data.magic_number, IMAGES_MAGIC_NUMBER,
            "Magic number for training data does not match expected value."
        );
        assert_eq!(
            train_data.num_images, NUM_TRAIN_IMAGES,
            "Number of images in training data does not match expected value."
        );
        assert_eq!(
            train_data.num_rows, IMAGE_ROWS,
            "Number of rows per image in training data does not match expected value."
        );
        assert_eq!(
            train_data.num_cols, IMAGE_COLUMNS,
            "Number of columns per image in training data does not match expected value."
        );

        // Get Testing Data.
        let test_data = parse_images(&[mnist_path, TEST_DATA_FILENAME].concat()).expect(
            &format!(
                "Test data file \"{}{}\" not found; did you \
                     remember to download and extract it?",
                mnist_path, TEST_DATA_FILENAME,
            )[..],
        );

        // Assert that numbers extracted from the file were as expected.
        assert_eq!(
            test_data.magic_number, IMAGES_MAGIC_NUMBER,
            "Magic number for testing data does not match expected value."
        );
        assert_eq!(
            test_data.num_images, NUM_TEST_IMAGES,
            "Number of images in testing data does not match expected value."
        );
        assert_eq!(
            test_data.num_rows, IMAGE_ROWS,
            "Number of rows per image in testing data does not match expected value."
        );
        assert_eq!(
            test_data.num_cols, IMAGE_COLUMNS,
            "Number of columns per image in testing data does not match expected value."
        );

        // Get Training Labels.
        let (magic_number, num_labels, train_labels) =
            parse_labels(&[mnist_path, TRAIN_LABEL_FILENAME].concat()).expect(
                &format!(
                    "Training label file \"{}{}\" not found; did you \
                     remember to download and extract it?",
                    mnist_path, TRAIN_LABEL_FILENAME,
                )[..],
            );

        // Assert that numbers extracted from the file were as expected.
        assert_eq!(
            magic_number, LABELS_MAGIC_NUMBER,
            "Magic number for training labels does not match expected value."
        );
        assert_eq!(
            num_labels, NUM_TRAIN_IMAGES,
            "Number of labels in training labels does not match expected value."
        );

        // Get Testing Labels.
        let (magic_number, num_labels, test_labels) =
            parse_labels(&[mnist_path, TEST_LABEL_FILENAME].concat()).expect(
                &format!(
                    "Test labels file \"{}{}\" not found; did you \
                     remember to download and extract it?",
                    mnist_path, TEST_LABEL_FILENAME,
                )[..],
            );

        // Assert that numbers extracted from the file were as expected.
        assert_eq!(
            magic_number, LABELS_MAGIC_NUMBER,
            "Magic number for testing labels does not match expected value."
        );
        assert_eq!(
            num_labels, NUM_TEST_IMAGES,
            "Number of labels in testing labels does not match expected value."
        );

        Mnist {
            train_data: train_data.images,
            test_data: test_data.images,
            train_labels,
            test_labels,
        }
    }
}

/// Print a sample image.
///
/// # Examples
/// ```
/// use rust_mnist::{print_image, Mnist};
///
/// let mnist = Mnist::new("examples/MNIST_data/");
///
/// // Print one image (the one at index 5).
/// print_image(&mnist.train_data[5], mnist.train_labels[5]);
/// ```
pub fn print_image(image: &[u8; IMAGE_ROWS * IMAGE_COLUMNS], label: u8) {
    println!("Sample image label: {label} \nSample image:");

    // Print each row.
    for row in 0..IMAGE_ROWS {
        for col in 0..IMAGE_COLUMNS {
            if image[row * IMAGE_COLUMNS + col] == 0 {
                print!("__");
            } else {
                print!("##");
            }
        }
        println!();
    }
}

struct MnistImages {
    magic_number: usize,
    num_images: usize,
    num_rows: usize,
    num_cols: usize,
    images: Vec<[u8; IMAGE_ROWS * IMAGE_COLUMNS]>,
}

fn parse_images(filename: &str) -> io::Result<MnistImages> {
    // Open the file.
    let images_data_bytes = fs::File::open(filename)?;
    let images_data_bytes = io::BufReader::new(images_data_bytes);
    let mut buffer_32: [u8; 4] = [0; 4];

    // Get the magic number.
    images_data_bytes
        .get_ref()
        .take(4)
        .read_exact(&mut buffer_32)?;
    let magic_number = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Get number of images.
    images_data_bytes
        .get_ref()
        .take(4)
        .read_exact(&mut buffer_32)?;
    let num_images = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Get number or rows per image.
    images_data_bytes
        .get_ref()
        .take(4)
        .read_exact(&mut buffer_32)?;
    let num_rows = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Get number or columns per image.
    images_data_bytes
        .get_ref()
        .take(4)
        .read_exact(&mut buffer_32)?;
    let num_cols = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Buffer for holding image pixels.
    let mut image_buffer: [u8; IMAGE_ROWS * IMAGE_COLUMNS] = [0; IMAGE_ROWS * IMAGE_COLUMNS];

    // Vector to hold all images in the file.
    let mut images: Vec<[u8; IMAGE_ROWS * IMAGE_COLUMNS]> = Vec::with_capacity(num_images);

    // Get images from file.
    for _image in 0..num_images {
        images_data_bytes
            .get_ref()
            .take(u64::try_from(num_rows * num_cols).unwrap())
            .read_exact(&mut image_buffer)
            .unwrap();
        images.push(image_buffer);
    }

    Ok(MnistImages {
        magic_number,
        num_images,
        num_rows,
        num_cols,
        images,
    })
}

fn parse_labels(filename: &str) -> io::Result<(usize, usize, Vec<u8>)> {
    let labels_data_bytes = fs::File::open(filename)?;
    let labels_data_bytes = io::BufReader::new(labels_data_bytes);
    let mut buffer_32: [u8; 4] = [0; 4];

    // Get the magic number.
    labels_data_bytes
        .get_ref()
        .take(4)
        .read_exact(&mut buffer_32)
        .unwrap();
    let magic_number = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Get number of labels.
    labels_data_bytes
        .get_ref()
        .take(4)
        .read_exact(&mut buffer_32)
        .unwrap();
    let num_labels = usize::try_from(u32::from_be_bytes(buffer_32)).unwrap();

    // Buffer for holding image label.
    let mut label_buffer: [u8; 1] = [0; 1];

    // Vector to hold all labels in the file.
    let mut labels: Vec<u8> = Vec::with_capacity(num_labels);

    // Get labels from file.
    for _label in 0..num_labels {
        labels_data_bytes
            .get_ref()
            .take(1)
            .read_exact(&mut label_buffer)
            .unwrap();
        labels.push(label_buffer[0]);
    }
    Ok((magic_number, num_labels, labels))
}

/// Returns (data, labels) but formatted
pub fn format_data<'a>(data: &'a Vec<[u8; 784]>, labels: &'a Vec<u8>) -> Vec<(Vec<f64>, Vec<f64>)> {
    // Map all u8 labels to vectors looking like this:
    // [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (representing 2)
    let map_labels = |a: u8| {
        let mut vec = vec![0.0f64; 10];
        vec[a as usize] = 1.0f64;
        vec
    };
    let map_data = |a: &[u8; 784]| {
        return a.iter().map(|e| *e as f64).collect::<Vec<f64>>();
    };

    let _labels: Vec<Vec<f64>> = labels.iter().map(|e| map_labels(*e)).collect();
    let _data: Vec<Vec<f64>> = data.iter().map(map_data).collect();

    let mut data: Vec<(Vec<f64>, Vec<f64>)> = _data.into_iter().zip(_labels).collect();
    data.shuffle(&mut thread_rng());

    data
}
