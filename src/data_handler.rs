use std::path::Path;

// Imports
use image::{self, DynamicImage, EncodableLayout, GenericImage, ImageBuffer, Luma, Pixel, Rgba};
use imageproc::{definitions::Image, geometric_transformations::{rotate_about_center, translate, warp, Interpolation, Projection}};
use rand::{thread_rng, Rng};

use crate::{layer::Layer, utils::{colorize, Color}};

type Data = Vec<(Vec<f64>, Vec<f64>)>;
pub fn modify_images(data: Data, size: u32) -> Data {
    let mut modified_data: Data = Vec::with_capacity(data.len());
    let mut rng = thread_rng();
    let mut index = 0;

    let total_len = data.len();
    let noise_strength: u32 = 10;

    for (input, expected_output) in data.into_iter() {
        type P = Luma<u8>;
        let image_as_u8 = input.into_iter().map(|e| (e * 256.0) as u8).collect::<Vec<u8>>();
        let img: ImageBuffer<P, Vec<<P as Pixel>::Subpixel>> = image::ImageBuffer::from_vec(size, size, image_as_u8).unwrap();
        
        /* Scale */
        let rand_scale = rng.gen_range(0.925..1.075);
        let transform_extra = ((1.0 - rand_scale) * size as f32 / 2.0) as i32; // Because scaling scales towards top left for some reason
        let warped_image = warp(&img, &Projection::scale(rand_scale, rand_scale), Interpolation::Nearest, Luma([0]));

        /* Rotate */
        let rand_rotation = rng.gen_range(-0.3..0.3);
        let rotated_image = rotate_about_center(&warped_image, rand_rotation, Interpolation::Nearest, Luma([0]));

        /* Translate */
        let (rand_x, rand_y) = (rng.gen_range(-3..3), rng.gen_range(-3..3));
        let mut translated_image = translate(&rotated_image, (rand_x + transform_extra, rand_y + transform_extra));
        
        /* Noise */
        for _ in 0..noise_strength {
            let (x, y) = (rng.gen_range(0..size), rng.gen_range(0..size));
            *translated_image.get_pixel_mut(x, y) = Luma([(255.0 * rng.gen::<f32>()) as u8]);
        }

        let flattened_f64: Vec<f64> = translated_image.pixels().into_iter().flat_map(|e| [e.0[0] as f64 / 255.0]).collect();
        modified_data.push((flattened_f64, expected_output));

        if index % (total_len / 100) == 0 {
            println!("{}", colorize(Color::Yellow, format!("Transformed {:.0}% of data", 100.0 * index as f32 / total_len as f32)));
        }
        index += 1;
    }

    modified_data
}


pub fn print_matrix(matrix: &Vec<f64>, matrix_size: usize) {
    let mut idx = 0;
    for _ in 0..matrix_size {
        let mut a = String::new();

        for _ in 0..matrix_size {
            a.push_str(if matrix[idx] > 0.5 { "##" } else { "__" });

            idx += 1;
        }

        println!("{a}");
    }
}

pub fn load_data(path: &str) -> Data {
    bincode::deserialize::<Data>(&std::fs::read(Path::new(path)).unwrap()).unwrap()
}
pub fn save_data(path: &str, data: Data) -> () {
    std::fs::write(path, &bincode::serialize(&data).unwrap()).unwrap();
}

pub fn save_weight_layer_image(layer: &Layer, path: &str) -> () {
    let mut image: DynamicImage = DynamicImage::new(layer.prev_size() as u32, layer.size() as u32, image::ColorType::Rgba8);
    let max = 300.0;

    for i in 0..layer.prev_size() {
        for j in 0..layer.size() {
            let weight = layer.weight(i, j);
            let bias = layer.bias(j);

            let blue = ((weight * max).min(255.0).max(0.0) as u8).checked_mul(2).unwrap_or(255);
            let red = (bias * max * 0.5).min(255.0).max(0.0) as u8;

            let max = *[blue, red].iter().max().unwrap();

            image.put_pixel(i as u32, j as u32, Rgba([max, 0, max, 255]));
        }
    }


    image.save(path).unwrap();
}

