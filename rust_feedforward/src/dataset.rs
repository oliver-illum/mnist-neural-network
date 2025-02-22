#[allow(unused)]
use image::{GrayImage, Luma};
use mnist::*;
use ndarray::{Array1, Array2};

use flate2::read::GzDecoder;
use reqwest;
use std::{collections::HashSet, fs, io};

#[allow(unused)]
pub struct MNISTData {
    pub training_images: Array2<f32>,
    pub training_labels: Array1<u8>,
    pub test_images: Array2<f32>,
    pub test_labels: Array1<u8>,
}

pub fn load_dataset() -> Option<MNISTData> {
    check_data();
    #[allow(unused)]
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        //.download_and_extract()
        .finalize();

    let training_images = Array2::from_shape_vec((60_000, 28 * 28), trn_img)
        .expect("Error converting training images")
        .mapv(|x| x as f32 / 255.0);
    let test_images = Array2::from_shape_vec((10_000, 28 * 28), tst_img)
        .expect("Error converting test images")
        .mapv(|x| x as f32 / 255.0);

    let training_labels = Array1::from(trn_lbl);
    let test_labels = Array1::from(tst_lbl);

    Some(MNISTData {
        training_images,
        training_labels,
        test_images,
        test_labels,
    })
}

fn check_data() {
    let file_names = [
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    ];
    let folder = "../data";
    let dir_check_list: Result<fs::ReadDir, io::Error> = fs::read_dir(folder);

    if let Ok(dir_list) = dir_check_list {
        let mut files_not_found: Vec<String> = Vec::new();

        let file_entries = dir_list.filter_map(Result::ok).collect::<Vec<_>>();
        let file_list: HashSet<String> = file_entries
            .iter()
            .map(|x| x.file_name().into_string().unwrap())
            .collect();
        let _files_found_len = file_list.len();

        for name in file_names {
            if file_list.contains(name) {
                continue;
            } else {
                files_not_found.push(name.to_string());
            }
        }

        if files_not_found.len() != 0 {
            download_data(files_not_found, folder);
        }
    } else {
        download_data(
            file_names
                .into_iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>(),
            folder,
        );
    }
}

fn download_data(files: Vec<String>, folder: &str) {
    let base_url = "https://ossci-datasets.s3.amazonaws.com/mnist";
    fs::create_dir_all(folder).expect("failed to createa directory");

    for file in files {
        let rqst_url = format!("{}/{}.gz", base_url, file);
        let reponse = reqwest::blocking::get(rqst_url).expect("request failed");

        let bytes = reponse.bytes().expect("failed to get bytes");
        let mut decoder = GzDecoder::new(&bytes[..]);

        let file_path = format!("{}/{}", folder, file);

        let mut out = fs::File::create(file_path).expect("failed to create file");
        io::copy(&mut decoder, &mut out).expect("failed to decompress and copy content");
    }
}

#[allow(unused)]
fn save_image(pixels: &[u8], path: &str) {
    //makes an empty 28 x 28 grayscale image
    let mut img = GrayImage::new(28, 28);
    for y in 0..28 {
        for x in 0..28 {
            let pixel_value = pixels[y * 28 + x];
            img.put_pixel(x as u32, y as u32, Luma([pixel_value]));
        }
    }
    img.save(path).unwrap();
    println!("the image is saved at: {path}");
}
