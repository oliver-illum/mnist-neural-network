mod dataset;
use dataset::MNISTData;

fn main() {
    let MNISTData {
        training_images,
        training_labels,
        test_images,
        test_labels,
    } = dataset::load_dataset().unwrap();
}
