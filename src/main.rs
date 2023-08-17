use std::fs::File;

use ndarray::Array3;
use ndarray_npy::NpzWriter;
use rand::Rng;

mod schedule;
use schedule::Schedule;

mod mining_width;
use mining_width::SquareMiningWidth;

mod mining_width_maintainer;
use mining_width_maintainer::MiningWidthMaintainer;

fn main() {
    // let mut rng = rand::thread_rng();
    // //create a 3D grade array of size 20x20x20 with random values
    // let mut grade = Array3::<f32>::from_shape_fn((20, 20, 20), |_| rng.gen());

    // //create a 3D tonnage array of size 20x20x20 with 1000
    // let mut tonnage = Array3::<f32>::from_elem((20, 20, 20), 1000.0);

    // //create a 3D period array of size 20x20x20 with 0
    // let mut sched = Array3::<usize>::from_elem((20, 20, 20), 0);

    let dim = [20, 20, 20];
    let mine_life = 10;
    let mut mining_width_maintainer = MiningWidthMaintainer::new(dim, mine_life);

    let mut rng = rand::thread_rng();
    for cnt in 0..1 {
        let i = rng.gen_range(0..17);
        let j = rng.gen_range(0..17);
        let k = rng.gen_range(0..20);
        let period = rng.gen_range(0..10);
        let mining_width = SquareMiningWidth::new([i, j, k], 8);
        mining_width_maintainer.perturb(&mining_width, period);
        println!("Perturbed {} times", cnt);
        // if !mining_width_maintainer.verify_mining_width(3) {
        //     let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
        //     npz.add_array("sched", &mining_width_maintainer.sched.sched)
        //         .unwrap();
        //     npz.finish().unwrap();
        //     panic!();
        // }
    }

    let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
    npz.add_array("sched", &mining_width_maintainer.sched.sched)
        .unwrap();
    npz.finish().unwrap();
}
