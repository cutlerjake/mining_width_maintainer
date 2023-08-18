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

    // let dim = [20, 20, 10];
    // let mine_life = 10;
    // let mut mining_width_maintainer = MiningWidthMaintainer::new(dim, mine_life);

    // let mut best = mining_width_maintainer.clone();
    // let mut best_value = f32::MIN;

    // let capacity = 100;

    // let mut rng = rand::thread_rng();
    // for cnt in 0..1000 {
    //     let i = rng.gen_range(0..17);
    //     let j = rng.gen_range(0..17);
    //     let k = rng.gen_range(0..10);
    //     let period = rng.gen_range(0..10);
    //     let mining_width = SquareMiningWidth::new([i, j, k], 3);
    //     let curr_mw = mining_width_maintainer.clone();

    //     //println!("Perturbing: {:?}, period: {}", mining_width, period);

    //     mining_width_maintainer.perturb(&mining_width, period);
    //     // let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
    //     // npz.add_array("sched", &mining_width_maintainer.sched.sched)
    //     //     .unwrap();
    //     // npz.finish().unwrap();

    //     // break;

    //     //compute value
    //     let period_mined = mining_width_maintainer.sched.sched.iter().fold(
    //         vec![0; mine_life as usize],
    //         |mut acc, &x| {
    //             if x < mine_life {
    //                 acc[x as usize] += 1;
    //             }
    //             acc
    //         },
    //     );

    //     let cap_penalty = period_mined
    //         .iter()
    //         .filter(|mined| **mined > capacity)
    //         .map(|mined| mined - capacity)
    //         .sum::<usize>() as f32;

    //     let depth = mining_width_maintainer.sched.sched.indexed_iter().fold(
    //         0f32,
    //         |mut acc, (ind, period)| {
    //             if *period == mine_life {
    //                 return acc;
    //             }

    //             acc += ind.2 as f32;
    //             acc
    //             // //compute euclidian distance to (10,10,10)
    //             // let dist = ((ind.0 as i32 - 10).pow(2)
    //             //     + (ind.1 as i32 - 10).pow(2)
    //             //     + (ind.2 as i32 - 20).pow(2));
    //             // acc.min(dist as f32)
    //         },
    //     );

    //     let value = depth - cap_penalty;

    //     // println!("Value: {}", value);
    //     // println!("Dist: {}", dist);
    //     // println!("Cap penalty: {}", cap_penalty);
    //     //break;

    //     if value > best_value {
    //         println!("New best value: {}", value);
    //         best = mining_width_maintainer.clone();
    //         best_value = value;
    //     } else {
    //         mining_width_maintainer = curr_mw;
    //     }
    //     println!("Perturbed {} times", cnt);
    //     // if !mining_width_maintainer.verify_mining_width(3) {
    //     //     let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
    //     //     npz.add_array("sched", &mining_width_maintainer.sched.sched)
    //     //         .unwrap();
    //     //     npz.finish().unwrap();
    //     //     panic!();
    //     // }
    // }

    // let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
    // npz.add_array("sched", &best.sched.sched).unwrap();
    // npz.finish().unwrap();

    let dim = [20, 20, 10];
    let mine_life = 10;
    let mut mining_width_maintainer = MiningWidthMaintainer::new(dim, mine_life);

    let inds = vec![[10, 10, 2], [10, 11, 2], [10, 10, 3]];
    let periods = vec![0, 0, 1];

    for (ind, period) in inds.iter().zip(periods) {
        mining_width_maintainer.perturb(&SquareMiningWidth::new(*ind, 3), period);
    }

    let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
    npz.add_array("sched", &mining_width_maintainer.sched.sched)
        .unwrap();
    npz.finish().unwrap();
}
