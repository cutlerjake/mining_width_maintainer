use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use ndarray::{s, Array3};
use ndarray_npy::NpzWriter;
use std::fs::File;

use crate::{mining_width::SquareMiningWidth, schedule::Schedule};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockPerturbType {
    Advanced([usize; 3], u8),
    Delayed([usize; 3], u8),
    Unchanged([usize; 3]),
}

impl BlockPerturbType {
    pub fn new(origin: [usize; 3], current_period: u8, target_period: u8) -> Self {
        match current_period.cmp(&target_period) {
            std::cmp::Ordering::Less => Self::Delayed(origin, target_period - current_period),
            std::cmp::Ordering::Equal => Self::Unchanged(origin),
            std::cmp::Ordering::Greater => Self::Advanced(origin, current_period - target_period),
        }
    }
    pub fn origin(&self) -> [usize; 3] {
        match self {
            Self::Advanced(origin, _) => *origin,
            Self::Delayed(origin, _) => *origin,
            Self::Unchanged(origin) => *origin,
        }
    }

    pub fn original_period(&self, curr_period: u8) -> u8 {
        match self {
            Self::Advanced(_, adv) => curr_period + adv,
            Self::Delayed(_, del) => curr_period - del,
            Self::Unchanged(_) => curr_period,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighbordMask {
    mask: Array3<usize>,
    id: usize,
}

impl NeighbordMask {
    fn new(dim: [usize; 3]) -> Self {
        Self {
            mask: Array3::from_elem(dim, 0),
            id: 0,
        }
    }

    fn reset(&mut self) {
        self.id += 1;
    }

    fn visit(&mut self, ind: [usize; 3]) {
        self.mask[ind] = self.id;
    }

    fn is_visited(&self, ind: [usize; 3]) -> bool {
        self.mask[ind] == self.id
    }

    fn curr_id(&self) -> usize {
        self.id
    }

    fn mask(&self) -> &Array3<usize> {
        &self.mask
    }
}

#[derive(Debug, Clone)]
pub struct MiningWidthMaintainer {
    pub(crate) sched: Schedule,
    pub(crate) neighbor_mask: NeighbordMask,
    pub(crate) neighbor_ind_buffer: Vec<[usize; 3]>,
}

impl MiningWidthMaintainer {
    pub fn new(dim: [usize; 3], mine_life: u8) -> Self {
        Self {
            sched: Schedule::new(dim, mine_life),
            neighbor_mask: NeighbordMask::new(dim),
            neighbor_ind_buffer: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn all_periods_same(&self, mining_width: &SquareMiningWidth) -> bool {
        mining_width
            .inds(self.sched.sched.dim().into())
            .iter()
            .all(|ind| self.sched.sched[*ind] == self.sched.sched[mining_width.origin])
    }

    #[inline(always)]
    fn set_internal_inds(
        &mut self,
        mining_width: &SquareMiningWidth,
        period: u8,
    ) -> Vec<BlockPerturbType> {
        let mut perturbed_inds = Vec::new();
        mining_width
            .inds(self.sched.sched.dim().into())
            .iter()
            .for_each(|ind| {
                perturbed_inds.push(BlockPerturbType::new(*ind, self.sched.sched[*ind], period));
                self.sched.sched[*ind] = period;
            });

        perturbed_inds
    }

    fn best_covering_width(
        &self,
        ind: [usize; 3],
        mining_width: u8,
        period: u8,
    ) -> (SquareMiningWidth, usize) {
        //println!("\n ind: {:?}", ind);
        //if neighbor is not visited, then check if it is invalid
        let covering_widths = SquareMiningWidth::gen_all_intersecting(
            ind,
            mining_width,
            self.sched.sched.dim().into(),
        );

        return covering_widths
            .into_iter()
            .fold_while(
                (
                    SquareMiningWidth::new(ind, mining_width as usize),
                    usize::MAX,
                ),
                |(best, best_count), cw| {
                    if self.all_periods_same(&cw) {
                        return Done((cw, 0));
                    }
                    let count = cw
                        .inds(self.sched.sched.dim().into())
                        .into_iter()
                        .filter(|ind| self.sched.sched[*ind] != period)
                        .count();

                    //println!("Count: {}", count);

                    if count < best_count {
                        if count == 0 {
                            return Done((cw, count));
                        }
                        return Continue((cw, count));
                    } else {
                        return Continue((best, best_count));
                    }
                },
            )
            .into_inner();
    }

    fn fix_neighbors(
        &mut self,
        mining_width: &SquareMiningWidth,
        period: u8,
    ) -> Vec<BlockPerturbType> {
        //reset neighbor mask
        self.neighbor_mask.reset();

        //set internal nodes as visited
        mining_width
            .inds(self.sched.sched.dim().into())
            .iter()
            .for_each(|ind| self.neighbor_mask.visit(*ind));

        //get original set of neighbors
        let mut neighbors = mining_width.neighbors(self.sched.sched.dim().into());

        self._fix_neighbors(&mut neighbors, mining_width.width as u8, period)
    }

    fn fix_neighbors_no_mask_reset(
        &mut self,
        mining_width: &SquareMiningWidth,
        period: u8,
    ) -> Vec<BlockPerturbType> {
        //set internal nodes as visited
        mining_width
            .inds(self.sched.sched.dim().into())
            .iter()
            .for_each(|ind| self.neighbor_mask.visit(*ind));

        //get original set of neighbors
        let mut neighbors = mining_width.neighbors(self.sched.sched.dim().into());

        self._fix_neighbors(&mut neighbors, mining_width.width as u8, period)
    }

    fn _fix_neighbors(
        &mut self,
        neighbors: &mut Vec<[usize; 3]>,
        mining_width: u8,
        period: u8,
    ) -> Vec<BlockPerturbType> {
        unsafe {
            self.neighbor_ind_buffer.set_len(0);
        }

        let mut i = 0;
        let mut perturbed = Vec::with_capacity(10000);
        while i < neighbors.len() {
            let neighbor = neighbors[i];
            //if neighbor is already visited, then it is valid so skip
            if self.neighbor_mask.is_visited(neighbor) || self.sched.sched[neighbor] == period {
                i += 1;
                continue;
            }

            let (best_width, delta) = self.best_covering_width(neighbor, mining_width, period);

            if delta == 0 {
                i += 1;
                continue;
            }

            //set neighbor to current period
            let mut perturbed_inds = self.set_internal_inds(&best_width, period);
            perturbed_inds.retain(|perturbation| match perturbation {
                BlockPerturbType::Unchanged(_) => false,
                _ => true,
            });

            perturbed.append(&mut perturbed_inds);

            //set as visited
            let mut inds = best_width.inds(self.sched.sched.dim().into());
            inds.iter().for_each(|ind| {
                if !self.neighbor_mask.is_visited(*ind) {
                    self.neighbor_mask.visit(*ind);
                }
            });

            //ADD NEIGHBORS TO LIST
            neighbors.append(&mut best_width.neighbors(self.sched.sched.dim().into()));
            i += 1;
        }

        perturbed
    }

    fn fix_preds(
        &mut self,
        ind: [usize; 3],
        mining_width: u8,
        period: u8,
        curr_mask_id: usize,
    ) -> Vec<BlockPerturbType> {
        //reset neighbor mask
        //self.neighbor_mask.reset();

        //let mut neighbors = Vec::new();

        // get peturbed pred inds
        let mut perturbed = self
            .sched
            .preds(ind)
            .iter()
            .map(|ind| {
                //visit ind
                //self.neighbor_mask.visit([ind[0], ind[1]]);

                if self.sched.sched[*ind] <= period {
                    return BlockPerturbType::Unchanged(*ind);
                }

                let bp = BlockPerturbType::new(*ind, self.sched.sched[*ind], period);

                self.sched.sched[*ind] = period;
                bp
            })
            .collect::<Vec<_>>();

        let mut tmp = Vec::new();
        perturbed.iter().for_each(|pert| {
            if self.neighbor_mask.mask()[pert.origin()] >= curr_mask_id {
                return;
            }

            let (best_width, delta) = self.best_covering_width(pert.origin(), mining_width, period);

            self.set_internal_inds(&best_width, period);
            best_width
                .inds(self.sched.sched.dim().into())
                .iter()
                .for_each(|ind| {
                    self.neighbor_mask.visit(*ind);
                });

            tmp.append(&mut &mut self.fix_neighbors_no_mask_reset(&best_width, period));
        });

        perturbed.append(&mut tmp);

        //perturbed.append(&mut self._fix_neighbors(&mut neighbors, mining_width, period));
        perturbed
    }

    fn fix_succs(
        &mut self,
        ind: [usize; 3],
        mining_width: u8,
        period: u8,
        curr_mask_id: usize,
    ) -> Vec<BlockPerturbType> {
        //reset neighbor mask
        self.neighbor_mask.reset();

        //let mut neighbors = Vec::new();

        // get peturbed pred inds
        let mut perturbed = self
            .sched
            .succs(ind)
            .iter()
            .map(|ind| {
                //visit ind
                //self.neighbor_mask.visit([ind[0], ind[1]]);

                if self.sched.sched[*ind] >= period {
                    return BlockPerturbType::Unchanged(*ind);
                }

                let bp = BlockPerturbType::new(*ind, self.sched.sched[*ind], period);

                self.sched.sched[*ind] = period;
                bp
            })
            .collect::<Vec<_>>();

        let mut tmp = Vec::new();
        perturbed.iter().for_each(|pert| {
            if self.neighbor_mask.mask()[pert.origin()] >= curr_mask_id {
                return;
            }
            let (best_width, delta) = self.best_covering_width(pert.origin(), mining_width, period);

            self.set_internal_inds(&best_width, period);
            best_width
                .inds(self.sched.sched.dim().into())
                .iter()
                .for_each(|ind| {
                    self.neighbor_mask.visit(*ind);
                });

            tmp.append(&mut &mut self.fix_neighbors_no_mask_reset(&best_width, period));
        });

        perturbed.append(&mut tmp);

        //perturbed.append(&mut self._fix_neighbors(&mut neighbors, mining_width, period));
        perturbed
    }

    fn fix_preds_and_succs(
        &mut self,
        peturbed: &mut Vec<BlockPerturbType>,
        mining_width: u8,
        period: u8,
    ) {
        self.neighbor_mask.reset();
        let curr_id = self.neighbor_mask.curr_id();
        let mut i = 0;

        while i < peturbed.len() {
            let pert = peturbed[i];
            match pert {
                BlockPerturbType::Advanced(ind, delta) => {
                    let mut fix_perturbed = self.fix_preds(ind, mining_width, period, curr_id);
                    peturbed.append(&mut fix_perturbed);
                }
                BlockPerturbType::Delayed(ind, delta) => {
                    let mut fix_perturbed = self.fix_succs(ind, mining_width, period, curr_id);
                    peturbed.append(&mut fix_perturbed);
                }
                BlockPerturbType::Unchanged(ind) => {}
            }
            i += 1;
        }
    }

    fn fix_preds_and_succs_no_reset(
        &mut self,
        peturbed: &mut Vec<BlockPerturbType>,
        mining_width: u8,
        period: u8,
        curr_mask_id: usize,
    ) {
        let mut i = 0;

        while i < peturbed.len() {
            let pert = peturbed[i];
            match pert {
                BlockPerturbType::Advanced(ind, delta) => {
                    let mut fix_perturbed = self.fix_preds(ind, mining_width, period, curr_mask_id);
                    peturbed.append(&mut fix_perturbed);
                }
                BlockPerturbType::Delayed(ind, delta) => {
                    let mut fix_perturbed = self.fix_succs(ind, mining_width, period, curr_mask_id);
                    peturbed.append(&mut fix_perturbed);
                }
                BlockPerturbType::Unchanged(ind) => {}
            }
            i += 1;
        }
    }

    pub fn perturb(&mut self, mining_width: &SquareMiningWidth, period: u8) {
        // set period for internal inds
        let mut perturbed_nodes = self.set_internal_inds(mining_width, period);

        // fix neighbors violating mining width
        self.neighbor_mask.reset();
        let curr_id = self.neighbor_mask.curr_id();
        perturbed_nodes.append(&mut &mut self.fix_neighbors_no_mask_reset(mining_width, period));

        //fix preds and succs
        self.fix_preds_and_succs_no_reset(
            &mut perturbed_nodes,
            mining_width.width as u8,
            period,
            curr_id,
        );
    }

    fn verify_mining_width(&self, mining_width: u8) -> bool {
        self.sched.sched.indexed_iter().all(|(ind, period)| {
            //if neighbor is not visited, then check if it is invalid
            let covering_widths = SquareMiningWidth::gen_all_intersecting(
                ind.into(),
                mining_width,
                self.sched.sched.dim().into(),
            );
            let period = self.sched.sched[ind];
            //if neighbor is invalid, then set it to current period
            let best_width = covering_widths
                .iter()
                .max_by_key(|cw| {
                    cw.inds(self.sched.sched.dim().into())
                        .iter()
                        .filter(|ind| self.sched.sched[**ind] == period)
                        .count()
                })
                .expect("Error unwraping best width");

            if !self.all_periods_same(best_width) {
                println!(
                    "Error: mining width violated at ind {:?} with period {}",
                    ind, period,
                );
                return false;
            }

            true
        })
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::mining_width;

    use super::*;

    #[test]
    fn set_internal_inds() {
        let dim = [10, 10, 10];
        let mine_life = 10;
        let mut mining_width_maintainer = MiningWidthMaintainer::new(dim, mine_life);
        let mining_width = SquareMiningWidth::new([0, 0, 0], 3);
        let perturbed_inds = mining_width_maintainer.set_internal_inds(&mining_width, 5);
        assert_eq!(mining_width_maintainer.sched.sched[[0, 0, 0]], 5);
        assert_eq!(mining_width_maintainer.sched.sched[[2, 2, 0]], 5);
    }

    #[test]
    fn fix_neighbors() {
        let dim = [10, 10, 10];
        let mine_life = 10;
        let mut mining_width_maintainer = MiningWidthMaintainer::new(dim, mine_life);
        let mining_width = SquareMiningWidth::new([0, 0, 0], 9);
        let mining_width_2 = SquareMiningWidth::new([4, 2, 0], 3);
        mining_width_maintainer.set_internal_inds(&mining_width, 5);
        mining_width_maintainer.fix_neighbors(&mining_width, 5);

        mining_width_maintainer.set_internal_inds(&mining_width_2, 4);
        let inds = mining_width_maintainer.fix_neighbors(&mining_width_2, 4);

        println!(
            "{:?}",
            mining_width_maintainer.sched.sched.slice(s![.., .., 0])
        );

        println!("Perturbed inds:");
        inds.iter().for_each(|p| println!("\t{:?}", p));
    }

    #[test]
    fn fix_mining_width() {
        let i_size = 100;
        let j_size = 100;
        let k_size = 100;
        let mw = 4;
        let dim = [i_size, j_size, k_size];
        let mine_life = 10;
        let mut mining_width_maintainer = MiningWidthMaintainer::new(dim, mine_life);

        let mut rng = rand::thread_rng();
        for cnt in 0..1000 {
            let i = rng.gen_range(0..i_size - mw);
            let j = rng.gen_range(0..j_size - mw);
            let k = rng.gen_range(0..k_size);
            let period = rng.gen_range(0..10);
            let mining_width = SquareMiningWidth::new([i, j, k], mw);
            mining_width_maintainer.perturb(&mining_width, period);
            println!("Perturbed {} times", cnt);
            if !mining_width_maintainer.verify_mining_width(4) {
                let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
                npz.add_array("sched", &mining_width_maintainer.sched.sched)
                    .unwrap();
                npz.finish().unwrap();
                panic!();
            }
        }

        let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
        npz.add_array("sched", &mining_width_maintainer.sched.sched)
            .unwrap();
        npz.finish().unwrap();
    }
}
