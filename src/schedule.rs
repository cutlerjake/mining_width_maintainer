use itertools::iproduct;
use ndarray::Array3;

pub struct Schedule {
    pub sched: Array3<u8>,
    pub max_period: u8,
}

impl Schedule {
    pub fn new(dim: [usize; 3], max_period: u8) -> Self {
        Self {
            sched: Array3::from_elem(dim, max_period),
            max_period,
        }
    }

    pub fn preds(&self, [i, j, k]: [usize; 3]) -> Vec<[usize; 3]> {
        if k == 0 {
            return vec![];
        }

        let k = k - 1;
        let i_rng = i.saturating_sub(1)..num::clamp(i + 2, 0, self.sched.dim().0);
        let j_rng = j.saturating_sub(1)..num::clamp(j + 2, 0, self.sched.dim().1);

        iproduct!(i_rng, j_rng)
            .map(move |(i, j)| [i, j, k])
            .collect()
    }

    pub fn succs(&self, [i, j, k]: [usize; 3]) -> Vec<[usize; 3]> {
        if k == self.sched.dim().2 - 1 {
            return vec![];
        }

        let k = k + 1;
        let i_rng = i.saturating_sub(1)..num::clamp(i + 2, 0, self.sched.dim().0);
        let j_rng = j.saturating_sub(1)..num::clamp(j + 2, 0, self.sched.dim().1);

        iproduct!(i_rng, j_rng)
            .map(move |(i, j)| [i, j, k])
            .collect()
    }
}
