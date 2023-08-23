use itertools::iproduct;
use num::clamp;
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
pub struct SquareMiningWidth {
    pub origin: [usize; 3],
    pub width: usize,
}

impl SquareMiningWidth {
    #[inline(always)]
    pub fn new(origin: [usize; 3], width: usize) -> Self {
        Self { origin, width }
    }

    #[inline(always)]
    pub fn inds(&self, dim: [usize; 3]) -> Vec<[usize; 3]> {
        let mut inds = Vec::new();
        for (i, j) in iproduct!(0..self.width, 0..self.width) {
            let ind = [self.origin[0] + i, self.origin[1] + j, self.origin[2]];
            if ind[0] < dim[0] && ind[1] < dim[1] && ind[2] < dim[2] {
                inds.push(ind);
            }
        }

        inds
    }

    #[inline(always)]
    pub fn bench(&self) -> usize {
        self.origin[2]
    }

    #[inline(always)]
    pub fn gen_all_intersecting(ind: [usize; 3], mining_width: u8, dim: [usize; 3]) -> Vec<Self> {
        let mut mining_widths = Vec::new();
        let i_rng = clamp_range(
            ind[0].saturating_sub(mining_width.saturating_sub(1) as usize)..ind[0] + 1,
            &(0..dim[0] - mining_width as usize + 1),
        );
        let j_rng = clamp_range(
            ind[1].saturating_sub(mining_width.saturating_sub(1) as usize)..ind[1] + 1,
            &(0..dim[1] - mining_width as usize + 1),
        );
        for (i, j) in iproduct!(i_rng, j_rng) {
            mining_widths.push(Self::new([i, j, ind[2]], mining_width as usize));
        }

        mining_widths
    }

    #[inline(always)]
    pub fn neighbors(&self, dim: [usize; 3]) -> Vec<[usize; 3]> {
        fn clamp_range(range: &mut Range<usize>, max: usize) {
            range.start = range.start.min(max);
            range.end = range.end.min(max);
        }
        //create iters of neighboring inds
        let mut i1 = self.origin[0].saturating_sub(self.width.saturating_sub(1))..self.origin[0];
        let mut j1 = self.origin[1].saturating_sub(self.width.saturating_sub(1))
            ..self.origin[1] + 2 * self.width - 1;

        clamp_range(&mut i1, dim[0]);
        clamp_range(&mut j1, dim[1]);

        let mut i2 = self.origin[0] + self.width..self.origin[0] + 2 * self.width - 1;
        let mut j2 = self.origin[1].saturating_sub(self.width.saturating_sub(1))
            ..self.origin[1] + 2 * self.width - 1;

        clamp_range(&mut i2, dim[0]);
        clamp_range(&mut j2, dim[1]);

        let mut i3 = self.origin[0]..self.origin[0] + self.width;
        let mut j3 = self.origin[1].saturating_sub(self.width.saturating_sub(1))..self.origin[1];

        clamp_range(&mut i3, dim[0]);
        clamp_range(&mut j3, dim[1]);

        let mut i4 = self.origin[0]..self.origin[0] + self.width;
        let mut j4 = self.origin[1] + self.width..self.origin[1] + 2 * self.width - 1;

        clamp_range(&mut i4, dim[0]);
        clamp_range(&mut j4, dim[1]);

        iproduct!(i1, j1)
            .chain(iproduct!(i2, j2))
            .chain(iproduct!(i3, j3))
            .chain(iproduct!(i4, j4))
            .map(|(i, j)| [i, j, self.origin[2]])
            .collect()
    }

    #[inline(always)]
    pub fn neighborhood_window(&self, dim: [usize; 3]) -> impl Iterator<Item = [usize; 3]> + '_ {
        let i_bounds = 0..dim[0];
        let j_bounds = 0..dim[1];

        let i_rng = clamp_range(
            self.origin[0].saturating_sub(self.width.saturating_sub(1))
                ..self.origin[0] + 2 * self.width - 1,
            &i_bounds,
        );

        let j_rng = clamp_range(
            self.origin[1].saturating_sub(self.width.saturating_sub(1))
                ..self.origin[1] + 2 * self.width - 1,
            &j_bounds,
        );

        iproduct!(i_rng, j_rng).map(move |(i, j)| [i, j, self.origin[2]])
    }

    #[inline(always)]
    fn perimeter(&self, dim: [usize; 3]) -> Vec<[usize; 3]> {
        let mut inds = Vec::new();

        let i_bounds = 0..dim[0];
        let j_bounds = 0..dim[1];

        let i_top = clamp_range(self.origin[0].saturating_sub(1)..self.origin[0], &i_bounds);
        let i_bottom = clamp_range(
            self.origin[0] + self.width..self.origin[0] + self.width + 1,
            &i_bounds,
        );
        let i_long = clamp_range(self.origin[0]..self.origin[0] + self.width, &i_bounds);

        let j_long = clamp_range(
            self.origin[1].saturating_sub(1)..self.origin[1] + self.width + 1,
            &j_bounds,
        );

        let j_left = clamp_range(self.origin[1].saturating_sub(1)..self.origin[1], &j_bounds);

        let j_right = clamp_range(
            self.origin[1] + self.width..self.origin[1] + self.width + 1,
            &j_bounds,
        );

        for (i, j) in iproduct!(i_top, j_long.clone())
            .chain(iproduct!(i_bottom, j_long))
            .chain(iproduct!(i_long.clone(), j_left))
            .chain(iproduct!(i_long, j_right))
        {
            inds.push([i, j, self.origin[2]]);
        }

        inds
    }

    #[inline(always)]
    pub fn perimeter_of_width(&self, dim: [usize; 3], width: usize) -> Vec<[usize; 3]> {
        let mut inds = Vec::new();

        let i_bounds = 0..dim[0];
        let j_bounds = 0..dim[1];

        let i_top = clamp_range(
            self.origin[0].saturating_sub(width)..self.origin[0],
            &i_bounds,
        );
        let i_bottom = clamp_range(
            self.origin[0] + self.width..self.origin[0] + width + 1,
            &i_bounds,
        );
        let i_long = clamp_range(self.origin[0]..self.origin[0] + self.width, &i_bounds);

        let j_long = clamp_range(
            self.origin[1].saturating_sub(width)..self.origin[1] + self.width + width,
            &j_bounds,
        );

        let j_left = clamp_range(
            self.origin[1].saturating_sub(width)..self.origin[1],
            &j_bounds,
        );

        let j_right = clamp_range(
            self.origin[1] + self.width..self.origin[1] + 1 + width,
            &j_bounds,
        );

        for (i, j) in iproduct!(i_top, j_long.clone())
            .chain(iproduct!(i_bottom, j_long))
            .chain(iproduct!(i_long.clone(), j_left))
            .chain(iproduct!(i_long, j_right))
        {
            inds.push([i, j, self.origin[2]]);
        }

        inds
    }
}

fn clamp_range(range: Range<usize>, range_bounds: &Range<usize>) -> Range<usize> {
    let start = range.start.max(range_bounds.start);
    let end = range.end.min(range_bounds.end);
    start..end
}

#[cfg(test)]
mod tests {
    use ndarray::{s, Array, Array3};

    use super::*;

    #[test]
    fn test_inds_full() {
        let width = 3;
        let origin = [0, 0, 0];
        let dim = [10, 10, 10];
        let square_mining_width = SquareMiningWidth::new(origin, width);
        let inds = square_mining_width.inds(dim);
        assert_eq!(inds.len(), 9);
    }

    #[test]
    fn test_inds_cut() {
        let width = 3;
        let origin = [8, 8, 8];
        let dim = [10, 10, 10];
        let square_mining_width = SquareMiningWidth::new(origin, width);
        let inds = square_mining_width.inds(dim);
        assert_eq!(inds.len(), 4);
    }

    #[test]
    fn test_intersecting() {
        let ind = [5, 5, 5];
        let mining_width = 3;
        let mining_widths =
            SquareMiningWidth::gen_all_intersecting(ind, mining_width, [10, 10, 10]);
        assert_eq!(
            mining_widths
                .iter()
                .all(|mw| mw.inds([10, 10, 10]).contains(&[5, 5, 5])),
            true
        );
    }

    #[test]
    fn test_neighbors() {
        let width = 4;
        let origin = [5, 5, 5];
        let dim = [15, 15, 15];
        let square_mining_width = SquareMiningWidth::new(origin, width);
        let neighbors = square_mining_width.neighbors(dim);
        assert_eq!(
            neighbors,
            vec![
                [2, 2, 5],
                [2, 3, 5],
                [2, 4, 5],
                [2, 5, 5],
                [2, 6, 5],
                [2, 7, 5],
                [2, 8, 5],
                [2, 9, 5],
                [2, 10, 5],
                [2, 11, 5],
                [3, 2, 5],
                [3, 3, 5],
                [3, 4, 5],
                [3, 5, 5],
                [3, 6, 5],
                [3, 7, 5],
                [3, 8, 5],
                [3, 9, 5],
                [3, 10, 5],
                [3, 11, 5],
                [4, 2, 5],
                [4, 3, 5],
                [4, 4, 5],
                [4, 5, 5],
                [4, 6, 5],
                [4, 7, 5],
                [4, 8, 5],
                [4, 9, 5],
                [4, 10, 5],
                [4, 11, 5],
                [9, 2, 5],
                [9, 3, 5],
                [9, 4, 5],
                [9, 5, 5],
                [9, 6, 5],
                [9, 7, 5],
                [9, 8, 5],
                [9, 9, 5],
                [9, 10, 5],
                [9, 11, 5],
                [10, 2, 5],
                [10, 3, 5],
                [10, 4, 5],
                [10, 5, 5],
                [10, 6, 5],
                [10, 7, 5],
                [10, 8, 5],
                [10, 9, 5],
                [10, 10, 5],
                [10, 11, 5],
                [11, 2, 5],
                [11, 3, 5],
                [11, 4, 5],
                [11, 5, 5],
                [11, 6, 5],
                [11, 7, 5],
                [11, 8, 5],
                [11, 9, 5],
                [11, 10, 5],
                [11, 11, 5],
                [5, 2, 5],
                [5, 3, 5],
                [5, 4, 5],
                [6, 2, 5],
                [6, 3, 5],
                [6, 4, 5],
                [7, 2, 5],
                [7, 3, 5],
                [7, 4, 5],
                [8, 2, 5],
                [8, 3, 5],
                [8, 4, 5],
                [5, 9, 5],
                [5, 10, 5],
                [5, 11, 5],
                [6, 9, 5],
                [6, 10, 5],
                [6, 11, 5],
                [7, 9, 5],
                [7, 10, 5],
                [7, 11, 5],
                [8, 9, 5],
                [8, 10, 5],
                [8, 11, 5]
            ]
        );
    }

    #[test]
    fn neighbors_on_boundaries() {
        let width = 4;
        let origin = [10, 10, 10];
        let dim = [15, 15, 15];
        let square_mining_width = SquareMiningWidth::new(origin, width);
        let neighbors = square_mining_width.neighbors(dim);

        let mut arr = Array3::from_elem(dim, 0);

        square_mining_width.inds(dim).iter().for_each(|ind| {
            arr[[ind[0], ind[1], ind[2]]] = 1;
        });
        neighbors.iter().for_each(|ind| {
            arr[[ind[0], ind[1], ind[2]]] = 2;
        });
        println!("{:?}", arr.slice(s![.., .., 10]));
        println!("{:?}", neighbors);
    }

    #[test]
    fn test_perimeter() {
        let width = 1;
        let origin = [5, 5, 5];
        let dim = [15, 15, 15];
        let square_mining_width = SquareMiningWidth::new(origin, width);
        let neighbors = square_mining_width.perimeter(dim);

        assert_eq!(
            neighbors,
            vec![
                [4, 4, 5],
                [4, 5, 5],
                [4, 6, 5],
                [6, 4, 5],
                [6, 5, 5],
                [6, 6, 5],
                [5, 4, 5],
                [5, 6, 5]
            ]
        )
    }

    #[test]
    fn test_wide_perimeter() {
        let width = 3;
        let origin = [5, 5, 5];
        let dim = [15, 15, 15];
        let square_mining_width = SquareMiningWidth::new(origin, 1);
        let neighbors = square_mining_width.perimeter_of_width(dim, width);

        let mut arr = Array3::from_elem(dim, 0);
        neighbors.iter().for_each(|ind| {
            arr[[ind[0], ind[1], ind[2]]] = 2;
        });
        println!("{:?}", arr.slice(s![.., .., 5]));

        assert_eq!(
            neighbors,
            vec![
                [4, 4, 5],
                [4, 5, 5],
                [4, 6, 5],
                [6, 4, 5],
                [6, 5, 5],
                [6, 6, 5],
                [5, 4, 5],
                [5, 6, 5]
            ]
        )
    }
}
