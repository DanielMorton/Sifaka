use sprs::{CsMat, CsVec};
use num_traits::identities::One;
use num_traits::Num;
use std::iter::Sum;

pub trait CsMatExt<N: Num + Copy + Default + Sum> {
    fn col_sum(&self) -> CsVec<N>;
    fn row_sum(&self) -> CsVec<N>;

   // fn col_avg(&self) -> CsVec<N>;
   // fn row_avg(&self) -> CsVec<N>;

    fn one_vec(size: usize) -> CsVec<N> {
        let mut ind_vec: Vec<usize> = Vec::with_capacity(size);
        let one_vec = vec![One::one(); size];
        for i in 0..size {
            ind_vec.push(i);
        }
        return CsVec::new(size, ind_vec, one_vec);
    }
}

impl<N> CsMatExt<N> for CsMat<N> where N: Num + Copy + Default + Sum {
    fn col_sum(&self) -> CsVec<N> {
        let cols = self.cols();
        let col_vec = Self::one_vec(cols);
        self * &col_vec
    }

    fn row_sum(&self) -> CsVec<N> {
        let rows = self.cols();
        let row_vec = Self::one_vec(rows);
        &row_vec * self
    }

}

