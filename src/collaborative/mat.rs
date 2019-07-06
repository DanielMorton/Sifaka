use sprs::{CsMat, CsVec};
use num_traits::identities::One;
use num_traits::Num;
use std::iter::Sum;
use crate::CsVecExt;

pub trait CsMatExt<N: Num + Copy + Default + Sum> {
    fn outer_sum(&self) -> CsVec<N>;
    fn inner_sum(&self) -> CsVec<N>;

    fn col_sum(&self) -> CsVec<N>;
    fn row_sum(&self) -> CsVec<N>;

   // fn col_avg(&self) -> CsVec<N>;
   // fn row_avg(&self) -> CsVec<N>;
}

impl<N> CsMatExt<N> for CsMat<N> where N: Num + Copy + Default + Sum {
    fn outer_sum(&self) -> CsVec<N> {
        let mut ind_vec: Vec<usize> = Vec::new();
        let mut sum_vec: Vec<N> = Vec::new();
        for (ind, vec) in self.outer_iterator().enumerate() {
            ind_vec.push( ind);
            sum_vec.push(vec.sum());
        }
        CsVec::new(self.cols(), ind_vec, sum_vec)
    }

    fn inner_sum(&self) -> CsVec<N> {
        self.to_other_storage().outer_sum()
    }

    fn col_sum(&self) -> CsVec<N> {
        if self.is_csc() {self.outer_sum()} else {self.inner_sum()}
    }

    fn row_sum(&self) -> CsVec<N> {
        if self.is_csr() {self.outer_sum()} else {self.inner_sum()}
    }
}

