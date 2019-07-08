use std::iter::Sum;
use std::ops::Deref;

use num_traits::Num;
use sprs::{CsMatBase, CsVecI, CsMatI};
use sprs::SpIndex;

use crate::CsVecBaseExt;

pub trait CsMatBaseExt<N, I> {
    fn outer_sum(&self) -> CsVecI<N, I>;
    fn inner_sum(&self) -> CsVecI<N, I>;

    fn col_sum(&self) -> CsVecI<N, I>;
    fn row_sum(&self) -> CsVecI<N, I>;

    fn outer_avg(&self) ->CsVecI<N, I>;
    fn inner_avg(&self) -> CsVecI<N, I>;

    fn col_avg(&self) -> CsVecI<N, I>;
    fn row_avg(&self) -> CsVecI<N, I>;
}

impl<N, I, IS, DS> CsMatBaseExt<N, I> for CsMatBase<N, I, IS, IS, DS>
    where
        I: SpIndex + From<usize>,
        IS: Deref<Target = [I]>,
        DS: Deref<Target = [N]>,
        N: Num + Copy + Default + Sum {
    fn outer_sum(&self) -> CsVecI<N, I> {
        let mut ind_vec: Vec<I> = Vec::new();
        let mut sum_vec: Vec<N> = Vec::new();
        for (ind, vec) in self.outer_iterator().enumerate() {
            let v = vec.sum();
            if v != N::zero() {
                ind_vec.push( From::from(ind));
                sum_vec.push(v);
            }
        }
        CsVecI::new(self.cols(), ind_vec, sum_vec)
    }

    fn inner_sum(&self) -> CsVecI<N, I> {
        self.to_other_storage().outer_sum()
    }

    fn col_sum(&self) -> CsVecI<N, I> {
        if self.is_csc() {self.outer_sum()} else {self.inner_sum()}
    }

    fn row_sum(&self) -> CsVecI<N, I> {
        if self.is_csr() {self.outer_sum()} else {self.inner_sum()}
    }

    fn outer_avg(&self) -> CsVecI<N, I> {
        let mut ind_vec: Vec<I> = Vec::new();
        let mut avg_vec: Vec<N> = Vec::new();
        for (ind, vec) in self.outer_iterator().enumerate() {
            let v = vec.avg();
            if v != N::zero() {
                ind_vec.push( From::from(ind));
                avg_vec.push(vec.avg());
            }
        }
        CsVecI::new(self.cols(), ind_vec, avg_vec)
    }

    fn inner_avg(&self) -> CsVecI<N, I> {
        self.to_other_storage().outer_avg()
    }

    fn col_avg(&self) -> CsVecI<N, I> {
        if self.is_csc() {self.outer_avg()} else {self.inner_avg()}
    }

    fn row_avg(&self) -> CsVecI<N, I> {
        if self.is_csr() {self.outer_avg()} else {self.inner_avg()}
    }
}

trait CsMatIExt {
    fn outer_center(&self) -> Self;
    fn inner_center(&self) -> Self;
    fn col_center(&self) -> Self;
    fn row_center(&self) -> Self;
}

/*impl<N, I> CsMatIExt for CsMatI<N, I> where
    I: SpIndex + From<usize>,
    N: Num + Copy + Default + Sum {
    fn outer_center(&self) -> Self {

    }
}*/