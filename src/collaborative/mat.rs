use std::iter::Sum;
use std::ops::Deref;

use num_traits::Num;
use sprs::{CsMatBase, CsVecI, CsMatI};
use sprs::SpIndex;

use super::CsVecBaseExt;

pub trait CsMatBaseExt<N, I>
    where I:SpIndex {

    fn ip_vec(&self) -> Vec<I>;
    fn ind_vec(&self) -> Vec<I>;
    fn outer_sum(&self) -> CsVecI<N, I>;
    fn inner_sum(&self) -> CsVecI<N, I>;

    fn col_sum(&self) -> CsVecI<N, I>;
    fn row_sum(&self) -> CsVecI<N, I>;

    fn outer_avg(&self) ->CsVecI<N, I>;
    fn inner_avg(&self) -> CsVecI<N, I>;

    fn col_avg(&self) -> CsVecI<N, I>;
    fn row_avg(&self) -> CsVecI<N, I>;

    fn outer_center(&self) -> CsMatI<N, I>;
    fn inner_center(&self) -> CsMatI<N, I>;

    fn col_center(&self) -> CsMatI<N, I>;
    fn row_center(&self) -> CsMatI<N, I>;
}

impl<N, I, IS, DS> CsMatBaseExt<N, I> for CsMatBase<N, I, IS, IS, DS>
    where
        I: SpIndex + From<usize>,
        IS: Deref<Target = [I]>,
        DS: Deref<Target = [N]>,
        N: Num + Copy + Default + Sum {

    fn ip_vec(&self) -> Vec<I> {
        self.indptr().to_vec()
    }

    fn ind_vec(&self) -> Vec<I> {
        self.indices().to_vec()
    }

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

    fn outer_center(&self) -> CsMatI<N, I> {
        let mut data: Vec<N> = Vec::new();
        for (_, vec) in self.outer_iterator().enumerate() {
            data.append(&mut vec.center().data().to_vec());
        }
        CsMatI::new(self.shape(), self.ip_vec(), self.ind_vec(), data)
    }

    fn inner_center(&self) -> CsMatI<N, I> { self.to_other_storage().outer_center() }

    fn col_center(&self) -> CsMatI<N, I> {
        if self.is_csc() {self.outer_center()} else {self.inner_center()}
    }

    fn row_center(&self) -> CsMatI<N, I> {
        if self.is_csr() {self.outer_center()} else {self.inner_center()}
    }

}