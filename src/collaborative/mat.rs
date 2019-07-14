use std::iter::Sum;
use std::ops::Deref;

use num_traits::{Float, Num};
use num_traits::real::Real;
use sprs::{CsMatBase, CsMatI, CsVecI};
use sprs::SpIndex;

use super::{CsFloatVec, CsVecBaseExt};

pub trait CsMatBaseExt<N, I>
    where I:SpIndex {

    fn ip_vec(&self) -> Vec<I>;
    fn ind_vec(&self) -> Vec<I>;
    fn data_vec(&self) -> Vec<N>;

    fn outer_sum(&self) -> CsVecI<N, I>;
    fn outer_avg(&self) ->CsVecI<N, I>;
    fn outer_center(&self) -> CsMatI<N, I>;
    fn outer_l1_norm(&self) -> CsVecI<N, I>;

    fn inner_sum(&self) -> CsVecI<N, I>;
    fn inner_avg(&self) -> CsVecI<N, I>;
    fn inner_center(&self) -> CsMatI<N, I>;
    fn inner_l1_norm(&self) -> CsVecI<N, I>;

    fn col_sum(&self) -> CsVecI<N, I>;
    fn row_sum(&self) -> CsVecI<N, I>;

    fn col_avg(&self) -> CsVecI<N, I>;
    fn row_avg(&self) -> CsVecI<N, I>;

    fn col_center(&self) -> CsMatI<N, I>;
    fn row_center(&self) -> CsMatI<N, I>;

    fn col_l1_norm(&self) -> CsVecI<N, I>;
    fn row_l1_norm(&self) -> CsVecI<N, I>;
}

impl<N, I, IS, DS> CsMatBaseExt<N, I> for CsMatBase<N, I, IS, IS, DS>
    where
        I: SpIndex + From<usize>,
        IS: Deref<Target = [I]>,
        DS: Deref<Target = [N]>,
        N: Num + Copy + Default + Sum + Real {

    fn ip_vec(&self) -> Vec<I> {
        self.indptr().to_vec()
    }

    fn ind_vec(&self) -> Vec<I> {
        self.indices().to_vec()
    }

    fn data_vec(&self) -> Vec<N> {
        self.data().to_vec()
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

    fn outer_avg(&self) -> CsVecI<N, I> {
        let mut ind_vec: Vec<I> = Vec::new();
        let mut avg_vec: Vec<N> = Vec::new();
        for (ind, vec) in self.outer_iterator().enumerate() {
            let v = vec.avg();
            if v != N::zero() {
                ind_vec.push( From::from(ind));
                avg_vec.push(v);
            }
        }
        CsVecI::new(self.cols(), ind_vec, avg_vec)
    }

    fn outer_center(&self) -> CsMatI<N, I> {
        let mut data: Vec<N> = Vec::new();
        for (_, vec) in self.outer_iterator().enumerate() {
            data.append(&mut vec.center().data_vec());
        }
        CsMatI::new(self.shape(), self.ip_vec(), self.ind_vec(), data)
    }

    fn outer_l1_norm(&self) -> CsVecI<N, I> {
        let mut ind_vec: Vec<I> = Vec::new();
        let mut avg_vec: Vec<N> = Vec::new();
        for (ind, vec) in self.outer_iterator().enumerate() {
            let v = vec.l1_norm();
            if v != N::zero() {
                ind_vec.push( From::from(ind));
                avg_vec.push(v);
            }
        }
        CsVecI::new(self.cols(), ind_vec, avg_vec)
    }

    fn inner_sum(&self) -> CsVecI<N, I> { self.to_other_storage().outer_sum() }

    fn inner_avg(&self) -> CsVecI<N, I> { self.to_other_storage().outer_avg() }

    fn inner_center(&self) -> CsMatI<N, I> { self.to_other_storage().outer_center() }

    fn inner_l1_norm(&self) -> CsVecI<N, I> { self.to_other_storage().outer_l1_norm() }

    fn col_sum(&self) -> CsVecI<N, I> {
        if self.is_csc() {self.outer_sum()} else {self.inner_sum()}
    }

    fn row_sum(&self) -> CsVecI<N, I> {
        if self.is_csr() {self.outer_sum()} else {self.inner_sum()}
    }

    fn col_avg(&self) -> CsVecI<N, I> {
        if self.is_csc() {self.outer_avg()} else {self.inner_avg()}
    }

    fn row_avg(&self) -> CsVecI<N, I> {
        if self.is_csr() {self.outer_avg()} else {self.inner_avg()}
    }

    fn col_center(&self) -> CsMatI<N, I> {
        if self.is_csc() {self.outer_center()} else {self.inner_center()}
    }

    fn row_center(&self) -> CsMatI<N, I> {
        if self.is_csr() {self.outer_center()} else {self.inner_center()}
    }

    fn col_l1_norm(&self) -> CsVecI<N, I> {
        if self.is_csc() {self.outer_l1_norm()} else {self.inner_l1_norm()}
    }

    fn row_l1_norm(&self) -> CsVecI<N, I> {
        if self.is_csr() {self.outer_l1_norm()} else {self.inner_l1_norm()}
    }

}

pub trait CsFloatMat<N, I> {
    fn outer_norm(&self) -> CsVecI<N, I>;
    fn inner_norm(&self) -> CsVecI<N, I>;

    fn col_norm(&self) -> CsVecI<N, I>;
    fn row_norm(&self) -> CsVecI<N, I>;
}

impl<N, I, IS, DS> CsFloatMat<N, I> for CsMatBase<N, I, IS, IS, DS>
    where
        I: SpIndex + From<usize>,
        IS: Deref<Target = [I]>,
        DS: Deref<Target = [N]>,
        N: Num + Copy + Default + Sum + Float {

    fn outer_norm(&self) -> CsVecI<N, I> {
        let mut ind_vec: Vec<I> = Vec::new();
        let mut norm_vec: Vec<N> = Vec::new();
        for (ind, vec) in self.outer_iterator().enumerate() {
            let v = vec.l2_norm();
            if v != N::zero() {
                ind_vec.push( From::from(ind));
                norm_vec.push(v);
            }
        }
        CsVecI::new(self.cols(), ind_vec, norm_vec)
    }

    fn inner_norm(&self) -> CsVecI<N, I> {
        self.to_other_storage().outer_norm()
    }

    fn col_norm(&self) -> CsVecI<N, I> {
        if self.is_csc() {self.outer_norm()} else {self.inner_norm()}
    }

    fn row_norm(&self) -> CsVecI<N, I> {
        if self.is_csr() {self.outer_norm()} else {self.inner_norm()}
    }
}