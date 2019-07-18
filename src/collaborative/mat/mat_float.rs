use std::iter::Sum;
use std::ops::Deref;

use num_traits::{Float, Num, Signed};
use sprs::{CsMatBase, CsMatI, CsVecI};
use sprs::SpIndex;

use super::{CsMatBaseExt, CsVecBaseExt, CsVecFloat};

trait CSMatFloatHelp<N, I>: CsMatBaseExt<N, I>
    where I: SpIndex {
    fn outer_l2_norm(&self) -> CsVecI<N, I>;
    fn outer_normalize(&self) -> CsMatI<N, I>;

    fn inner_l2_norm(&self) -> CsVecI<N, I>;
    fn inner_normalize(&self) -> CsMatI<N, I>;
}

impl<N, I, IS, DS> CSMatFloatHelp<N, I> for CsMatBase<N, I, IS, IS, DS>
    where
        I: SpIndex,
        IS: Deref<Target = [I]>,
        DS: Deref<Target = [N]>,
        N: Num + Sum + Clone + Copy + Signed + Default + Float {

    fn outer_l2_norm(&self) -> CsVecI<N, I> {
        let mut ind_vec: Vec<I> = Vec::new();
        let mut norm_vec: Vec<N> = Vec::new();
        for (ind, vec) in self.outer_iterator().enumerate() {
            let v = vec.l2_norm();
            if v != N::zero() {
                ind_vec.push( SpIndex::from_usize(ind));
                norm_vec.push(v);
            }
        }
        CsVecI::new(self.outer_dims(), ind_vec, norm_vec)
    }

    fn outer_normalize(&self) -> CsMatI<N, I> {
        let mut data: Vec<N> = Vec::new();
        for (_, vec) in self.outer_iterator().enumerate() {
            data.append(&mut vec.normalize().data_vec());
        }
        CsMatI::new(self.shape(), self.ip_vec(), self.ind_vec(), data)
    }

    fn inner_l2_norm(&self) -> CsVecI<N, I> {
        self.to_other_storage().outer_l2_norm()
    }

    fn inner_normalize(&self) -> CsMatI<N, I> {
        self.to_other_storage().outer_normalize()
    }
}

pub trait CsMatFloat<N, I>
    where I: SpIndex {

    fn col_l2_norm(&self) -> CsVecI<N, I>;
    fn row_l2_norm(&self) -> CsVecI<N, I>;

    fn col_normalize(&self) -> CsMatI<N, I>;
    fn row_normalize(&self) -> CsMatI<N, I>;
}

impl<N, I, IS, DS> CsMatFloat<N, I> for CsMatBase<N, I, IS, IS, DS>
    where
        I: SpIndex,
        IS: Deref<Target = [I]>,
        DS: Deref<Target = [N]>,
        N: Num + Sum + Clone + Copy + Signed + Default + Float {

    fn col_l2_norm(&self) -> CsVecI<N, I> {
        if self.is_csc() {self.outer_l2_norm()} else {self.inner_l2_norm()}
    }

    fn row_l2_norm(&self) -> CsVecI<N, I> {
        if self.is_csr() {self.outer_l2_norm()} else {self.inner_l2_norm()}
    }

    fn col_normalize(&self) -> CsMatI<N, I> {
        if self.is_csc() {self.outer_normalize()} else {self.inner_normalize()}
    }

    fn row_normalize(&self) -> CsMatI<N, I> {
        if self.is_csr() {self.outer_normalize()} else {self.inner_normalize()}
    }
}