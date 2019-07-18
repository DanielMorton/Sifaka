use std::iter::Sum;
use std::ops::Deref;

use num_traits::{Float, Num, Signed};
use sprs::{CsVecBase, CsVecI};
use sprs::SpIndex;

use super::CsVecBaseExt;

pub trait CsVecFloat<N, I>: CsVecBaseExt<N, I> {
    fn l2_norm(&self) -> N;
    fn normalize(&self) -> CsVecI<N, I>;
}

impl<N, I, IS, DS> CsVecFloat<N, I> for CsVecBase<IS, DS>
    where  I: SpIndex,
           IS: Deref<Target = [I]>,
           DS: Deref<Target = [N]>,
           N: Num + Sum + Signed + Float {

    fn l2_norm(&self) -> N  {
        self.data_fold(N::zero(), |s, &x| s + x * x).sqrt()
    }

    fn normalize(&self) -> CsVecI<N, I> {
        let norm = self.l2_norm();
        self.map(|x| *x/norm)
    }
}