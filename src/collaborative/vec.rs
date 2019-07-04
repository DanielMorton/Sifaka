use sprs::CsVec;
use num_traits::identities::One;
use num_traits::Num;
use std::iter::Sum;

pub trait CsVecExt<N: Num + Copy + Default + Sum> {
    fn sum(&self) -> N;
    fn length(&self) -> usize;
    fn avg(&self) -> N;
}

impl<N> CsVecExt<N> for CsVec<N> where N: Num + Copy + Default + Sum  {

    fn sum(&self) -> N {
        self.data().iter().map(|x| *x).sum()
    }

    fn length(&self) -> usize {
        self.indices().len()
    }

    fn avg(&self) -> N {
        self.sum()/vec![One::one(); self.length()].iter().map(|x| *x).sum()
    }
}