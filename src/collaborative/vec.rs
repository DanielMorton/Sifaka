use std::iter::Sum;
use std::ops::Deref;

use num_traits::identities::One;
use num_traits::Num;
use sprs::CsVecBase;
use sprs::SpIndex;

pub trait CsVecExt<N> {
    fn sum(&self) -> N;
    fn length(&self) -> usize;
    fn avg(&self) -> N;
}

impl<N, I, IS, DS> CsVecExt<N> for CsVecBase<IS, DS>
where  I: SpIndex,
       IS: Deref<Target = [I]>,
       DS: Deref<Target = [N]>,
       N: Num + Copy + Sum {
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