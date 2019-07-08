use std::iter::Sum;
use std::ops::Deref;

use num_traits::Num;
use sprs::{CsVecBase, CsVecI};
use sprs::SpIndex;

pub trait CsVecBaseExt<N> {
    fn sum(&self) -> N;
    fn length(&self) -> usize;
    fn avg(&self) -> N;
}

impl<N, I, IS, DS> CsVecBaseExt<N> for CsVecBase<IS, DS>
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
        self.sum()/vec![N::one(); self.length()].iter().map(|x| *x).sum()
    }
}

pub trait CsVecIExt {
    fn center(&self) -> Self;
}


impl<N, I> CsVecIExt for CsVecI<N, I>
    where I: SpIndex,
          N: Num + Copy + Sum {
    fn center(&self) -> Self {
        let avg = self.avg();
        CsVecI::new(self.dim(), self.indices().to_vec(),
                    self.data().iter().map(|x| *x - avg).collect())
    }
}