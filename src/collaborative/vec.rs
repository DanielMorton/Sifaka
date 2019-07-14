use std::iter::Sum;
use std::ops::Deref;
use std::slice::Iter;

use num_traits::{Float, Num};
use sprs::{CsVecBase, CsVecI};
use sprs::SpIndex;

pub trait CsVecBaseExt<N, I> {
    fn ind_iter(&self) -> Iter<I>;
    fn data_iter(&self) -> Iter<N>;

    fn ind_vec(&self) -> Vec<I>;
    fn data_vec(&self) -> Vec<N>;
    fn data_map<T>(&self, f: T) -> Vec<N> where T: Fn(&N) -> N;
    fn sum(&self) -> N;
    fn length(&self) -> usize;
    fn avg(&self) -> N;
    fn center(&self) -> CsVecI<N, I>;
}

impl<N, I, IS, DS> CsVecBaseExt<N, I> for CsVecBase<IS, DS>
where  I: SpIndex,
       IS: Deref<Target = [I]>,
       DS: Deref<Target = [N]>,
       N: Num + Copy + Sum {

    fn ind_iter(&self) -> Iter<I> {
        self.indices().iter()
    }

    fn data_iter(&self) -> Iter<N> {
        self.data().iter()
    }

    fn ind_vec(&self) -> Vec<I> {
        self.indices().to_vec()
    }

    fn data_vec(&self) -> Vec<N> {
        self.data().to_vec()
    }

    fn data_map<T>(&self, f: T) -> Vec<N> where T: Fn(&N) -> N {
        self.data_iter().map(|x| f(x)).collect()
    }

    fn sum(&self) -> N {
        self.data_iter().fold(N::zero(), |s, &x| s + x)
    }

    fn length(&self) -> usize {
        self.indices().len()
    }

    fn avg(&self) -> N {
        if self.length() != 0 {
            self.sum()/vec![N::one(); self.length()].iter().map(|x| *x).sum()
        } else {
            N::zero()
        }

    }

    fn center(&self) -> CsVecI<N, I> {
        let avg = self.avg();
        let c = |x: &N| *x - avg;
        CsVecI::new(self.dim(), self.ind_vec(),
                    self.data_map(c))
    }
}

pub trait CsFloatVec<N, I> {
    fn l2_norm(&self) -> N;
}

impl<N, I, IS, DS> CsFloatVec<N, I> for CsVecBase<IS, DS>
    where  I: SpIndex,
           IS: Deref<Target = [I]>,
           DS: Deref<Target = [N]>,
           N: Num + Copy + Sum + Float {

    fn l2_norm(&self) -> N  {
        self.data_map(|x| *x * *x).iter()
            .fold(N::zero(), |s, &x| s + x).sqrt()
    }
}