use std::iter::Sum;
use std::ops::Deref;
use std::slice::Iter;

use num_traits::Num;
use num_traits::real::Real;
use sprs::{CsVecBase, CsVecI};
use sprs::SpIndex;

pub trait CsVecBaseExt<N, I> {
    fn length(&self) -> usize;
    fn ind_iter(&self) -> Iter<I>;
    fn data_iter(&self) -> Iter<N>;

    fn ind_vec(&self) -> Vec<I>;
    fn data_vec(&self) -> Vec<N>;
    fn data_fold<T>(&self, init: N, f: T) -> N where T: Fn(N, &N) -> N;
    fn sum(&self) -> N;
    fn avg(&self) -> N;
    fn center(&self) -> CsVecI<N, I>;
    fn l1_norm(&self) -> N;
}

impl<N, I, IS, DS> CsVecBaseExt<N, I> for CsVecBase<IS, DS>
where  I: SpIndex,
       IS: Deref<Target = [I]>,
       DS: Deref<Target = [N]>,
       N: Num + Sum + Real {

    fn length(&self) -> usize { self.indices().len() }

    fn ind_iter(&self) -> Iter<I> { self.indices().iter() }

    fn data_iter(&self) -> Iter<N> { self.data().iter() }

    fn ind_vec(&self) -> Vec<I> { self.indices().to_vec() }

    fn data_vec(&self) -> Vec<N> { self.data().to_vec() }

    fn data_fold<T>(&self, init: N, f: T) -> N where T: Fn(N, &N) -> N {
        self.data_iter().fold(init, f)
    }

    fn sum(&self) -> N {
        let s = |s: N, &x: &N| s + x;
        self.data_fold(N::zero(), s)
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
        self.map(|x: &N| *x - avg)
    }

    fn l1_norm(&self) -> N {
        self.data_fold(N::zero(), |s, &x| s + x.abs())
    }
}

#[cfg(test)]
mod tests {
    use sprs::CsVecI;

    use assert_approx_eq::assert_approx_eq;

    use super::CsVecBaseExt;

    #[test]
    fn test_length() {
        let v  = CsVecI::new(5, vec![0, 2, 4], vec![3.14, 2.7, 1.6]);
        assert_eq!(v.length(), 3);
    }

    #[test]
    fn test_float_sum() {
        let v  = CsVecI::new(5, vec![0, 2, 4], vec![3.14, 2.70, 1.60]);
        assert_approx_eq!(v.sum(), 7.44f64);
    }
}