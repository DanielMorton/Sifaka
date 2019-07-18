use std::iter::Sum;
use std::ops::Deref;
use std::slice::Iter;

use num_traits::{Num, Signed};
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
    fn l1_norm(&self) -> N;
    fn center(&self) -> CsVecI<N, I>;
}

impl<N, I, IS, DS> CsVecBaseExt<N, I> for CsVecBase<IS, DS>
where  I: SpIndex,
       IS: Deref<Target = [I]>,
       DS: Deref<Target = [N]>,
       N: Num + Sum + Copy + Clone + Signed {

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
        } else { N::zero() }
    }

    fn l1_norm(&self) -> N { self.data_fold(N::zero(), |s, &x| s + x.abs()) }

    fn center(&self) -> CsVecI<N, I> {
        let avg = self.avg();
        self.map(|x: &N| *x - avg)
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

    #[test]
    fn test_int_sum() {
        let v  = CsVecI::new(5, vec![0, 2, 4], vec![3, 2, 1]);
        assert_eq!(v.sum(), 6);
    }

    #[test]
    fn test_float_avg() {
        let v  = CsVecI::new(5, vec![0, 2, 4], vec![3.14f64, 2.7, 1.6]);
        assert_approx_eq!(v.avg(), 2.48);
        let n: CsVecI<f64, usize> = CsVecI::new(6, Vec::new(), Vec::new());
        assert_eq!(n.avg(), 0.0)
    }

    #[test]
    fn test_int_avg() {
        let v  = CsVecI::new(5, vec![0, 2, 4], vec![3, 2, 1]);
        assert_eq!(v.avg(), 2);
        let n: CsVecI<i32, usize> = CsVecI::new(6, Vec::new(), Vec::new());
        assert_eq!(n.avg(), 0)
    }

    #[test]
    fn test_int_norm() {
        let v1  = CsVecI::new(5, vec![0, 2, 4], vec![3, 2, 1]);
        let v2  = CsVecI::new(5, vec![0, 2, 4], vec![-3, -2, -1]);
        assert_eq!(v1.l1_norm(), 6);
        assert_eq!(v2.l1_norm(), 6);
    }

    #[test]
    fn test_float_norm() {
        let v1  = CsVecI::new(5, vec![0, 2, 4], vec![3.14f64, 2.7, 1.6]);
        let v2  = CsVecI::new(5, vec![0, 2, 4], vec![-3.14f64, -2.7, -1.6]);
        assert_approx_eq!(v1.l1_norm(), 7.44f64);
        assert_approx_eq!(v2.l1_norm(), 7.44f64);
    }

    fn test_int_center() {
        let v  = CsVecI::new(5, vec![0, 2, 4], vec![3, 2, 1]);
        let v_cent = CsVecI::new(5, vec![0, 2, 4], vec![1, 0, -1]);
        assert_eq!(v.center(), v_cent);
    }

    fn test_float_center() {
        let v  = CsVecI::new(5, vec![0, 2, 4], vec![3.14f64, 2.7, 1.6]).center();
        let v_cent = CsVecI::new(5, vec![0, 2, 4], vec![0.66, 0.22, -0.88]);
        assert_eq!(v, v_cent);
    }
}