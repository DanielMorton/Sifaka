use std::ops::Deref;
use std::slice::Iter;

use sprs::{CsVecBase, CsVecI, SpIndex};

use super::Value;

pub trait CsVecBaseExt<N, I> {
    fn ind_iter(&self) -> Iter<I>;
    fn data_iter(&self) -> Iter<N>;

    fn ind_vec(&self) -> Vec<I>;
    fn data_vec(&self) -> Vec<N>;
    fn data_fold<T>(&self, init: N, f: T) -> N
    where
        T: Fn(N, &N) -> N;

    fn sum(&self) -> N;
    fn avg(&self) -> N;
    fn center(&self) -> CsVecI<N, I>;

    fn top_n(&self, n: I, pos: bool) -> CsVecI<N, I>;
    fn top_n_positive(&self, n: I) -> CsVecI<N, I>;
    fn threshold(&self, n: N) -> CsVecI<N, I>;
}

impl<N, I, IS, DS> CsVecBaseExt<N, I> for CsVecBase<IS, DS>
where
    I: SpIndex,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [N]>,
    N: Value,
{
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

    fn data_fold<T>(&self, init: N, f: T) -> N
    where
        T: Fn(N, &N) -> N,
    {
        self.data_iter().fold(init, f)
    }

    fn sum(&self) -> N {
        let s = |s: N, &x: &N| s + x;
        self.data_fold(N::zero(), s)
    }

    fn avg(&self) -> N {
        if self.nnz() != 0 {
            self.sum() / vec![N::one(); self.nnz()].iter().copied().sum()
        } else {
            N::zero()
        }
    }

    fn center(&self) -> CsVecI<N, I> {
        let avg = self.avg();
        self.map(|x: &N| *x - avg)
    }

    fn top_n(&self, n: I, pos: bool) -> CsVecI<N, I> {
        let mut pairs: Vec<(&I, &N)> = self.ind_iter().zip(self.data_iter()).collect();
        pairs.sort_by(|a, b| (b.1).partial_cmp(a.1).unwrap());
        pairs = pairs[..n.index()].to_vec();

        if pos {
            pairs = pairs.into_iter().filter(|p| p.1 > &N::zero()).collect();
        }

        pairs.sort_by(|a, b| (a.0).partial_cmp(b.0).unwrap());
        CsVecI::new(
            self.dim(),
            pairs.iter().map(|p| *p.0).collect(),
            pairs.iter().map(|p| *p.1).collect(),
        )
    }

    fn top_n_positive(&self, n: I) -> CsVecI<N, I> {
        self.top_n(n, true)
    }

    fn threshold(&self, n: N) -> CsVecI<N, I> {
        let pairs: Vec<(&I, &N)> = self
            .ind_iter()
            .zip(self.data_iter())
            .filter(|p| p.1 >= &n)
            .collect();

        CsVecI::new(
            self.dim(),
            pairs.iter().map(|p| *p.0).collect(),
            pairs.iter().map(|p| *p.1).collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use sprs::CsVecI;

    use assert_approx_eq::assert_approx_eq;

    use super::CsVecBaseExt;

    lazy_static! {
        static ref V_FLOAT: CsVecI<f64, usize> =
            CsVecI::new(5, vec![0, 2, 4], vec![3.14, 2.70, 1.60]);
        static ref V_INT: CsVecI<i32, usize> = CsVecI::new(5, vec![0, 2, 4], vec![3, 2, 1]);
        static ref V_INT_T: CsVecI<i32, usize> =
            CsVecI::new(8, vec![0, 2, 3, 5, 7], vec![1, 5, 2, 4, 3]);
        static ref V_FLOAT_T: CsVecI<f64, usize> =
            CsVecI::new(8, vec![0, 2, 3, 5, 7], vec![1.60, 5.4, 2.718, 4.67, 3.14]);
    }
    #[test]
    fn test_float_sum() {
        assert_approx_eq!(V_FLOAT.sum(), 7.44f64);
    }

    #[test]
    fn test_int_sum() {
        assert_eq!(V_INT.sum(), 6);
    }

    #[test]
    fn test_float_avg() {
        assert_approx_eq!(V_FLOAT.avg(), 2.48);
        let n: CsVecI<f64, usize> = CsVecI::new(6, Vec::new(), Vec::new());
        assert_eq!(n.avg(), 0.0)
    }

    #[test]
    fn test_int_avg() {
        assert_eq!(V_INT.avg(), 2);
        let n: CsVecI<i32, usize> = CsVecI::new(6, Vec::new(), Vec::new());
        assert_eq!(n.avg(), 0)
    }

    #[test]
    fn test_int_center() {
        let v_cent = CsVecI::new(5, vec![0, 2, 4], vec![1, 0, -1]);
        assert_eq!(V_INT.center(), v_cent);
    }

    #[test]
    fn test_float_center() {
        let v_cent = CsVecI::new(5, vec![0, 2, 4], vec![0.66, 0.22, -0.88]);
        (&V_FLOAT.center() - &v_cent)
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
    }

    #[test]
    fn test_int_top_n() {
        let v = CsVecI::new(8, vec![2, 5], vec![5, 4]);
        let v_neg = CsVecI::new(8, vec![0, 3], vec![-1, -2]);
        let v_emp: CsVecI<i32, usize> = CsVecI::new(8, Vec::new(), Vec::new());
        assert_eq!(V_INT_T.top_n(2, false), v);
        assert_eq!(V_INT_T.top_n(2, true), v);
        assert_eq!(V_INT_T.map(|x| -1 * x).top_n(2, false), v_neg);
        assert_eq!(V_INT_T.map(|x| -1 * x).top_n(2, true), v_emp)
    }

    #[test]
    fn test_float_top_n() {
        let v = CsVecI::new(8, vec![2, 5], vec![5.4, 4.67]);
        let v_neg = CsVecI::new(8, vec![0, 3], vec![-1.6, -2.718]);
        let v_emp: CsVecI<f64, usize> = CsVecI::new(8, Vec::new(), Vec::new());
        assert_eq!(V_FLOAT_T.top_n(2, false), v);
        assert_eq!(V_FLOAT_T.top_n(2, true), v);
        assert_eq!(V_FLOAT_T.map(|x| -1.0 * x).top_n(2, false), v_neg);
        assert_eq!(V_FLOAT_T.map(|x| -1.0 * x).top_n(2, true), v_emp)
    }

    #[test]
    fn test_int_top_positive() {
        let v = CsVecI::new(8, vec![2, 5], vec![5, 4]);
        let v_emp: CsVecI<i32, usize> = CsVecI::new(8, Vec::new(), Vec::new());
        assert_eq!(V_INT_T.top_n_positive(2), v);
        assert_eq!(V_INT_T.map(|x| -1 * x).top_n_positive(2), v_emp)
    }

    #[test]
    fn test_float_top_positive() {
        let v = CsVecI::new(8, vec![2, 5], vec![5.4, 4.67]);
        let v_emp: CsVecI<f64, usize> = CsVecI::new(8, Vec::new(), Vec::new());
        assert_eq!(V_FLOAT_T.top_n_positive(2), v);
        assert_eq!(V_FLOAT_T.map(|x| -1.0 * x).top_n_positive(2), v_emp)
    }

    #[test]
    fn test_int_threshold() {
        let v = CsVecI::new(8, vec![2, 5, 7], vec![5, 4, 3]);
        assert_eq!(V_INT_T.threshold(3), v)
    }

    #[test]
    fn test_float_threshold() {
        let v = CsVecI::new(8, vec![2, 5, 7], vec![5.4, 4.67, 3.14]);
        assert_eq!(V_FLOAT_T.threshold(3.0), v)
    }
}
