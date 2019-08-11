use std::ops::Deref;

use num_traits::Float;
use sprs::{CsVecBase, CsVecI, SpIndex};

use super::{CsVecBaseExt, Value};

pub trait CsVecFloat<N, I>: CsVecBaseExt<N, I> {
    fn l2_norm(&self) -> N;
    fn normalize(&self) -> CsVecI<N, I>;
}

impl<N, I, IS, DS> CsVecFloat<N, I> for CsVecBase<IS, DS>
where
    I: SpIndex,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [N]>,
    N: Value + Float,
{
    fn l2_norm(&self) -> N {
        self.data_fold(N::zero(), |s, &x| s + x * x).sqrt()
    }

    fn normalize(&self) -> CsVecI<N, I> {
        let norm = self.l2_norm();
        self.map(|x| *x / norm)
    }
}

#[cfg(test)]
mod tests {
    use sprs::CsVecI;

    use assert_approx_eq::assert_approx_eq;

    use super::CsVecFloat;

    #[test]
    fn test_norm() {
        let v1 = CsVecI::new(5, vec![0, 2, 4], vec![3.14f64, 2.7, 1.6]);
        assert_approx_eq!(v1.l2_norm(), 4.439549);
        let v2 = CsVecI::new(5, vec![0, 2, 4], vec![-3.14f64, -2.7, -1.6]);
        assert_approx_eq!(v2.l2_norm(), 4.439549);
    }

    #[test]
    fn test_normalize() {
        let v1 = CsVecI::new(14, vec![0, 8, 13], vec![3f64, 4.0, 12.0]);
        let v1_norm = CsVecI::new(
            14,
            vec![0, 8, 13],
            vec![3f64 / 13f64, 4.0 / 13.0, 12.0 / 13.0],
        );
        assert_eq!(v1.normalize(), v1_norm);
        let v2 = CsVecI::new(14, vec![0, 8, 13], vec![-3f64, -4.0, 12.0]);
        let v2_norm = CsVecI::new(
            14,
            vec![0, 8, 13],
            vec![-3f64 / 13f64, -4.0 / 13.0, 12.0 / 13.0],
        );
        assert_eq!(v2.normalize(), v2_norm);
    }
}
