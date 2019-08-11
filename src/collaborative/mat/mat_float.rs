use std::iter::Sum;
use std::ops::Deref;

use num_traits::Float;
use sprs::{CsMatBase, CsMatI, CsVecBase, CsVecI, SpIndex};

use super::{CsMatBaseExt, CsMatBaseHelp, CsVecBaseExt, CsVecFloat, Value};

trait CSMatFloatHelp<N, I>: CsMatBaseExt<N, I>
where
    I: SpIndex,
{
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
    N: Value + Default + Float,
{
    fn outer_l2_norm(&self) -> CsVecI<N, I> {
        self.outer_agg(CsVecBase::l2_norm)
    }

    fn outer_normalize(&self) -> CsMatI<N, I> {
        let mut data: Vec<N> = Vec::new();
        for (_, vec) in self.outer_iterator().enumerate() {
            data.append(&mut vec.normalize().data_vec());
        }
        if self.is_csc() {
            CsMatI::new_csc(self.shape(), self.ip_vec(), self.ind_vec(), data)
        } else {
            CsMatI::new(self.shape(), self.ip_vec(), self.ind_vec(), data)
        }
    }

    fn inner_l2_norm(&self) -> CsVecI<N, I> {
        self.to_other_storage().outer_l2_norm()
    }

    fn inner_normalize(&self) -> CsMatI<N, I> {
        self.to_other_storage().outer_normalize().to_other_storage()
    }
}

pub trait CsMatFloat<N, I>
where
    I: SpIndex,
{
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
    N: Value + Default + Float,
{
    fn col_l2_norm(&self) -> CsVecI<N, I> {
        if self.is_csc() {
            self.outer_l2_norm()
        } else {
            self.inner_l2_norm()
        }
    }

    fn row_l2_norm(&self) -> CsVecI<N, I> {
        if self.is_csr() {
            self.outer_l2_norm()
        } else {
            self.inner_l2_norm()
        }
    }

    fn col_normalize(&self) -> CsMatI<N, I> {
        if self.is_csc() {
            self.outer_normalize()
        } else {
            self.inner_normalize()
        }
    }

    fn row_normalize(&self) -> CsMatI<N, I> {
        if self.is_csr() {
            self.outer_normalize()
        } else {
            self.inner_normalize()
        }
    }
}

#[cfg(test)]
mod tests {
    use sprs::{CsMatI, CsVecI};

    use assert_approx_eq::assert_approx_eq;

    use super::{CsMatBaseExt, CsMatFloat, CsVecBaseExt};

    lazy_static! {
        static ref A_FLOAT: CsMatI<f64, usize> = CsMatI::new_csc(
            (5, 4),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![3.14, 2.7, 1.62, 0.58, 4.67]
        );
        static ref B_FLOAT: CsMatI<f64, usize> = CsMatI::new(
            (4, 5),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![3.14, 2.7, 1.62, 0.58, 4.67]
        );
    }

    #[test]
    fn test_l2_norm() {
        let outer = CsVecI::new(4, vec![0, 2, 3], vec![4.1412075533, 1.72069753, 4.67]);
        let inner = CsVecI::new(5, vec![0, 1, 3], vec![3.5332704, 2.7, 4.7058793]);

        assert_eq!(A_FLOAT.col_l2_norm().ind_vec(), outer.ind_vec());
        assert_eq!(A_FLOAT.row_l2_norm().ind_vec(), inner.ind_vec());
        assert_eq!(B_FLOAT.col_l2_norm().ind_vec(), inner.ind_vec());
        assert_eq!(B_FLOAT.row_l2_norm().ind_vec(), outer.ind_vec());
        (&A_FLOAT.col_l2_norm() - &outer)
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
        (&A_FLOAT.row_l2_norm() - &inner)
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
        (&B_FLOAT.col_l2_norm() - &inner)
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
        (&B_FLOAT.row_l2_norm() - &outer)
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
    }

    #[test]
    fn test_normalize() {
        let out_norm: CsMatI<f64, usize> = CsMatI::new_csc(
            (5, 4),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![0.7582329, 0.6519837, 0.9414787, 0.3370726, 1.0],
        );
        let in_norm: CsMatI<f64, usize> = CsMatI::new_csc(
            (5, 4),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![0.8886951, 1.0, 0.4584987, 0.12325008, 0.99237564],
        );
        assert_eq!(A_FLOAT.col_normalize().ind_vec(), out_norm.ind_vec());
        assert_eq!(A_FLOAT.row_normalize().ind_vec(), in_norm.ind_vec());
        assert_eq!(
            B_FLOAT.col_normalize().ind_vec(),
            in_norm.transpose_view().ind_vec()
        );
        assert_eq!(
            B_FLOAT.row_normalize().ind_vec(),
            out_norm.transpose_view().ind_vec()
        );
        assert_eq!(A_FLOAT.col_normalize().ip_vec(), out_norm.ip_vec());
        assert_eq!(A_FLOAT.row_normalize().ip_vec(), in_norm.ip_vec());
        assert_eq!(
            B_FLOAT.col_normalize().ip_vec(),
            in_norm.transpose_view().ip_vec()
        );
        assert_eq!(
            B_FLOAT.row_normalize().ip_vec(),
            out_norm.transpose_view().ip_vec()
        );
        (&A_FLOAT.col_normalize() - &out_norm)
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
        (&A_FLOAT.row_normalize() - &in_norm)
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
        (&B_FLOAT.col_normalize() - &in_norm.transpose_view())
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
        (&B_FLOAT.row_normalize() - &out_norm.transpose_view())
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
    }
}
