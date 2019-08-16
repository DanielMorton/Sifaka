use std::ops::Deref;

use num_traits::Num;
use sprs::{CsMatBase, CsMatI, CsVecBase, CsVecI, CsVecViewI, SpIndex};

use super::{CsVecBaseExt, Value};

pub(super) trait CsMatBaseHelp<N, I>
where
    I: SpIndex,
{
    fn outer_agg<F: Copy, M>(&self, func: F) -> CsVecI<M, I>
    where
        F: Fn(&CsVecI<N, I>) -> M,
        M: Num + Copy + Default;

    fn outer_transform<F: Copy, M>(&self, func: F) -> CsMatI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> CsVecI<M, I>,
        M: Value + Default;

    fn outer_top_n(&self, n: I, pos: bool) -> CsMatI<N, I>;

    fn inner_agg< F: Copy, M>(& self, func: F) -> CsVecI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> M,
            M: Num + Copy + Default;

    fn inner_transform<F: Copy, M>(&self, func: F) -> CsMatI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> CsVecI<M, I>,
            M: Value + Default;

    fn inner_top_n(&self, n: I, pos: bool) -> CsMatI<N, I>;
}

impl<N, I, IS, DS> CsMatBaseHelp<N, I> for CsMatBase<N, I, IS, IS, DS>
where
    I: SpIndex,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [N]>,
    N: Value + Default,
{
    fn outer_agg<F: Copy, M>(&self, func: F) -> CsVecI<M, I>
    where
        F: Fn(&CsVecI<N, I>) -> M,
        M: Num + Copy + Default
    {
        let mut ind_vec: Vec<I> = Vec::new();
        let mut agg_vec: Vec<M> = Vec::new();
        for (ind, vec) in self.outer_iterator().enumerate() {
            let v = func(&vec.to_owned());
            if v != M::zero() {
                ind_vec.push(SpIndex::from_usize(ind));
                agg_vec.push(v);
            }
        }
        CsVecI::new(self.outer_dims(), ind_vec, agg_vec)
    }

    fn outer_transform<F: Copy, M>(&self, func: F) -> CsMatI<M, I>
    where
        F: Fn(&CsVecI<N, I>) -> CsVecI<M, I>,
        M: Value + Default,
    {
        let mut data: Vec<M> = Vec::new();
        for (_, vec) in self.outer_iterator().enumerate() {
            data.append(&mut func(&vec.to_owned()).data_vec());
        }
        if self.is_csc() {
            CsMatI::new_csc(self.shape(), self.ip_vec(), self.ind_vec(), data)
        } else {
            CsMatI::new(self.shape(), self.ip_vec(), self.ind_vec(), data)
        }
    }

    fn outer_top_n(&self, n: I, pos: bool) -> CsMatI<N, I> {
        let mut ip_vec: Vec<I> = Vec::new();
        let mut ind_vec: Vec<I> = Vec::new();
        let mut data: Vec<N> = Vec::new();
        for (_, vec) in self.outer_iterator().enumerate() {
            let top_n_vec = vec.top_n(n, pos);
            ind_vec.append(&mut top_n_vec.ind_vec());
            data.append(&mut top_n_vec.data_vec());
            ip_vec.push(SpIndex::from_usize(ind_vec.len()));
        }
        if self.is_csc() {
            CsMatI::new_csc(self.shape(), ip_vec, ind_vec, data)
        } else {
            CsMatI::new(self.shape(), ip_vec, ind_vec, data)
        }
    }

    fn inner_agg<F: Copy, M>(&self, func: F) -> CsVecI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> M,
            M: Num + Copy + Default {
        self.to_other_storage().outer_agg(func)
    }

    fn inner_transform<F: Copy, M>(&self, func: F) -> CsMatI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> CsVecI<M, I>,
            M: Value + Default {
        self.to_other_storage().outer_transform(func).to_other_storage()
    }

    fn inner_top_n(&self, n: I, pos: bool) -> CsMatI<N, I> {
        self.to_other_storage()
            .outer_top_n(n, pos)
            .to_other_storage()
    }
}

pub trait CsMatBaseExt<N, I>
where
    I: SpIndex,
{
    fn ip_vec(&self) -> Vec<I>;
    fn ind_vec(&self) -> Vec<I>;
    fn data_vec(&self) -> Vec<N>;

    fn col_agg< F: Copy, M>(& self, func: F) -> CsVecI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> M,
            M: Num + Copy + Default;

    fn row_agg< F: Copy, M>(& self, func: F) -> CsVecI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> M,
            M: Num + Copy + Default;

    fn col_transform<F: Copy, M>(&self, func: F) -> CsMatI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> CsVecI<M, I>,
            M: Value + Default;

    fn row_transform<F: Copy, M>(&self, func: F) -> CsMatI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> CsVecI<M, I>,
            M: Value + Default;

    fn col_sum(&self) -> CsVecI<N, I>;
    fn row_sum(&self) -> CsVecI<N, I>;

    fn col_avg(&self) -> CsVecI<N, I>;
    fn row_avg(&self) -> CsVecI<N, I>;

    fn col_l1_norm(&self) -> CsVecI<N, I>;
    fn row_l1_norm(&self) -> CsVecI<N, I>;

    fn col_nnz(&self) -> CsVecI<usize, I>;
    fn row_nnz(&self) -> CsVecI<usize, I>;

    fn col_center(&self) -> CsMatI<N, I>;
    fn row_center(&self) -> CsMatI<N, I>;

    fn col_top_n(&self, n: I, pos: bool) -> CsMatI<N, I>;
    fn row_top_n(&self, n: I, pos: bool) -> CsMatI<N, I>;

    fn threshold(&self, n: N) -> CsMatI<N, I>;
}

impl<N, I, IS, DS> CsMatBaseExt<N, I> for CsMatBase<N, I, IS, IS, DS>
where
    I: SpIndex,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [N]>,
    N: Value + Default,
{
    fn ip_vec(&self) -> Vec<I> {
        self.indptr().to_vec()
    }

    fn ind_vec(&self) -> Vec<I> {
        self.indices().to_vec()
    }

    fn data_vec(&self) -> Vec<N> {
        self.data().to_vec()
    }

    fn col_agg< F: Copy, M>(&self, func: F) -> CsVecI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> M,
            M: Num + Copy + Default {
        if self.is_csc() {
            self.outer_agg(func)
        } else {
            self.inner_agg(func)
        }
    }

    fn row_agg< F: Copy, M>(&self, func: F) -> CsVecI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> M,
            M: Num + Copy + Default {
        if self.is_csr() {
            self.outer_agg(func)
        } else {
            self.inner_agg(func)
        }
    }

    fn col_transform<F: Copy, M>(&self, func: F) -> CsMatI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> CsVecI<M, I>,
            M: Value + Default {
        if self.is_csc() {
            self.outer_transform(func)
        } else {
            self.inner_transform(func)
        }
    }

    fn row_transform<F: Copy, M>(&self, func: F) -> CsMatI<M, I>
        where
            F: Fn(&CsVecI<N, I>) -> CsVecI<M, I>,
            M: Value + Default {
        if self.is_csr() {
            self.outer_transform(func)
        } else {
            self.inner_transform(func)
        }
    }

    fn col_sum(&self) -> CsVecI<N, I> {
        self.col_agg(CsVecBase::sum)
    }

    fn row_sum(&self) -> CsVecI<N, I> {
        self.row_agg(CsVecBase::sum)
    }

    fn col_avg(&self) -> CsVecI<N, I> {
        self.col_agg(CsVecBase::avg)
    }

    fn row_avg(&self) -> CsVecI<N, I> {
        self.row_agg(CsVecBase::avg)
    }

    fn col_l1_norm(&self) -> CsVecI<N, I> {
        self.col_agg(CsVecBase::l1_norm)
    }

    fn row_l1_norm(&self) -> CsVecI<N, I> {
        self.row_agg(CsVecBase::l1_norm)
    }

    fn col_nnz(&self) -> CsVecI<usize, I> {
        self.col_agg(CsVecBase::nnz)
    }

    fn row_nnz(&self) -> CsVecI<usize, I> {
        self.row_agg(CsVecBase::nnz)
    }

    fn col_center(&self) -> CsMatI<N, I> {
        self.col_transform(CsVecBase::center)
    }

    fn row_center(&self) -> CsMatI<N, I> {
        self.row_transform(CsVecBase::center)
    }

    fn col_top_n(&self, n: I, pos: bool) -> CsMatI<N, I> {
        if self.is_csc() {
            self.outer_top_n(n, pos)
        } else {
            self.inner_top_n(n, pos)
        }
    }

    fn row_top_n(&self, n: I, pos: bool) -> CsMatI<N, I> {
        if self.is_csr() {
            self.outer_top_n(n, pos)
        } else {
            self.inner_top_n(n, pos)
        }
    }

    fn threshold(&self, n: N) -> CsMatI<N, I> {
        let mut ip_vec: Vec<I> = Vec::new();
        let mut ind_vec: Vec<I> = Vec::new();
        let mut data: Vec<N> = Vec::new();
        for (_, vec) in self.outer_iterator().enumerate() {
            let threshold_vec = vec.threshold(n);
            ind_vec.append(&mut threshold_vec.ind_vec());
            data.append(&mut threshold_vec.data_vec());
            ip_vec.push(SpIndex::from_usize(ind_vec.len()));
        }
        if self.is_csc() {
            CsMatI::new_csc(self.shape(), ip_vec, ind_vec, data)
        } else {
            CsMatI::new(self.shape(), ip_vec, ind_vec, data)
        }
    }
}

#[cfg(test)]
mod tests {
    use sprs::{CsMatI, CsVecI};

    use assert_approx_eq::assert_approx_eq;

    use super::CsMatBaseExt;

    lazy_static! {
        static ref A_FLOAT: CsMatI<f64, usize> = CsMatI::new_csc(
            (5, 4),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![3.14, 2.7, 1.62, 0.58, 4.67]
        );
        static ref A_INT: CsMatI<i32, usize> = CsMatI::new_csc(
            (5, 4),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![1, 2, 3, 4, 5]
        );
        static ref B_FLOAT: CsMatI<f64, usize> = CsMatI::new(
            (4, 5),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![3.14, 2.7, 1.62, 0.58, 4.67]
        );
        static ref B_INT: CsMatI<i32, usize> = CsMatI::new(
            (4, 5),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn test_int_sum() {
        let outer = CsVecI::new(4, vec![0, 2, 3], vec![3, 7, 5]);
        let inner = CsVecI::new(5, vec![0, 1, 3], vec![4, 2, 9]);
        assert_eq!(A_INT.col_sum(), outer);
        assert_eq!(A_INT.row_sum(), inner);
        assert_eq!(B_INT.col_sum(), inner);
        assert_eq!(B_INT.row_sum(), outer)
    }

    #[test]
    fn test_float_sum() {
        let outer = CsVecI::new(4, vec![0, 2, 3], vec![5.84, 2.2, 4.67]);
        let inner = CsVecI::new(5, vec![0, 1, 3], vec![4.76, 2.7, 5.25]);
        assert_eq!(A_FLOAT.col_sum(), outer);
        assert_eq!(A_FLOAT.row_sum(), inner);
        assert_eq!(B_FLOAT.col_sum(), inner);
        assert_eq!(B_FLOAT.row_sum(), outer)
    }

    #[test]
    fn test_int_avg() {
        let outer = CsVecI::new(4, vec![0, 2, 3], vec![1, 3, 5]);
        let inner = CsVecI::new(5, vec![0, 1, 3], vec![2, 2, 4]);
        assert_eq!(A_INT.col_avg(), outer);
        assert_eq!(A_INT.row_avg(), inner);
        assert_eq!(B_INT.col_avg(), inner);
        assert_eq!(B_INT.row_avg(), outer)
    }

    #[test]
    fn test_float_avg() {
        let outer = CsVecI::new(4, vec![0, 2, 3], vec![2.92, 1.1, 4.67]);
        let inner = CsVecI::new(5, vec![0, 1, 3], vec![2.38, 2.7, 2.625]);
        assert_eq!(A_FLOAT.col_avg(), outer);
        assert_eq!(A_FLOAT.row_avg(), inner);
        assert_eq!(B_FLOAT.col_avg(), inner);
        assert_eq!(B_FLOAT.row_avg(), outer)
    }

    #[test]
    fn test_int_center() {
        let out_center = CsMatI::new_csc(
            (5, 4),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![0, 1, 0, 1, 0],
        );
        let in_center = CsMatI::new_csc(
            (5, 4),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![-1, 0, 1, 0, 1],
        );
        assert_eq!(A_INT.col_center(), out_center);
        assert_eq!(A_INT.row_center(), in_center);
        assert_eq!(B_INT.col_center(), in_center.transpose_into());
        assert_eq!(B_INT.row_center(), out_center.transpose_into());
    }

    #[test]
    fn test_float_center() {
        let out_center: CsMatI<f64, usize> = CsMatI::new_csc(
            (5, 4),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![0.22, -0.22, 0.52, -0.52, 0.0],
        );
        let in_center: CsMatI<f64, usize> = CsMatI::new_csc(
            (5, 4),
            vec![0, 2, 2, 4, 5],
            vec![0, 1, 0, 3, 3],
            vec![0.76, 0.0, -0.76, -2.045, 2.045],
        );
        assert_eq!(A_FLOAT.col_center().ind_vec(), out_center.ind_vec());
        assert_eq!(A_FLOAT.row_center().ind_vec(), in_center.ind_vec());
        assert_eq!(
            B_FLOAT.col_center().ind_vec(),
            in_center.transpose_view().ind_vec()
        );
        assert_eq!(
            B_FLOAT.row_center().ind_vec(),
            out_center.transpose_view().ind_vec()
        );
        assert_eq!(A_FLOAT.col_center().ip_vec(), out_center.ip_vec());
        assert_eq!(A_FLOAT.row_center().ip_vec(), in_center.ip_vec());
        assert_eq!(
            B_FLOAT.col_center().ip_vec(),
            in_center.transpose_view().ip_vec()
        );
        assert_eq!(
            B_FLOAT.row_center().ip_vec(),
            out_center.transpose_view().ip_vec()
        );
        (&A_FLOAT.col_center() - &out_center)
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
        (&A_FLOAT.row_center() - &in_center)
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
        (&B_FLOAT.col_center() - &in_center.transpose_view())
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
        (&B_FLOAT.row_center() - &out_center.transpose_view())
            .data_vec()
            .iter()
            .for_each(|x| assert_approx_eq!(x, 0.0));
    }
}
