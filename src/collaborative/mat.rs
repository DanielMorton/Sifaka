use sprs::{CsMat, CsVec};
use num_traits::identities::One;
use num_traits::Num;
use std::iter::Sum;
use ndarray::Array1;

pub trait CsMatExt<N: Num + Copy + Default + Sum> {
    fn col_sum(&self) -> Array1<N>;
    fn row_sum(&self) -> Array1<N>;

   // fn col_avg(&self) -> CsVec<N>;
   // fn row_avg(&self) -> CsVec<N>;

    fn one_vec(size: usize) -> Array1<N> {
        let mut ind_vec: Vec<usize> = Vec::with_capacity(size);
        //let one_vec =
        Array1::from_vec(vec![One::one(); size])
        //for i in 0..size {
       //     ind_vec.push(i);
       // }
       // return CsVec::new(size, ind_vec, one_vec);
    }
}

impl<N> CsMatExt<N> for CsMat<N> where N: Num + Copy + Default + Sum {
    fn col_sum(&self) -> Array1<N> {
        let cols = self.cols();
        let col_vec = Self::one_vec(cols);
        self * &col_vec
    }

    fn row_sum(&self) -> Array1<N> {
        let rows = self.cols();
        let row_vec = Self::one_vec(rows);
        &self.transpose_view() * &row_vec
    }

}

