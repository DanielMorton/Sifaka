use std::iter::Sum;

use num_traits::{Num, Signed};

pub use mat::CsMatBaseExt;
pub use mat_float::CsMatFloat;
pub use vec::CsVecBaseExt;
pub use vec_float::CsVecFloat;

pub mod mat;
pub mod mat_float;
pub mod vec;
pub mod vec_float;

pub trait Value: Num + Sum + Copy + Clone + Signed {}
impl<T> Value for T where T: Num + Sum + Copy + Clone + Signed {}
