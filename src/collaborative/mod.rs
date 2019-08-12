pub use algorithms::Correlation;
pub use correlation::{item_correlation, user_correlation};
pub use mat::{CsMatBaseExt, CsMatFloat, CsVecBaseExt, CsVecFloat, Value};

mod algorithms;
mod correlation;
mod mat;
