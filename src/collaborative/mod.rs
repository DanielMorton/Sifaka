pub mod mat;
pub mod user;
pub mod vec;
pub mod correlation;

pub use mat::CsMatBaseExt;
pub use vec::{CsFloatVec, CsVecBaseExt};
pub use correlation::{SimType, correlation};