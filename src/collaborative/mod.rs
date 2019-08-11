pub use algorithms::Correlation;
pub use correlation::{item_correlation, user_correlation};
pub use mat::{CsMatBaseExt, CsMatFloat, CsVecBaseExt, CsVecFloat};
pub use recommender_type::RecommenderType;

pub use crate::collaborative::mat::Value;

mod algorithms;
mod correlation;
mod mat;
mod recommender_type;
