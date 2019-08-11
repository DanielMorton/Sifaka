pub use algorithms::Correlation;
pub use correlation::{item_correlation, user_correlation};
pub use mat::{CsMatBaseExt, CsMatFloat, CsVecBaseExt, CsVecFloat};
pub use recommender_type::RecommenderType;

pub use crate::collaborative::mat::Value;

pub mod algorithms;
pub mod correlation;
pub mod mat;
pub mod recommender_type;
