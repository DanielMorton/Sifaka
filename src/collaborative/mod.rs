pub use algorithms::Correlation;
pub use correlation::{item_correlation, user_correlation};
pub use mat::{CsMatBaseExt, CsMatFloat, CsVecBaseExt, CsVecFloat, Value};
pub use recommender_type::RecommenderType;

mod algorithms;
mod correlation;
mod mat;
mod recommender_type;
