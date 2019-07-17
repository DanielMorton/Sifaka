pub use cor_type::Correlation as Correlation;
pub use correlation::correlation as correlation;
pub use mat::CsMatBaseExt as CsMatBaseExt;
pub use mat::CsMatFloat as CsMatFloat;
pub use mat::CsVecBaseExt as CsVecBaseExt;
pub use mat::CsVecFloat as CsVecFloat;
pub use recommender_type::RecommenderType as RecommenderType;

pub mod mat;
pub mod correlation;
pub mod cor_type;
pub mod ranking;
pub mod recommender_type;

