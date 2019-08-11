pub use correlation::Correlation;
use predictor::Predictor;

pub use super::{CsMatBaseExt, CsMatFloat, Value};

mod correlation;
mod item_item;
mod predictor;
mod user_user;