use std::iter::Sum;

use num_traits::{Float, Num, Signed};
use sprs::{CsMatI, SpIndex};

use super::{
    Correlation, CsMatBaseExt, item_correlation, RecommenderType, user_correlation, Value,
};

fn k_neighbors<N, I>(
    user_item: &CsMatI<N, I>,
    neighbors: &u32,
    corr: &Correlation,
    rec: &RecommenderType,
) -> ()
where
    I: SpIndex,
    N: Value + Default,
{

}
