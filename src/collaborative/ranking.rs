use std::iter::Sum;

use num_traits::{Float, Num, Signed};
use sprs::{CsMatI, SpIndex};

use super::{Correlation, CsMatBaseExt, item_correlation, RecommenderType, user_correlation};

fn top_n_recommendations<N, I>(
    user_item: &CsMatI<N, I>,
    users: &Vec<I>,
    neighbors: &u32,
    corr: &Correlation,
    rec: &RecommenderType,
) -> ()
where
    I: SpIndex,
    N: Num + Sum + Clone + Copy + Signed + Default + Float,
{

}
