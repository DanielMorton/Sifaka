use std::fmt::Display;
use std::iter::Sum;

use num_traits::{Float, Num};
use sprs::{CsMatI, CsVecI, SpIndex};

use super::{Correlation, RecommenderType};
use super::correlation;
use super::CsMatBaseExt;

fn top_n_recommendations<N, I>(user_item: &CsMatI<N, I>,
                               users: &Vec<I>,
                               neighbors: &u32,
                               corr: &Correlation,
                               rec: &RecommenderType) -> () where I:SpIndex,
N: Num + Copy + Default + Display + Sum + Float {
    let sim_mat = match corr {
        Correlation::Cosine => correlation(user_item, rec),
        Correlation::Pearson => match rec {
            RecommenderType::UserUser => correlation(&user_item.col_center(), rec),
            RecommenderType::ItemItem => correlation(&user_item.row_center(), rec)
        }
    };

}