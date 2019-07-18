use std::iter::Sum;

use num_traits::{Float, Num, Signed};
use sprs::CsMatI;
use sprs::SpIndex;

use super::mat::CsMatFloat;
use super::RecommenderType;

pub fn correlation<N, I>(user_item: &CsMatI<N, I>,
                         rec: &RecommenderType) -> CsMatI<N, I>
    where
        I: SpIndex,
        N: Num + Sum + Clone + Copy + Signed + Default + Float {
    match rec {
        RecommenderType::UserUser => {
            let user_norm = user_item.col_normalize();
            &user_norm.view() * &user_norm.transpose_view()
        },
        RecommenderType::ItemItem => {
            let item_norm = user_item.row_normalize();
            &item_norm.transpose_view() * &item_norm.view()
        }
    }
}