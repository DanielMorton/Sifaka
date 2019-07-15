use std::convert::TryFrom;
use std::iter::Sum;

use num_traits::{Float, Num};
use sprs::CsMatI;
use sprs::SpIndex;
use super::mat::CsMatFloat;

enum Correlation {
    Pearson,
    Cosine
}

pub enum SimType {
    UserUser,
    ItemItem
}

pub fn correlation<N, I>(user_item: &CsMatI<N, I>, sim: &SimType) -> CsMatI<N, I>
    where I: SpIndex + From<usize>,
          N: Num + Copy + Default + Sum + Float {
    match sim {
        SimType::UserUser=> {
            let user_norm = user_item.col_normalize();
            &user_norm.view() * &user_norm.transpose_view()
        },
        SimType::ItemItem => {
            let item_norm = user_item.row_normalize();
            &item_norm.transpose_view() * &item_norm.view()
        }
    }
}