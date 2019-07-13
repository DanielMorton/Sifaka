use std::iter::Sum;

use num_traits::{Float, Num};
use sprs::CsMatI;
use sprs::SpIndex;
use std::convert::TryFrom;

enum Correlation {
    Pearson,
    Cosine
}

pub enum SimType {
    UserUser,
    ItemItem
}

pub fn correlation<N, I>(user_item: &CsMatI<N, I>, sim: SimType) -> CsMatI<N, I>
    where I: SpIndex + TryFrom<usize>,
          N: Num + Copy + Default + Sum + Float {
    let item_user = user_item.clone().transpose_into();
    match sim {
        SimType::UserUser=> user_item * &item_user,
        SimType::ItemItem => &item_user * user_item
    }
}