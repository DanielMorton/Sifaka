use std::convert::TryFrom;
use std::iter::Sum;

use num_traits::{Float, Num};
use sprs::CsMatI;
use sprs::SpIndex;

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
    match sim {
        SimType::UserUser=> &user_item.view() * &user_item.transpose_view(),
        SimType::ItemItem => &user_item.transpose_view() * &user_item.view()
    }
}