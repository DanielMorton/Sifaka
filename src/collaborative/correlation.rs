use std::iter::Sum;

use num_traits::{Float, Num, Signed};
use sprs::{CsMatI, SpIndex};

use crate::collaborative::CsMatBaseExt;
use crate::collaborative::mat::Value;

use super::{Correlation, RecommenderType};
use super::mat::CsMatFloat;

fn correlation<N, I>(
    user_item: &CsMatI<N, I>,
    corr: &Correlation,
    target_user: Option<&CsMatI<N, I>>,
) -> CsMatI<N, I>
where
    I: SpIndex,
    N: Value + Default + Float,
{
    if let Some(tu) = target_user {
        let (user_norm, target_norm) = match corr {
            Correlation::Cosine => (user_item.col_normalize(), tu.col_normalize()),
            Correlation::Pearson => (
                user_item.col_center().col_normalize(),
                tu.col_center().col_normalize(),
            ),
        };
        &user_norm.view() * &target_norm.transpose_view()
    } else {
        let item_norm = match corr {
            Correlation::Cosine => user_item.row_normalize(),
            Correlation::Pearson => user_item.row_center().row_normalize(),
        };
        &item_norm.transpose_view() * &item_norm.view()
    }
}

pub fn item_correlation<N, I>(user_item: &CsMatI<N, I>, corr: &Correlation) -> CsMatI<N, I>
where
    I: SpIndex,
    N: Value + Default + Float,
{
    correlation(user_item, corr, None)
}

pub fn user_correlation<N, I>(
    user_item: &CsMatI<N, I>,
    target_user: &CsMatI<N, I>,
    corr: &Correlation,
) -> CsMatI<N, I>
where
    I: SpIndex,
    N: Value + Default + Float,
{
    correlation(user_item, corr, Some(target_user))
}
