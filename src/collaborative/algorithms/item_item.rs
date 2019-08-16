use num_traits::Float;
use sprs::{CsMatI, SpIndex};

use super::{Correlation, CsMatBaseExt, CsMatFloat, Predictor, Value};

struct ItemItem<N, I>
where
    I: SpIndex,
{
    neighbors: usize,
    item_item: CsMatI<N, I>,
}

impl<N, I> Predictor<N, I> for ItemItem<N, I>
where
    I: SpIndex,
    N: Value + Default + Float,
{
    fn new_k_neighbors(user_item: &CsMatI<N, I>, neighbors: I, correlation: Correlation) -> Self {
        ItemItem {
            neighbors: SpIndex::index(neighbors),
            item_item: user_item
                .corr(&correlation, false)
                .col_top_n(neighbors, true),
        }
    }

    fn new_threshold(user_item: &CsMatI<N, I>, threshold: N, correlation: Correlation) -> Self {
        let item_item = user_item.corr(&correlation, false).threshold(threshold);
        let neighbors = *item_item
            .row_nnz()
            .data()
            .iter()
            .max_by_key(|x| *x)
            .unwrap();
        ItemItem {
            neighbors,
            item_item,
        }
    }
}
