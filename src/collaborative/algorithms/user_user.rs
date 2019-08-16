use num_traits::Float;
use sprs::{CsMatI, SpIndex};

use super::{Correlation, CsMatBaseExt, CsMatFloat, Predictor, Value};

struct UserUser<N, I>
where
    I: SpIndex,
{
    neighbors: usize,
    user_user: CsMatI<N, I>,
}

impl<N, I> Predictor<N, I> for UserUser<N, I>
where
    I: SpIndex,
    N: Value + Default + Float,
{
    fn new_k_neighbors(user_item: &CsMatI<N, I>, neighbors: I, correlation: Correlation) -> Self {
        UserUser {
            neighbors: SpIndex::index(neighbors),
            user_user: user_item
                .corr(&correlation, true)
                .row_top_n(neighbors, true),
        }
    }

    fn new_threshold(user_item: &CsMatI<N, I>, threshold: N, correlation: Correlation) -> Self {
        let user_user = user_item.corr(&correlation, true).threshold(threshold);
        let neighbors = *user_user
            .row_nnz()
            .data()
            .iter()
            .max_by_key(|x| *x)
            .unwrap();
        UserUser {
            neighbors,
            user_user,
        }
    }
}
