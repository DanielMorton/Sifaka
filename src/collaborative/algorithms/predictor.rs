use sprs::{CsMatI, SpIndex};

use super::Correlation;

pub(super) trait Predictor<N, I>
where
    I: SpIndex,
{
    fn new(user_item: &CsMatI<N, I>,
           neighbors: I,
           correlation: Correlation) -> Self;

    //    fn predict(&self, user: &I, item: &I) -> N;

    //    fn predict_user(&self, user: &I, items: &[I]) -> N;
}
