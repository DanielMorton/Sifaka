use num_traits::Float;
use sprs::{CsMatI, SpIndex};

use super::{Correlation, CsMatBaseExt, CsMatFloat, Predictor, Value};

struct UserUser<N, I>
where
    I: SpIndex,
{
    user_user: CsMatI<N, I>,
    neighbors: I,
    correlation: Correlation,
}

impl<N, I> Predictor<N, I> for UserUser<N, I>
where
    I: SpIndex,
    N: Value + Default + Float,
{
    fn new(user_item: &CsMatI<N, I>,
           neighbors: I,
           correlation: Correlation) -> Self {
        let user_norm = match correlation {
            Correlation::Cosine => user_item.row_normalize(),
            Correlation::Pearson => user_item.row_center().row_normalize(),
        };

        UserUser {
            user_user: (&user_norm * &user_norm.transpose_view()).row_top_n(neighbors,true),
            neighbors,
            correlation,
        }
    }
}
