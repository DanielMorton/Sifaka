use num_traits::Float;
use sprs::{CsMatI, SpIndex};

use super::{Correlation, CsMatBaseExt, CsMatFloat, Predictor, Value};

struct ItemItem<N, I>
where
    I: SpIndex,
{
    item_item: CsMatI<N, I>,
    neighbors: I,
    correlation: Correlation,
}
