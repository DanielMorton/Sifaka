use super::Correlation;
use super::Predictor;

struct UserUser<N, I> {
    neighbors: usize,
    correlation: Correlation,
    user_user: CsMatI<N, I>
}

impl<N, I> Predictor<N, I> for UserUser<N, I>
 {

    fn new(user_item: &CsMatI<N, I>,
           neighbors: I,
           correlation: Correlation) -> Self {

    }
}

