use super::Correlation;

trait Predictor<N, I> {

    fn new(user_item: &CsMatI<N, I>,
          neighbors: I,
          correlation: Correlation) -> Self;

    fn predict(&self, user: &I, item: &I) -> N;

    fn predict_user(&self, user: &I, items: &[I]) -> N;
}