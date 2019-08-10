trait Predictor<N, I> {

    fn predict(&self, user: &I, item: &I) -> N;

    fn predict_user(&self, user: &I, items: &[I]) -> N;
}