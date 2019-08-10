trait Recommender<N, I> {

    fn recommend(&self, user: &I, size: usize) -> Vec<I>;

}