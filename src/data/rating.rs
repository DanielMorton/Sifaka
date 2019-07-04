pub trait Rating<U, I> {
    fn get_user(&self) -> &U {
        &self.user
    }
    fn get_item(&self) -> &I {
        &self.item
    }
    fn get_rating(&self) -> f64 {
        self.rating
    }
}

struct AbstractRating<U, I> {
    user: U,
    item: I,
    rating: f64
}

impl Rating<U, I> for AbstractRating<U, I> {}


pub type BasicRating = AbstractRating<i32, i32>;

pub type NamedRating = AbstractRating<String, String>;