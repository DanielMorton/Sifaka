#[macro_use]
extern crate lazy_static;

use sprs::{CsMatI, CsVec};

use collaborative::{CsMatBaseExt, CsVecBaseExt};
use collaborative::Correlation::Cosine;
use collaborative::RecommenderType::ItemItem;

mod collaborative;

fn main() {
    let a = CsMatI::new_csc((4, 3),
                           vec![0i32, 2, 4, 5],
                           vec![0, 1, 0, 3, 3],
                           vec![1f64, 2f64, 3f64, 4f64, 5f64]);

    let eye = CsMatI::eye(3);
    let _b = &a * &eye;
   // let c = a.col_sum();

    let x = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
    println!("{}", x.sum());
    println!("{}", x.avg());

    let c = a.col_sum();
    let r = a.row_sum();
    c.data_iter().for_each(|x| print!("{}", x));
    println!();
    r.data_iter().for_each(|x| print!("{}", x))

//    println!("{}", a.row_avg().sum());
}