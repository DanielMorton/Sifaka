use sprs::{CsMatI, CsVec};

use collaborative::{correlation, CsVecBaseExt};
use collaborative::Correlation::Cosine;
use collaborative::RecommenderType::ItemItem;

mod collaborative;

fn main() {
    let a = CsMatI::new_csc((3, 3),
                           vec![0usize, 2usize, 4usize, 5usize],
                           vec![0usize, 1usize, 0usize, 2usize, 2usize],
                           vec![1f64, 2f64, 3f64, 4f64, 5f64]);

    let eye = CsMatI::eye(3);
    let _b = &a * &eye;
   // let c = a.col_sum();

    let x = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
    println!("{}", x.sum());
    println!("{}", x.avg());
//    println!("{}", a.row_avg().sum());
}