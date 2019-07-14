use sprs::{CsMatI, CsVec};

use collaborative::{correlation, SimType};
use collaborative::CsVecBaseExt;

mod collaborative;

fn main() {
    let a = CsMatI::new_csc((3, 3),
                           vec![0u32, 2u32, 4u32, 5u32],
                           vec![0u32, 1u32, 0u32, 2u32, 2u32],
                           vec![1f64, 2f64, 3f64, 4f64, 5f64]);

    let eye = CsMatI::eye(3);
    let _b = &a * &eye;
   // let c = a.col_sum();

    let x = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
    println!("{}", x.sum());
    println!("{}", x.avg());
//    println!("{}", a.row_avg().sum());
    let _a_cor = correlation(&a, SimType::ItemItem);
}