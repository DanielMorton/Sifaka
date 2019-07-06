mod collaborative;
use collaborative::CsMatExt;
use collaborative::CsVecExt;

use sprs::{CsMat, CsVec};
fn main() {
    let a = CsMat::new_csc((3, 3),
                           vec![0, 2, 4, 5],
                           vec![0, 1, 0, 2, 2],
                           vec![1, 2, 3, 4, 5]);

    let eye = CsMat::eye(3);
    let _b = &a * &eye;
    let _c = a.col_sum();

    let x = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
    println!("{}", x.sum());
    println!("{}", x.avg())
}