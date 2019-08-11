use super::Correlation;

struct UserUser<N, I> {
    neighbors: usize,
    correlation: Correlation,
    user_user: CsMat<N, I>
}

