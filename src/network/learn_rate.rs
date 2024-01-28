use rand_distr::num_traits::Float;

/// (Start, end, current)
pub struct LearnRate(f64, f64, f64);

impl LearnRate {
    pub fn new(start: f64, end: f64) -> Self {
        Self(start, end, start)
    }

    /// Desmos for the win I made a function to linearly decrease LR
    /// y\ =\ x\frac{s-v}{-l}+s
    pub fn set(&mut self, current_iteration: usize, max_iterations: usize) -> () {
        self.2 = current_iteration as f64 * ((self.0 - self.1) / - (max_iterations as f64)) + self.0
    }

    pub fn get(&self) -> f64 {
        self.2
    }
}

// For the #[serde(skip)] macro
impl Default for LearnRate {
    fn default() -> Self {
        Self::new(0.01, 0.01)
    }
}
