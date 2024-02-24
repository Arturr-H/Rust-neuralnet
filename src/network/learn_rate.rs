use std::ops::Range;

use rand_distr::num_traits::Float;

/// (Start, end, current, step)
pub struct LearnRate(f64, f64, f64, f64);

impl LearnRate {
    pub fn new(start: f64) -> Self {
        Self(start, start, start, 1.0)
    }
    pub fn new_range(range: Range<f64>) -> Self {
        Self(range.start, range.end, range.start, 1.0)
    }
    pub fn new_range_with_step(range: Range<f64>, step: f64) -> Self {
        assert!(step > 0.0);
        Self(range.start, range.end, range.start, step)
    }

    /// Desmos for the win I made a function to linearly decrease LR
    // y\ =\ \max\left(x\frac{\left(s-v\right)\cdot t}{-l}+s,\ v\right)
    pub fn set(&mut self, current_iteration: usize, max_iterations: usize) -> () {
        let delta_start_stop = self.0 - self.1;
        self.2 = f64::max(current_iteration as f64 * ((delta_start_stop * self.3) / - (max_iterations as f64)) + self.0, self.1)
    }

    pub fn get(&self) -> f64 {
        self.2
    }
}

// For the #[serde(skip)] macro
impl Default for LearnRate {
    fn default() -> Self {
        Self::new(0.01)
    }
}
