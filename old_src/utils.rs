use std::{time::{SystemTime, UNIX_EPOCH}, fmt::Display};
use lazy_static::lazy_static;

lazy_static! {
    static ref PROGRAM_START_TIME: SystemTime = SystemTime::now();
}
fn elapsed_milliseconds() -> u64 {
    let elapsed_time = PROGRAM_START_TIME.elapsed().expect("Time went backwards");
    elapsed_time.as_secs() * 1000 + u64::from(elapsed_time.subsec_millis())
}

pub fn log(logtype: &str, text: impl Display) -> () {
    let space = 6;
    let elapsed = elapsed_milliseconds();
    let elapsed_space = elapsed.to_string().len();
    let space_left = space - elapsed_space;

    println!("{elapsed}{} [{logtype}] {text}", " ".repeat(space_left));
}
pub fn log_br() -> () {
    println!("------------------------------------------------------------");
}