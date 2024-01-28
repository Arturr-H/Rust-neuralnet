use std::{time::{SystemTime, UNIX_EPOCH}, fmt::Display};
use lazy_static::lazy_static;
use rand_distr::num_traits::Signed;

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

#[derive(Clone, Copy)]
pub enum Color {
    Red, Green, Yellow, Blue, Magenta,
    Cyan, White, Black, BrightRed, BrightGreen,
}
#[derive(Clone, Copy)]
pub enum Style {
    Reset, Bold, Dim, Italic,
    Underline, Blink, Reverse, Hidden,
}

pub fn colorize(color: Color, text: impl Display) -> String {
    let color_code = match color {
        Color::Red => "\x1b[91m",
        Color::Green => "\x1b[92m",
        Color::Yellow => "\x1b[93m",
        Color::Blue => "\x1b[94m",
        Color::Magenta => "\x1b[95m",
        Color::Cyan => "\x1b[96m",
        Color::White => "\x1b[97m",
        Color::Black => "\x1b[30m",
        Color::BrightRed => "\x1b[91;1m",
        Color::BrightGreen => "\x1b[92;1m",
    };

    format!("{}{}{}", color_code, text, "\x1b[0m")
}

pub fn stylize(style: Style, text: impl ToString) -> String {
    let (start, end) = match style {
        Style::Reset => ("\x1b[0m", ""),
        Style::Bold => ("\x1b[1m", "\x1b[22m"),
        Style::Dim => ("\x1b[2m", "\x1b[22m"),
        Style::Italic => ("\x1b[3m", "\x1b[23m"),
        Style::Underline => ("\x1b[4m", "\x1b[24m"),
        Style::Blink => ("\x1b[5m", "\x1b[25m"),
        Style::Reverse => ("\x1b[7m", "\x1b[27m"),
        Style::Hidden => ("\x1b[8m", "\x1b[28m"),
    };

    let a = text.to_string().replace("\x1b[0m", "");

    format!("{}{}{}{}", start, a, end, "\x1b[0m")
}

pub fn display_array(a: &Vec<f64>) -> String {
    a.iter().map(format_f64).collect::<String>()
}

pub fn format_f64(f: &f64) -> String {
    if f.is_sign_negative() {
        format!("{:.2}", f)
    }else {
        format!("{:.3}", f)
    }
}
