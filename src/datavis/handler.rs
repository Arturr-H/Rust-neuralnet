use std::{fs, io::Write};

const PATH: &'static str = "./data/";
pub struct Datavis;

impl Datavis {
    pub fn write<C: Iterator<Item = impl ToString>>(name: &str, contents: C) -> () {
        let fullpath = PATH.to_string() + name;

        let mut end = String::new();
        for a in contents.into_iter() {
            end.push_str(&a.to_string());
            end.push('\n');
        }

        std::fs::write(fullpath, end.as_bytes()).unwrap();
    }
}

