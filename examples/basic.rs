use hfs::prelude::*;

fn main() {
    dbg!(hfs::utils::Compare::new(&"{{}, {}}".parse().unwrap()).eq(&"{{}, {}}".parse().unwrap()));
}
