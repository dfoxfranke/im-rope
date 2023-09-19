//! Test strategies for ropes

use super::Rope;
use ::proptest::prelude::*;
use prop::strategy::ValueTree;

/// Value tree for ropes
pub struct RopeValueTree {
    cur_cut_start: usize,
    max_cut_start: usize,
    cur_cut_end: usize,
    max_cut_end: usize,
    tree: prop::string::RegexGeneratorValueTree<String>,
}

impl ValueTree for RopeValueTree {
    type Value = Rope;

    fn current(&self) -> Self::Value {
        let string = self.tree.current();
        let mut rope = Rope::from(&string);
        let mut cut_start = self.cur_cut_start;
        let mut cut_end = self.cur_cut_end;

        cut_end = std::cmp::min(cut_end, string.len());
        cut_start = std::cmp::min(cut_start, cut_end);

        unsafe {
            let mut v = rope.as_mut_vector();
            let post = v.split_off(cut_end);
            let mid = v.split_off(cut_start);
            v.append(mid);
            v.append(post);
        }

        rope
    }

    fn simplify(&mut self) -> bool {
        if self.tree.simplify() {
            self.max_cut_start = self.cur_cut_start;
            self.cur_cut_start /= 2;
            self.max_cut_end = self.cur_cut_end;
            self.cur_cut_end /= 2;
            true
        } else {
            false
        }
    }

    fn complicate(&mut self) -> bool {
        if self.tree.complicate() {
            self.cur_cut_start += (self.max_cut_start - self.cur_cut_start) / 2;
            self.cur_cut_end += (self.max_cut_end - self.cur_cut_end) / 2;
            true
        } else {
            false
        }
    }
}

/// Provides default parameters for `Rope`'s `Arbitrary` instance.
///
/// Arbitrary ropes can range up to 128 characters to allow testing
/// with up to a few chunks.
#[derive(Debug)]
pub struct RopeParam(prop::string::RegexGeneratorStrategy<String>);

impl Default for RopeParam {
    fn default() -> Self {
        RopeParam(prop::string::string_regex(".{0,128}").unwrap())
    }
}

#[derive(Debug)]
/// Proptest strategy for generating ropes
pub struct RopeStrategy(pub(crate) RopeParam);

impl Strategy for RopeStrategy {
    type Tree = RopeValueTree;
    type Value = Rope;

    fn new_tree(
        &self,
        runner: &mut ::proptest::test_runner::TestRunner,
    ) -> ::proptest::strategy::NewTree<Self> {
        let tree = self.0 .0.new_tree(runner)?;
        let cut_end = runner.rng().gen_range(0usize..=tree.current().len());
        let cut_start = runner.rng().gen_range(0usize..=cut_end);
        Ok(RopeValueTree {
            cur_cut_start: cut_start,
            max_cut_start: cut_start,
            cur_cut_end: cut_end,
            max_cut_end: cut_end,
            tree,
        })
    }
}

/// Returns a strategy for generating ropes matching `regex`.
/// 
/// # Panics
/// 
/// Panics if the regex is invalid.
#[must_use]
pub fn rope(regex: &str) -> RopeStrategy {
    RopeStrategy(RopeParam(prop::string::string_regex(regex).unwrap()))
}
