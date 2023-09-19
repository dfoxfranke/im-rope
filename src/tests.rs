use std::fmt::Write;

use super::*;
use crate::pattern::tests::{string_pattern, CharPatternStrategy};
use crate::proptest::rope;
use crate::test_utils::{Stream, StreamStrategy};
use ::proptest::prelude::*;

const ANY_STRING: &str = ".{0,128}";

#[derive(Debug, Clone)]
enum ABytesLike {
    Str(String),
    String(String),
    StringRef(String),
    Rope(Rope),
    RopeRef(Rope),
    Char(char),
    CharRef(char),
    Slice(Vec<u8>),
    Vec(Vec<u8>),
    VecRef(Vec<u8>),
    Vector(Vector<u8>),
    VectorRef(Vector<u8>),
}

impl std::fmt::Display for ABytesLike {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ABytesLike::Str(s) | ABytesLike::String(s) | ABytesLike::StringRef(s) => {
                f.write_str(s.as_str())
            }
            ABytesLike::Rope(r) | ABytesLike::RopeRef(r) => f.write_str(String::from(r).as_str()),
            ABytesLike::Char(ch) | ABytesLike::CharRef(ch) => f.write_char(*ch),
            ABytesLike::Slice(v) | ABytesLike::Vec(v) | ABytesLike::VecRef(v) => {
                f.write_str(std::str::from_utf8(v.as_slice()).unwrap())
            }
            ABytesLike::Vector(v) | ABytesLike::VectorRef(v) => f.write_str(
                String::from_utf8(v.iter().copied().collect())
                    .unwrap()
                    .as_str(),
            ),
        }
    }
}

impl BytesLike for &ABytesLike {
    fn into_vector(self) -> Vector<u8> {
        match self {
            ABytesLike::Str(s) => s.as_str().into_vector(),
            ABytesLike::String(s) => s.clone().into_vector(),
            ABytesLike::StringRef(s) => s.into_vector(),
            ABytesLike::Rope(r) => r.clone().into_vector(),
            ABytesLike::RopeRef(r) => r.into_vector(),
            ABytesLike::Char(ch) => ch.into_vector(),
            ABytesLike::CharRef(ch) => (*ch).into_vector(),
            ABytesLike::Slice(v) => v.as_slice().into_vector(),
            ABytesLike::Vec(v) => v.clone().into_vector(),
            ABytesLike::VecRef(v) => v.into_vector(),
            ABytesLike::Vector(v) => v.clone().into_vector(),
            ABytesLike::VectorRef(v) => v.into_vector(),
        }
    }
}

enum BytesLikeValueTree {
    Str(prop::string::RegexGeneratorValueTree<String>),
    String(prop::string::RegexGeneratorValueTree<String>),
    StringRef(prop::string::RegexGeneratorValueTree<String>),
    Rope(prop::string::RegexGeneratorValueTree<String>),
    RopeRef(prop::string::RegexGeneratorValueTree<String>),
    Char(prop::char::CharValueTree),
    CharRef(prop::char::CharValueTree),
    Slice(prop::string::RegexGeneratorValueTree<String>),
    Vec(prop::string::RegexGeneratorValueTree<String>),
    VecRef(prop::string::RegexGeneratorValueTree<String>),
    Vector(prop::string::RegexGeneratorValueTree<String>),
    VectorRef(prop::string::RegexGeneratorValueTree<String>),
}

impl prop::strategy::ValueTree for BytesLikeValueTree {
    type Value = ABytesLike;

    fn current(&self) -> Self::Value {
        match self {
            BytesLikeValueTree::Str(s) => ABytesLike::Str(s.current()),
            BytesLikeValueTree::String(s) => ABytesLike::String(s.current()),
            BytesLikeValueTree::StringRef(s) => ABytesLike::StringRef(s.current()),
            BytesLikeValueTree::Rope(r) => ABytesLike::Rope(Rope::from(r.current())),
            BytesLikeValueTree::RopeRef(r) => ABytesLike::RopeRef(Rope::from(r.current())),
            BytesLikeValueTree::Char(ch) => ABytesLike::Char(ch.current()),
            BytesLikeValueTree::CharRef(ch) => ABytesLike::CharRef(ch.current()),
            BytesLikeValueTree::Slice(v) => ABytesLike::Slice(v.current().into()),
            BytesLikeValueTree::Vec(v) => ABytesLike::Vec(v.current().into()),
            BytesLikeValueTree::VecRef(v) => ABytesLike::VecRef(v.current().into()),
            BytesLikeValueTree::Vector(v) => {
                ABytesLike::Vector(Vec::<u8>::from(v.current()).into())
            }
            BytesLikeValueTree::VectorRef(v) => {
                ABytesLike::VectorRef(Vec::<u8>::from(v.current()).into())
            }
        }
    }

    fn simplify(&mut self) -> bool {
        match self {
            BytesLikeValueTree::Str(s)
            | BytesLikeValueTree::String(s)
            | BytesLikeValueTree::StringRef(s)
            | BytesLikeValueTree::Rope(s)
            | BytesLikeValueTree::RopeRef(s)
            | BytesLikeValueTree::Slice(s)
            | BytesLikeValueTree::Vec(s)
            | BytesLikeValueTree::VecRef(s)
            | BytesLikeValueTree::Vector(s)
            | BytesLikeValueTree::VectorRef(s) => s.simplify(),
            BytesLikeValueTree::Char(ch) | BytesLikeValueTree::CharRef(ch) => ch.simplify(),
        }
    }

    fn complicate(&mut self) -> bool {
        match self {
            BytesLikeValueTree::Str(s)
            | BytesLikeValueTree::String(s)
            | BytesLikeValueTree::StringRef(s)
            | BytesLikeValueTree::Rope(s)
            | BytesLikeValueTree::RopeRef(s)
            | BytesLikeValueTree::Slice(s)
            | BytesLikeValueTree::Vec(s)
            | BytesLikeValueTree::VecRef(s)
            | BytesLikeValueTree::Vector(s)
            | BytesLikeValueTree::VectorRef(s) => s.complicate(),
            BytesLikeValueTree::Char(ch) | BytesLikeValueTree::CharRef(ch) => ch.complicate(),
        }
    }
}

#[derive(Debug)]
struct BytesLikeStrategy<'a> {
    string_strategy: prop::string::RegexGeneratorStrategy<String>,
    char_strategy: prop::char::CharStrategy<'a>,
}

fn bytes_like<'a>(
    regex: &str,
    char_strategy: prop::char::CharStrategy<'a>,
) -> BytesLikeStrategy<'a> {
    BytesLikeStrategy {
        string_strategy: prop::string::string_regex(regex).unwrap(),
        char_strategy,
    }
}

impl<'a> Strategy for BytesLikeStrategy<'a> {
    type Tree = BytesLikeValueTree;

    type Value = ABytesLike;

    fn new_tree(
        &self,
        runner: &mut prop::test_runner::TestRunner,
    ) -> prop::strategy::NewTree<Self> {
        Ok(match runner.rng().gen_range(0..12) {
            0 => BytesLikeValueTree::Str(self.string_strategy.new_tree(runner)?),
            1 => BytesLikeValueTree::String(self.string_strategy.new_tree(runner)?),
            2 => BytesLikeValueTree::StringRef(self.string_strategy.new_tree(runner)?),
            3 => BytesLikeValueTree::Rope(self.string_strategy.new_tree(runner)?),
            4 => BytesLikeValueTree::RopeRef(self.string_strategy.new_tree(runner)?),
            5 => BytesLikeValueTree::Char(self.char_strategy.new_tree(runner)?),
            6 => BytesLikeValueTree::CharRef(self.char_strategy.new_tree(runner)?),
            7 => BytesLikeValueTree::Slice(self.string_strategy.new_tree(runner)?),
            8 => BytesLikeValueTree::Vec(self.string_strategy.new_tree(runner)?),
            9 => BytesLikeValueTree::VecRef(self.string_strategy.new_tree(runner)?),
            10 => BytesLikeValueTree::Vector(self.string_strategy.new_tree(runner)?),
            11 => BytesLikeValueTree::VectorRef(self.string_strategy.new_tree(runner)?),
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, Clone)]
enum AStrLike {
    Str(String),
    String(String),
    StringRef(String),
    Rope(Rope),
    RopeRef(Rope),
    Char(char),
    CharRef(char),
}

impl std::fmt::Display for AStrLike {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AStrLike::Str(s) | AStrLike::String(s) | AStrLike::StringRef(s) => {
                f.write_str(s.as_str())
            }
            AStrLike::Rope(r) | AStrLike::RopeRef(r) => f.write_str(String::from(r).as_str()),
            AStrLike::Char(ch) | AStrLike::CharRef(ch) => f.write_char(*ch),
        }
    }
}

impl BytesLike for &AStrLike {
    fn into_vector(self) -> Vector<u8> {
        match self {
            AStrLike::Str(s) | AStrLike::String(s) | AStrLike::StringRef(s) => {
                s.as_str().into_vector()
            }
            AStrLike::Rope(r) | AStrLike::RopeRef(r) => r.into_vector(),
            AStrLike::Char(ch) | AStrLike::CharRef(ch) => ch.into_vector(),
        }
    }
}

unsafe impl StrLike for &AStrLike {}

enum StrLikeValueTree {
    Str(prop::string::RegexGeneratorValueTree<String>),
    String(prop::string::RegexGeneratorValueTree<String>),
    StringRef(prop::string::RegexGeneratorValueTree<String>),
    Rope(prop::string::RegexGeneratorValueTree<String>),
    RopeRef(prop::string::RegexGeneratorValueTree<String>),
    Char(prop::char::CharValueTree),
    CharRef(prop::char::CharValueTree),
}

impl prop::strategy::ValueTree for StrLikeValueTree {
    type Value = AStrLike;

    fn current(&self) -> Self::Value {
        match self {
            StrLikeValueTree::Str(s) => AStrLike::Str(s.current()),
            StrLikeValueTree::String(s) => AStrLike::String(s.current()),
            StrLikeValueTree::StringRef(s) => AStrLike::StringRef(s.current()),
            StrLikeValueTree::Rope(r) => AStrLike::Rope(Rope::from(r.current())),
            StrLikeValueTree::RopeRef(r) => AStrLike::RopeRef(Rope::from(r.current())),
            StrLikeValueTree::Char(ch) => AStrLike::Char(ch.current()),
            StrLikeValueTree::CharRef(ch) => AStrLike::CharRef(ch.current()),
        }
    }

    fn simplify(&mut self) -> bool {
        match self {
            StrLikeValueTree::Str(s)
            | StrLikeValueTree::String(s)
            | StrLikeValueTree::StringRef(s)
            | StrLikeValueTree::Rope(s)
            | StrLikeValueTree::RopeRef(s) => s.simplify(),
            StrLikeValueTree::Char(ch) | StrLikeValueTree::CharRef(ch) => ch.simplify(),
        }
    }

    fn complicate(&mut self) -> bool {
        match self {
            StrLikeValueTree::Str(s)
            | StrLikeValueTree::String(s)
            | StrLikeValueTree::StringRef(s)
            | StrLikeValueTree::Rope(s)
            | StrLikeValueTree::RopeRef(s) => s.complicate(),
            StrLikeValueTree::Char(ch) | StrLikeValueTree::CharRef(ch) => ch.complicate(),
        }
    }
}

#[derive(Debug)]
struct StrLikeStrategy<'a> {
    string_strategy: prop::string::RegexGeneratorStrategy<String>,
    char_strategy: prop::char::CharStrategy<'a>,
}

fn str_like<'a>(regex: &str, char_strategy: prop::char::CharStrategy<'a>) -> StrLikeStrategy<'a> {
    StrLikeStrategy {
        string_strategy: prop::string::string_regex(regex).unwrap(),
        char_strategy,
    }
}

impl<'a> Strategy for StrLikeStrategy<'a> {
    type Tree = StrLikeValueTree;

    type Value = AStrLike;

    fn new_tree(
        &self,
        runner: &mut prop::test_runner::TestRunner,
    ) -> prop::strategy::NewTree<Self> {
        Ok(match runner.rng().gen_range(0..7) {
            0 => StrLikeValueTree::Str(self.string_strategy.new_tree(runner)?),
            1 => StrLikeValueTree::String(self.string_strategy.new_tree(runner)?),
            2 => StrLikeValueTree::StringRef(self.string_strategy.new_tree(runner)?),
            3 => StrLikeValueTree::Rope(self.string_strategy.new_tree(runner)?),
            4 => StrLikeValueTree::RopeRef(self.string_strategy.new_tree(runner)?),
            5 => StrLikeValueTree::Char(self.char_strategy.new_tree(runner)?),
            6 => StrLikeValueTree::CharRef(self.char_strategy.new_tree(runner)?),
            _ => unreachable!(),
        })
    }
}

fn check_iterator_contract<F, I>(gen: &mut F) -> Result<(), TestCaseError>
where
    I: FusedIterator,
    I::Item: std::fmt::Debug + Eq,
    F: FnMut() -> I,
{
    let mut iter = gen();
    let (mut min, mut max) = iter.size_hint();
    let mut cur: Option<I::Item> = None;

    while let Some(new) = iter.next() {
        let (new_min, maybe_new_max) = iter.size_hint();
        cur = Some(new);

        min = std::cmp::max(min.saturating_sub(1), new_min);

        match (max, maybe_new_max) {
            (Some(old_max), Some(new_max)) => {
                prop_assert_ne!(old_max, 0);
                max = Some(std::cmp::min(old_max - 1, new_max));
            }
            (Some(old_max), None) => {
                prop_assert_ne!(old_max, 0);
                max = Some(old_max - 1);
            }
            (None, Some(new_max)) => {
                max = Some(new_max);
            }
            (None, None) => {}
        }
    }

    prop_assert_eq!(iter.next(), None);
    prop_assert_eq!(iter.next(), None);
    prop_assert_eq!(iter.next(), None);

    prop_assert_eq!(cur, gen().last());

    Ok(())
}

fn check_double_ended_iterator_equivalence<I1, I2>(
    mut iter1: I1,
    mut iter2: I2,
    bools: &mut Stream<prop::bool::Any>,
) -> Result<(), TestCaseError>
where
    I1: DoubleEndedIterator,
    I2: DoubleEndedIterator,
    I1::Item: std::fmt::Debug + PartialEq<I2::Item>,
    I2::Item: std::fmt::Debug,
{
    loop {
        if bools.gen() {
            let item1 = iter1.next();
            let item2 = iter2.next();
            if item1.is_none() {
                prop_assert!(item2.is_none());
                return Ok(());
            }
            prop_assert!(item2.is_some());
            prop_assert_eq!(item1.unwrap(), item2.unwrap());
        } else {
            let item1 = iter1.next_back();
            let item2 = iter2.next_back();
            if item1.is_none() {
                prop_assert!(item2.is_none());
                return Ok(());
            }
            prop_assert!(item2.is_some());
            prop_assert_eq!(item1.unwrap(), item2.unwrap());
        }
    }
}

fn check_iterator_into_owning<F, I>(gen: &mut F) -> Result<(), TestCaseError>
where
    I: FusedIterator + IntoOwning,
    I::Owning: FusedIterator<Item = I::Item>,
    I::Item: std::fmt::Debug + Eq,
    F: FnMut() -> I,
{
    check_iterator_contract(gen)?;
    prop_assert!(gen().eq(gen().to_owning()));
    prop_assert!(gen().eq(gen().into_owning()));
    prop_assert!(gen().to_owning().eq(gen().into_owning()));
    Ok(())
}

fn check_double_ended_iterator_into_owning<F, I>(
    gen: &mut F,
    bools: &mut Stream<prop::bool::Any>,
) -> Result<(), TestCaseError>
where
    I: DoubleEndedIterator + FusedIterator + IntoOwning,
    I::Owning: DoubleEndedIterator + FusedIterator<Item = I::Item>,
    I::Item: std::fmt::Debug + Eq,
    F: FnMut() -> I,
{
    check_iterator_contract(gen)?;
    check_double_ended_iterator_equivalence(gen(), gen().to_owning(), bools)?;
    check_double_ended_iterator_equivalence(gen(), gen().into_owning(), bools)?;
    check_double_ended_iterator_equivalence(gen().to_owning(), gen().into_owning(), bools)?;
    Ok(())
}

proptest! {
    #[test]
    fn rope_string_inv(rope in any::<Rope>()) {
        prop_assert_eq!(rope.clone(), Rope::from(String::from(rope)));
    }

    #[test]
    fn string_rope_inv(string in ANY_STRING) {
        prop_assert_eq!(string.clone(), String::from(Rope::from(string)));
    }

    #[test]
    fn rope_cmp_agrees_string(rope1 in any::<Rope>(), rope2 in any::<Rope>()) {
        prop_assert_eq!(rope1.cmp(&rope2), String::from(&rope1).cmp(&String::from(&rope2)));
    }

    #[test]
    fn rope_eq_agrees_string(rope1 in any::<Rope>(), rope2 in any::<Rope>()) {
        prop_assert_eq!(rope1.eq(&rope2), String::from(&rope1).eq(&String::from(&rope2)));
    }

    #[test]
    fn rope_eq_cmp(rope1 in any::<Rope>(), rope2 in any::<Rope>()) {
        prop_assert_eq!(rope1 == rope2, rope1.cmp(&rope2) == Ordering::Equal);
    }

    #[test]
    fn rope_cmp_string(rope1 in any::<Rope>(), rope2 in any::<Rope>()) {
        prop_assert_eq!(rope1.partial_cmp(&rope2), rope1.partial_cmp(&String::from(&rope2)));
        prop_assert_eq!(rope1.partial_cmp(&rope2), String::from(&rope1).partial_cmp(&rope2));
    }

    #[test]
    fn rope_eq_string(rope1 in any::<Rope>(), rope2 in any::<Rope>()) {
        prop_assert_eq!(rope1.eq(&rope2), rope1.eq(&String::from(&rope2)));
        prop_assert_eq!(rope1.eq(&rope2), String::from(&rope1).eq(&rope2));
    }

    #[test]
    fn is_empty_agrees_string(rope in any::<Rope>()) {
        prop_assert_eq!(rope.is_empty(), String::from(&rope).is_empty());
    }


    #[test]
    fn len_agrees_string(rope in any::<Rope>()) {
        prop_assert_eq!(rope.len(), String::from(&rope).len());
    }

    #[test]
    fn clear_empties(mut rope in any::<Rope>()) {
        rope.clear();
        prop_assert!(rope.is_empty());
    }

    fn bytes_iterator(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_into_owning(&mut || rope.bytes(), &mut bools)?;
    }

    #[test]
    fn bytes_agrees_string(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_equivalence(
            rope.bytes(),
            String::from(&rope).bytes(),
            &mut bools
        )?;
    }


    #[test]
    fn into_bytes_iterator(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_into_owning(
            &mut || rope.clone().into_bytes(),
            &mut bools
        )?;
    }

    #[test]
    fn into_bytes_agrees_string(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_equivalence(
            rope.clone().into_bytes(),
            String::from(&rope).bytes(),
            &mut bools
        )?;
    }

    #[test]
    fn chars_iterator(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_into_owning(&mut || rope.chars(), &mut bools)?;
    }

    #[test]
    fn chars_agrees_string(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_equivalence(
            rope.chars(),
            String::from(&rope).chars(),
            &mut bools
        )?;
    }


    #[test]
    fn into_chars_iterator(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_into_owning(
            &mut || rope.clone().into_chars(),
            &mut bools
        )?;
    }

    #[test]
    fn into_chars_agrees_string(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_equivalence(
            rope.clone().into_chars(),
            String::from(&rope).chars(),
            &mut bools
        )?;
    }

    #[test]
    fn char_indices_iterator(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_into_owning(&mut || rope.char_indices(), &mut bools)?;
    }

    #[test]
    fn char_indices_agrees_string(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_equivalence(
            rope.char_indices(),
            String::from(&rope).char_indices(),
            &mut bools
        )?;
    }


    #[test]
    fn into_char_indices_iterator(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_into_owning(
            &mut || rope.clone().into_char_indices(),
            &mut bools
        )?;
    }

    #[test]
    fn into_char_indices_agrees_string(rope in any::<Rope>(), mut bools in StreamStrategy(any::<bool>())) {
        check_double_ended_iterator_equivalence(
            rope.clone().into_char_indices(),
            String::from(&rope).char_indices(),
            &mut bools
        )?;
    }

    #[test]
    fn chunk_iterator(rope in any::<Rope>()) {
        check_iterator_contract(&mut || rope.chunks())?;
    }

    #[test]
    fn is_char_boundary_agrees_string(rope in any::<Rope>()) {
        let string = String::from(&rope);
        for i in 0..=rope.len() {
            prop_assert_eq!(rope.is_char_boundary(i), string.is_char_boundary(i));
        }
    }

    #[test]
    fn find_char_iterator(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        check_double_ended_iterator_into_owning(
            &mut || haystack.find_all(&needle),
            &mut bools
        )?;
    }

    #[test]
    fn find_char_agrees_string(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        let pred = needle.to_pred();
        check_double_ended_iterator_equivalence(
            haystack.find_all(&needle).map(|(range, _)| range),
            String::from(&haystack)
                .match_indices(pred)
                .map(|(index, str)| index .. index + str.len()),
            &mut bools
        )?;
    }

    #[test]
    fn rfind_char_iterator(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        check_double_ended_iterator_into_owning(
            &mut || haystack.rfind_all(&needle),
            &mut bools
        )?;
    }

    #[test]
    fn rfind_char_agrees_string(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        let pred = needle.to_pred();
        check_double_ended_iterator_equivalence(
            haystack.rfind_all(&needle).map(|(range, _)| range),
            String::from(&haystack)
                .rmatch_indices(pred)
                .map(|(index, str)| index .. index + str.len()),
            &mut bools
        )?;
    }

    #[test]
    fn find_str_iterator(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING)
    ) {
        check_iterator_into_owning(&mut || haystack.find_all(&needle))?;
    }

    #[test]
    fn find_str_agrees_string(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
    ) {
        let string = needle.to_string();
        prop_assert!(
            haystack
                .find_all(&needle)
                .map(|(range, _)| range)
                .eq(String::from(&haystack)
                    .match_indices(&string)
                    .map(|(index, str)| index .. index + str.len()))
        );
    }

    #[test]
    fn rfind_str_iterator(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING)
    ) {
        check_iterator_into_owning(&mut || haystack.rfind_all(&needle))?;
    }

    #[test]
    fn rfind_str_agrees_string(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
    ) {
        let string = needle.to_string();
        prop_assert!(
            haystack
                .rfind_all(&needle)
                .map(|(range, _)| range)
                .eq(String::from(&haystack)
                    .rmatch_indices(&string)
                    .map(|(index, str)| index .. index + str.len()))
        );
    }

    #[test]
    fn split_chars_iterator(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        check_double_ended_iterator_into_owning(
            &mut || haystack.split(&needle),
            &mut bools
        )?;
    }

    #[test]
    fn split_chars_agrees_string(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        let string = String::from(&haystack);
        let pred = needle.to_pred();
        check_double_ended_iterator_equivalence(
            haystack.split(&needle),
            string.split(&pred),
            &mut bools
        )?;
    }

    #[test]
    fn split_str_iterator(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING)
    ) {
        check_iterator_into_owning(&mut || haystack.split(&needle))?;
    }

    #[test]
    fn split_str_agrees_string(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
    ) {
        let string = String::from(&haystack);
        prop_assert!(
            haystack.split(&needle).eq(string.split(&needle.to_string()))
        );
    }

    #[test]
    fn splitn_chars_iterator(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        limit in 0usize..4,
    ) {
        check_iterator_into_owning(
            &mut || haystack.splitn(limit, &needle),
        )?;
    }

    #[test]
    fn splitn_chars_agrees_string(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        limit in 0usize..4,
    ) {
        let string = String::from(&haystack);
        prop_assert!(
            haystack.splitn(limit, &needle).eq(string.splitn(limit, &needle.to_pred()))
        );
    }

    #[test]
    fn splitn_str_iterator(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
        limit in 0usize..4,
    ) {
        check_iterator_into_owning(&mut || haystack.splitn(limit, &needle))?;
    }

    #[test]
    fn splitn_str_agrees_string(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
        limit in 0usize..4,
    ) {
        let string = String::from(&haystack);
        prop_assert!(
            haystack.splitn(limit, &needle).eq(string.splitn(limit, &needle.to_string()))
        );
    }

    #[test]
    fn split_terminator_chars_iterator(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        check_double_ended_iterator_into_owning(
            &mut || haystack.split_terminator(&needle),
            &mut bools
        )?;
    }

    #[test]
    fn split_terminator_chars_agrees_string(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        let string = String::from(&haystack);
        let pred = needle.to_pred();
        check_double_ended_iterator_equivalence(
            haystack.split_terminator(&needle),
            string.split_terminator(&pred),
            &mut bools
        )?;
    }

    #[test]
    fn split_terminator_str_iterator(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING)
    ) {
        check_iterator_into_owning(&mut || haystack.split_terminator(&needle))?;
    }

    #[test]
    fn split_terminator_str_agrees_string(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
    ) {
        let string = String::from(&haystack);
        prop_assert!(
            haystack.split_terminator(&needle).eq(string.split_terminator(&needle.to_string()))
        );
    }

    #[test]
    fn split_inclusive_chars_iterator(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        check_double_ended_iterator_into_owning(
            &mut || haystack.split_inclusive(&needle),
            &mut bools
        )?;
    }

    #[test]
    fn split_inclusive_chars_agrees_string(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        let string = String::from(&haystack);
        let pred = needle.to_pred();
        check_double_ended_iterator_equivalence(
            haystack.split_inclusive(&needle),
            string.split_inclusive(&pred),
            &mut bools
        )?;
    }

    #[test]
    fn split_inclusive_str_iterator(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING)
    ) {
        check_iterator_into_owning(&mut || haystack.split_inclusive(&needle))?;
    }

    #[test]
    fn split_inclusive_str_agrees_string(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
    ) {
        let string = String::from(&haystack);
        prop_assert!(
            haystack.split_inclusive(&needle).eq(string.split_inclusive(&needle.to_string()))
        );
    }

    #[test]
    fn rsplit_str_iterator(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING)
    ) {
        check_iterator_into_owning(&mut || haystack.rsplit(&needle))?;
    }

    #[test]
    fn rsplit_str_agrees_string(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
    ) {
        let string = String::from(&haystack);
        prop_assert!(
            haystack.rsplit(&needle).eq(string.rsplit(&needle.to_string()))
        );
    }

    #[test]
    fn rsplitn_chars_iterator(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        limit in 0usize..4,
    ) {
        check_iterator_into_owning(
            &mut || haystack.rsplitn(limit, &needle),
        )?;
    }

    #[test]
    fn rsplitn_chars_agrees_string(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        limit in 0usize..4,
    ) {
        let string = String::from(&haystack);
        prop_assert!(
            haystack.rsplitn(limit, &needle).eq(string.rsplitn(limit, &needle.to_pred()))
        );
    }

    #[test]
    fn rsplitn_str_iterator(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
        limit in 0usize..4,
    ) {
        check_iterator_into_owning(&mut || haystack.rsplitn(limit, &needle))?;
    }

    #[test]
    fn rsplitn_str_agrees_string(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
        limit in 0usize..4,
    ) {
        let string = String::from(&haystack);
        prop_assert!(
            haystack.rsplitn(limit, &needle).eq(string.rsplitn(limit, &needle.to_string()))
        );
    }

    #[test]
    fn rsplit_terminator_chars_iterator(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        check_double_ended_iterator_into_owning(
            &mut || haystack.rsplit_terminator(&needle),
            &mut bools
        )?;
    }

    #[test]
    fn rsplit_terminator_chars_agrees_string(
        haystack in any::<Rope>(),
        needle in CharPatternStrategy,
        mut bools in StreamStrategy(any::<bool>())
    ) {
        let string = String::from(&haystack);
        let pred = needle.to_pred();
        check_double_ended_iterator_equivalence(
            haystack.rsplit_terminator(&needle),
            string.rsplit_terminator(&pred),
            &mut bools
        )?;
    }

    #[test]
    fn rsplit_terminator_str_iterator(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING)
    ) {
        check_iterator_into_owning(&mut || haystack.rsplit_terminator(&needle))?;
    }

    #[test]
    fn rsplit_terminator_str_agrees_string(
        haystack in any::<Rope>(),
        needle in string_pattern(ANY_STRING),
    ) {
        let string = String::from(&haystack);
        prop_assert!(
            haystack
                .rsplit_terminator(&needle)
                .eq(string.rsplit_terminator(&needle.to_string()))
        );
    }

    #[test]
    fn lines_iterator(
        haystack in rope("(\r|\n|.){0,128}"),
        mut bools in StreamStrategy(any::<bool>()),
    ) {
        check_double_ended_iterator_into_owning(
            &mut || haystack.lines(),
            &mut bools
        )?;
    }

    #[test]
    fn lines_agrees_std(
        haystack in rope("(\r|\n|.){0,128}"),
        mut bools in StreamStrategy(any::<bool>()),
    ) {
        // Standard library has a bug here.
        prop_assume!(haystack.back() != Some('\r'));

        check_double_ended_iterator_equivalence(
            haystack.lines(),
            String::from(&haystack).lines(),
            &mut bools
        )?;
    }

    #[test]
    fn append_agrees_string(
        mut a in any::<Rope>(),
        b in str_like(ANY_STRING, any::<char>())
    ) {
        let mut string = String::from(&a);
        string.push_str(b.to_string().as_str());
        a.append(&b);
        prop_assert_eq!(a, string);
    }

    #[test]
    fn append_unchecked_agrees_string(
        mut a in any::<Rope>(),
        b in bytes_like(ANY_STRING, any::<char>())
    ) {
        let mut string = String::from(&a);
        string.push_str(b.to_string().as_str());
        unsafe { a.append_unchecked (&b); }
        prop_assert_eq!(a, string);
    }

    #[test]
    fn prepend_agrees_string(
        mut a in any::<Rope>(),
        b in str_like(ANY_STRING, any::<char>())
    ) {
        let mut string = b.to_string();
        string.push_str(String::from(&a).as_str());
        a.prepend(&b);
        prop_assert_eq!(a, string);
    }

    #[test]
    fn prepend_unchecked_agrees_string(
        mut a in any::<Rope>(),
        b in bytes_like(ANY_STRING, any::<char>())
    ) {
        let mut string = b.to_string();
        string.push_str(String::from(&a).as_str());
        unsafe { a.prepend_unchecked(&b); }
        prop_assert_eq!(a, string);
    }

    #[test]
    fn starts_with_rope(mut a in any::<Rope>(), b in any::<Rope>(), c in any::<char>()) {
        let rope = &a + &b;
        prop_assert!(rope.starts_with(&a));

        prop_assume!(b.front() != Some(c));
        a.append(c);
        prop_assert!(!rope.starts_with(&a));
    }

    #[test]
    fn starts_with_string(mut a in ANY_STRING, b in any::<Rope>(), c in any::<char>()) {
        let rope = Rope::from(&a) + &b;
        prop_assert!(rope.starts_with(&a));

        prop_assume!(b.front() != Some(c));
        a.push(c);
        prop_assert!(!rope.starts_with(&a));
    }

    #[test]
    fn starts_with_char(a in any::<char>(), b in any::<Rope>(), c in any::<char>()) {
        let mut rope = Rope::from(a) + &b;
        prop_assert!(rope.starts_with(a));

        prop_assume!(a != c);
        rope.push_front(c);
        prop_assert!(!rope.starts_with(a));
    }

    #[test]
    fn ends_with_rope(a in any::<Rope>(), mut b in any::<Rope>(), c in any::<char>()) {
        let rope = &a + &b;
        prop_assert!(rope.ends_with(&b));

        prop_assume!(a.back() != Some(c));
        b.prepend(c);
        prop_assert!(!rope.ends_with(&b));
    }

    #[test]
    fn ends_with_string(a in any::<Rope>(), mut b in ANY_STRING, c in any::<char>()) {
        let rope = &a + Rope::from(&b);
        prop_assert!(rope.ends_with(&b));

        prop_assume!(a.back() != Some(c));
        b.insert(0, c);
        prop_assert!(!rope.ends_with(&b));
    }

    #[test]
    fn ends_with_char(a in any::<Rope>(), b in any::<char>(), c in any::<char>()) {
        let mut rope = &a + Rope::from(b);
        prop_assert!(rope.ends_with(b));

        prop_assume!(b != c);
        rope.push_back(c);
        prop_assert!(!rope.ends_with(b));
    }

    #[test]
    fn vector_guard_clear_on_panic(mut rope in any::<Rope>()) {
        std::panic::catch_unwind(
            AssertUnwindSafe(|| unsafe {
                let _v = rope.as_mut_vector();
                std::panic::resume_unwind(Box::new(()));
            })
        ).unwrap_err();
        prop_assert!(rope.is_empty());
    }

    #[test]
    fn vector_guard_normal(mut rope in any::<Rope>()) {
        let orig = rope.clone();
        unsafe {
            let _v = rope.as_mut_vector();
        }
        prop_assert_eq!(rope, orig);
    }

    #[test]
    fn test_insertion(a in any::<Rope>(), b in any::<Rope>(), c in any::<Rope>()) {
        let mut rope_1 = &a + &c;
        rope_1.insert(a.len(), &b);
        let rope_2 = a + b + c;
        prop_assert_eq!(rope_1, rope_2);
    }

    #[test]
    fn test_unchecked_insertion(a in any::<Rope>(), b in any::<Rope>(), c in any::<Rope>()) {
        let mut rope_1 = &a + &c;
        unsafe {
            rope_1.insert_unchecked(a.len(), &b);

            let rope_2 = a + b + c;
            prop_assert_eq!(rope_1, rope_2);
        }
    }

    #[test]
    fn test_extraction(a in any::<Rope>(), b in any::<Rope>(), c in any::<Rope>()) {
        let mut rope_1 = &a + &b + &c;
        prop_assert_eq!(&rope_1.subrope(a.len() .. a.len() + b.len()), &b);
        prop_assert_eq!(&rope_1.extract(a.len() .. a.len() + b.len()), &b);

        let rope_2 = a + c;
        prop_assert_eq!(rope_1, rope_2);
    }

    #[test]
    fn test_unchecked_extraction(a in any::<Rope>(), b in any::<Rope>(), c in any::<Rope>()) {
        unsafe {
            let mut rope_1 = &a + &b + &c;
            prop_assert_eq!(&rope_1.subrope_unchecked(a.len() .. a.len() + b.len()), &b);
            prop_assert_eq!(&rope_1.extract_unchecked(a.len() .. a.len() + b.len()), &b);

            let rope_2 = a + c;
            prop_assert_eq!(rope_1, rope_2);
        }
    }

    #[test]
    fn test_split_at(a in any::<Rope>(), b in any::<Rope>()) {
        let rope = &a + &b;
        let (x, y) = rope.split_at(a.len());
        prop_assert_eq!(x, a);
        prop_assert_eq!(y, b);
    }

    #[test]
    fn test_split_at_unchecked(a in any::<Rope>(), b in any::<Rope>()) {
        unsafe {
            let rope = &a + &b;
            let (x, y) = rope.split_at_unchecked(a.len());
            prop_assert_eq!(x, a);
            prop_assert_eq!(y, b);
        }
    }

    #[test]
    fn test_push_pop(mut rope in rope(".{1,128}")) {
        let orig = rope.clone();
        let front = rope.front().unwrap();
        prop_assert_eq!(rope.pop_front().unwrap(), front);
        rope.push_front(front);
        prop_assert_eq!(&rope, &orig);

        let back = rope.back().unwrap();
        prop_assert_eq!(rope.pop_back().unwrap(), back);
        rope.push_back(back);
        prop_assert_eq!(&rope, &orig);
    }

    #[test]
    fn test_write_display_debug(string in ANY_STRING) {
        use std::fmt::Write;
        let mut rope = Rope::new();
        write!(rope, "{}", &string).unwrap();
        prop_assert_eq!(format!("{}", &rope), format!("{}", &string));
        prop_assert_eq!(format!("{:?}", &rope), format!("{:?}", &string));
    }

    #[test]
    fn insert_stable_under_failure(
        a in any::<Rope>(),
        b in prop::char::range('\u{80}', '\u{10ffff}'),
        c in any::<Rope>(),
        insert in str_like(ANY_STRING, any::<char>())
    ) {
        let mut rope = &a + Rope::from(b) + &c;
        let orig = rope.clone();

        let index = a.len() + 1;
        assert!(!rope.is_char_boundary(index));
        prop_assert!(rope.try_insert(index, &insert).is_err());
        prop_assert_eq!(rope, orig);
    }

    #[test]
    fn extract_stable_under_failure(
        a in any::<Rope>(),
        b in prop::char::range('\u{80}', '\u{10ffff}'),
        c in any::<Rope>(),
    ) {
        let mut rope = &a + Rope::from(b) + &c;
        let orig = rope.clone();

        let range = a.len() + 1 .. rope.len() - c.len();
        assert!(!rope.is_char_boundary(range.start));
        prop_assert!(rope.try_extract(range).is_err());
        prop_assert_eq!(&rope, &orig);

        #[allow(clippy::range_plus_one)]
        let range = a.len() .. a.len() + 1;
        assert!(!rope.is_char_boundary(range.end));
        prop_assert!(rope.try_extract(range).is_err());
        prop_assert_eq!(&rope, &orig);
    }


    #[test]
    fn serde_round_trip(rope in any::<Rope>()) {
        let serialized = serde_json::to_string(&rope).unwrap();
        let deserialized : Rope = serde_json::from_str(serialized.as_str()).unwrap();
        prop_assert_eq!(deserialized, rope);
    }

    #[test]
    fn from_utf8_agrees_string(bytes in prop::collection::vec(any::<u8>(), 0..128)) {
        let string = String::from_utf8(bytes.clone());
        let rope = Rope::try_from(bytes);
        match string {
            Ok(_) =>  { prop_assert!(rope.is_ok()); }
            Err(std_e) => {
                prop_assert!(rope.is_err());
                let my_e = rope.unwrap_err();
                prop_assert_eq!(my_e.utf8_error(), std_e.utf8_error());
            }
        }
    }

    #[test]
    fn test_ptr_eq_multichunk(s in ".{65,128}") {
        let a = Rope::from(&s);
        let b = Rope::from(&s);
        #[allow(clippy::redundant_clone)]
        let c = a.clone();

        prop_assert!(a.ptr_eq(&c));
        prop_assert!(c.ptr_eq(&a));
        prop_assert!(!a.ptr_eq(&b));
        prop_assert!(!b.ptr_eq(&c));
        prop_assert!(!b.ptr_eq(&a));
        prop_assert!(!c.ptr_eq(&b));
    }

    #[test]
    fn test_ptr_eq_singlechunk(s in ".{0,16}") {
        let a = Rope::from(&s);
        #[allow(clippy::redundant_clone)]
        let b = a.clone();

        prop_assert!(!a.ptr_eq(&b));
        prop_assert!(!b.ptr_eq(&a));
    }
}

#[test]
fn test_chunks() {
    let xx_31crabs_xx = "xx🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀xx";
    let rope = Rope::from(xx_31crabs_xx);

    let mut chunks = rope.chunks();
    assert_eq!(
        chunks.next(),
        Some(Chunk::Str("xx🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀"))
    );
    assert_eq!(chunks.next(), Some(Chunk::Char('🦀')));
    assert_eq!(
        chunks.next(),
        Some(Chunk::Str("🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀xx"))
    );
    assert_eq!(chunks.next(), None);

    chunks = rope.chunks();
    assert_eq!(
        chunks.next_back(),
        Some(Chunk::Str("🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀xx"))
    );
    assert_eq!(chunks.next_back(), Some(Chunk::Char('🦀')));
    assert_eq!(
        chunks.next_back(),
        Some(Chunk::Str("xx🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀"))
    );
    assert_eq!(chunks.next_back(), None);

    chunks = rope.chunks();
    assert_eq!(
        chunks.next(),
        Some(Chunk::Str("xx🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀"))
    );
    assert_eq!(
        chunks.next_back(),
        Some(Chunk::Str("🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀xx"))
    );
    assert_eq!(chunks.next_back(), Some(Chunk::Char('🦀')));
    assert_eq!(chunks.next_back(), None);

    chunks = rope.chunks();
    assert_eq!(
        chunks.next_back(),
        Some(Chunk::Str("🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀xx"))
    );
    assert_eq!(
        chunks.next(),
        Some(Chunk::Str("xx🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀"))
    );
    assert_eq!(chunks.next(), Some(Chunk::Char('🦀')));
    assert_eq!(chunks.next(), None);
}
