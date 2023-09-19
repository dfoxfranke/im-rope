//! Implementing types for pattern search
//!
//! This module provides the [`Pattern`] trait for types that can be used with
//! methods such as [`Rope::find_all`], along with some types that power its
//! implementation. These types are mostly implementation details, but they leak
//! out in a few obscure places such as the bound on [`FindAll`](crate::FindAll)'s
//! [`DoubleEndedIterator`] implementation, and so they are exposed and
//! documented. Nonetheless, it should rarely be necessary for users of this
//! crate to refer by name to any types defined in this module.

use crate::accessor::{Accessor, OwningAccessor};
use crate::{CharIndices, Rope};
use memchr::memmem;
use sealed::sealed;
use static_cow::{Borrowed, IntoOwning, Owned, StaticCow, ToOwning};
use std::collections::VecDeque;
use std::iter::{DoubleEndedIterator, FusedIterator, Iterator};
use std::ops::Range;

/// Trait for types that can be matched on.
///
/// This sealed trait provides ad-hoc polymorphism for methods such as
/// [`find`](Rope::find) which involve searching a rope for a substring,
/// Similarly to the standard library trait [`std::str::pattern::Pattern`], the
/// type of `Pattern` provided determines the match condition:
///
/// | Pattern type             | Match condition                           |
/// |--------------------------|-------------------------------------------|
/// | `&str`                   | is substring                              |
/// | `String`, `&String`      | is substring                              |
/// | `Rope`, `&Rope`          | is substring                              |
/// | `char`                   | is contained in rope                      |
/// | `&[char]`, `Vec<char>`   | any char in slice is contained in rope    |
/// | `F: Fn(char) -> bool`    | `F` returns `true` for a char in rope     |
///
/// The type of pattern likewise determines the complexity of the search. In the
/// following table, `N` represents the length of the needle, or when the needle
/// is a predicate, its execution time. `H` represents the length of the hasystack.
/// Note most search algorithms allocate O(N) temprorary storage regardless of
/// whether the needle is borrowed or owned.
///
/// | Pattern type             | Time complexity            | Space complexity |
/// |--------------------------|----------------------------|------------------|
/// | `&str`                   | O(N + H log H)             | O(N)             |
/// | `String`, `&String`      | O(N + H log H)             | O(N)             |
/// | `Rope`, `&Rope`          | O(N + H log H)             | O(N)             |
/// | `char`                   | O(H log H)                 | O(1)             |
/// | `&[char]`                | O(N log N + H log H log N) | O(N)             |
/// | `Vec<char>`              | O(N log N + H log H log N) | O(1)             |
/// | `F: Fn(char) -> bool`    | O(NH log H)                | O(1)             |
///
/// All of this trait's methods are `#[doc(hidden)]` and are excluded from this
/// crate's semantic versioning contract.

#[sealed]
pub trait Pattern {
    /// For some patterns, this type is returned along with the match range to
    /// indicate what was matched. Patterns that match on character predicates
    /// or sets of characters have an output of `char`. Patterns that match on
    /// a single string just output `()`.
    type Output;

    /// The equivalent owned version of this pattern. This type appears in the
    /// signature of [`FindAll::into_owning`](crate::FindAll::into_owning) but
    /// is otherwise uninteresting for consumers of this crate.
    type Owned: Pattern;

    /// The type which implements forward search for this pattern. This type
    /// in bounds on iterator implementations on `FindAll` but is otherwise
    /// uninteresting for consumers of this crate.
    type FindAllImpl<A>: Iterator<Item = (Range<usize>, Self::Output)> + FusedIterator
    where
        A: Accessor;

    /// The type which implements reverse search for this pattern. This type
    /// in bounds on iterator implementations on `RFindAll` but is otherwise
    /// uninteresting for consumers of this crate.
    type RFindAllImpl<A>: Iterator<Item = (Range<usize>, Self::Output)> + FusedIterator
    where
        A: Accessor;

    #[doc(hidden)]
    fn _find_all<A>(self, accessor: A) -> Self::FindAllImpl<A>
    where
        A: Accessor;
    #[doc(hidden)]
    fn _rfind_all<A>(self, accessor: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor;
    #[doc(hidden)]
    fn _is_prefix(self, haystack: &Rope) -> bool;
    #[doc(hidden)]
    fn _is_suffix(self, haystack: &Rope) -> bool;

    // We want a further bound of IntoOwning<Owning = <Self::Owned as
    // Pattern>::FindAllImpl<A::Owning>> on FindAllImpl and RFindAllImpl, but a
    // compiler bug (https://github.com/rust-lang/rust/issues/107160) prevents
    // this from working. These next four methods are a workaround.

    #[doc(hidden)]
    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor;
    #[doc(hidden)]
    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor;
    #[doc(hidden)]
    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor;
    #[doc(hidden)]
    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor;
}

#[sealed]
impl<F> Pattern for F
where
    F: Fn(char) -> bool,
{
    type Output = char;
    type Owned = F;

    type FindAllImpl<A> = FindPred<A, F, false> where A: Accessor;
    type RFindAllImpl<A> = FindPred<A, F, true> where A: Accessor;

    fn _find_all<A>(self, haystack: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        FindPred::new(haystack, self)
    }

    fn _rfind_all<A>(self, haystack: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        FindPred::new(haystack, self)
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        haystack.front().map_or(false, self)
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        haystack.back().map_or(false, self)
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }
    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }
    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
}

#[sealed]
impl Pattern for char {
    type Output = char;
    type Owned = Self;

    type FindAllImpl<A> = FindChar<A, false> where A: Accessor;
    type RFindAllImpl<A> = FindChar<A, true> where A: Accessor;

    fn _find_all<A>(self, accessor: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        FindChar::new(accessor, self)
    }

    fn _rfind_all<A>(self, accessor: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        FindChar::new(accessor, self)
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        haystack.front() == Some(self)
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        haystack.back() == Some(self)
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }

    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
}

#[sealed]
impl<'a> Pattern for &'a [char] {
    type Output = char;
    type Owned = Vec<char>;

    type FindAllImpl<A> = FindChars<A, false> where A:Accessor;

    type RFindAllImpl<A> = FindChars<A, true> where A:Accessor;

    fn _find_all<A>(self, accessor: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        FindChars::new(accessor, Borrowed(self))
    }

    fn _rfind_all<A>(self, accessor: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        FindChars::new(accessor, Borrowed(self))
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        haystack.front().map_or(false, |c| self.contains(&c))
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        haystack.back().map_or(false, |c| self.contains(&c))
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }

    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
}

#[sealed]
impl Pattern for Vec<char> {
    type Output = char;
    type Owned = Vec<char>;

    type FindAllImpl<A> = FindChars<A, false> where A:Accessor;

    type RFindAllImpl<A> = FindChars<A, true> where A:Accessor;

    fn _find_all<A>(self, accessor: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        FindChars::new(accessor, Owned(self))
    }

    fn _rfind_all<A>(self, accessor: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        FindChars::new(accessor, Owned(self))
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        haystack.front().map_or(false, |c| self.contains(&c))
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        haystack.back().map_or(false, |c| self.contains(&c))
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }

    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
}

#[sealed]
impl<'a> Pattern for &'a Vec<char> {
    type Output = char;
    type Owned = Vec<char>;

    type FindAllImpl<A> = FindChars<A, false> where A:Accessor;

    type RFindAllImpl<A> = FindChars<A, true> where A:Accessor;

    fn _find_all<A>(self, accessor: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        FindChars::new(accessor, Borrowed(self.as_ref()))
    }

    fn _rfind_all<A>(self, accessor: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        FindChars::new(accessor, Borrowed(self.as_ref()))
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        haystack.front().map_or(false, |c| self.contains(&c))
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        haystack.back().map_or(false, |c| self.contains(&c))
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }

    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
}

#[sealed]
impl<'n> Pattern for &'n str {
    type Output = ();
    type Owned = String;

    type FindAllImpl<A> = FindStr<'n, A> where A:Accessor;
    type RFindAllImpl<A> = RFindStr<'n, A> where A:Accessor;

    fn _find_all<A>(self, haystack: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        FindStr::borrowed(haystack, self)
    }

    fn _rfind_all<A>(self, haystack: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        RFindStr::borrowed(haystack, self)
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        if self.len() > haystack.len() {
            return false;
        }

        let mut needle_bytes = self.as_bytes();
        let mut chunks = haystack.as_ref().leaves();

        while !needle_bytes.is_empty() {
            let chunk = chunks.next().expect("haystack at least as long as needle");
            let len = std::cmp::min(needle_bytes.len(), chunk.len());
            if needle_bytes[..len].ne(&chunk[..len]) {
                return false;
            }
            needle_bytes = &needle_bytes[len..];
        }

        true
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        if self.len() > haystack.len() {
            return false;
        }

        let mut needle_bytes = self.as_bytes();
        let mut chunks = haystack.as_ref().leaves();

        while !needle_bytes.is_empty() {
            let chunk = chunks
                .next_back()
                .expect("haystack at least as long as needle");
            let len = std::cmp::min(needle_bytes.len(), chunk.len());
            if needle_bytes[needle_bytes.len() - len..].ne(&chunk[chunk.len() - len..]) {
                return false;
            }
            needle_bytes = &needle_bytes[..needle_bytes.len() - len];
        }

        true
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }

    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
}

#[sealed]
impl Pattern for String {
    type Output = ();
    type Owned = Self;

    type FindAllImpl<A> = FindStr<'static, A> where A:Accessor;

    type RFindAllImpl<A> = RFindStr<'static, A> where A:Accessor;

    fn _find_all<A>(self, haystack: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        FindStr::owned(haystack, self.as_str())
    }

    fn _rfind_all<A>(self, haystack: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        RFindStr::owned(haystack, self.as_str())
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        self.as_str()._is_prefix(haystack)
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        self.as_str()._is_suffix(haystack)
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }

    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
}

#[sealed]
impl<'n> Pattern for &'n String {
    type Output = ();
    type Owned = String;

    type FindAllImpl<A> = FindStr<'n, A> where A:Accessor;
    type RFindAllImpl<A> = RFindStr<'n, A> where A:Accessor;

    fn _find_all<A>(self, accessor: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        self.as_str()._find_all(accessor)
    }

    fn _rfind_all<A>(self, accessor: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        self.as_str()._rfind_all(accessor)
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        self.as_str()._is_prefix(haystack)
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        self.as_str()._is_suffix(haystack)
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        <&'n str>::_convert_to_owning(finder)
    }

    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        <&'n str>::_convert_into_owning(finder)
    }

    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        <&'n str>::_rconvert_to_owning(finder)
    }

    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        <&'n str>::_rconvert_into_owning(finder)
    }
}

#[sealed]
impl<'n> Pattern for &'n Rope {
    type Output = ();
    type Owned = Rope;

    type FindAllImpl<A> = FindStr<'static, A> where A:Accessor;
    type RFindAllImpl<A>  = RFindStr<'static, A> where A:Accessor;

    fn _find_all<A>(self, haystack: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        let needle: String = self.into();
        needle._find_all(haystack)
    }

    fn _rfind_all<A>(self, haystack: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        let needle: String = self.into();
        needle._rfind_all(haystack)
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        if self.len() > haystack.len() {
            return false;
        }

        let needle_chunks = self.as_ref().leaves();
        let mut haystack_chunks = haystack.as_ref().leaves();
        let mut haystack_chunk: &[u8] = &[];

        for mut needle_chunk in needle_chunks {
            while !needle_chunk.is_empty() {
                while haystack_chunk.is_empty() {
                    haystack_chunk = haystack_chunks
                        .next()
                        .expect("haystack at least as long as needle");
                }
                let len = std::cmp::min(needle_chunk.len(), haystack_chunk.len());
                if needle_chunk[..len].ne(&haystack_chunk[..len]) {
                    return false;
                }
                needle_chunk = &needle_chunk[len..];
                haystack_chunk = &haystack_chunk[len..];
            }
        }

        true
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        if self.len() > haystack.len() {
            return false;
        }

        let needle_chunks = self.as_ref().leaves().rev();
        let mut haystack_chunks = haystack.as_ref().leaves().rev();
        let mut haystack_chunk: &[u8] = &[];

        for mut needle_chunk in needle_chunks {
            while !needle_chunk.is_empty() {
                while haystack_chunk.is_empty() {
                    haystack_chunk = haystack_chunks
                        .next()
                        .expect("haystack at least as long as needle");
                }
                let len = std::cmp::min(needle_chunk.len(), haystack_chunk.len());
                if needle_chunk[needle_chunk.len() - len..]
                    .ne(&haystack_chunk[haystack_chunk.len() - len..])
                {
                    return false;
                }
                needle_chunk = &needle_chunk[..needle_chunk.len() - len];
                haystack_chunk = &haystack_chunk[..haystack_chunk.len() - len];
            }
        }
        true
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }

    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
}

#[sealed]
impl Pattern for Rope {
    type Output = ();
    type Owned = Self;

    type FindAllImpl<A> = FindStr<'static, A> where A:Accessor;
    type RFindAllImpl<A>  = RFindStr<'static, A> where A:Accessor;

    fn _find_all<A>(self, haystack: A) -> Self::FindAllImpl<A>
    where
        A: Accessor,
    {
        let needle: String = self.into();
        needle._find_all(haystack)
    }

    fn _rfind_all<A>(self, haystack: A) -> Self::RFindAllImpl<A>
    where
        A: Accessor,
    {
        let needle: String = self.into();
        needle._rfind_all(haystack)
    }

    fn _is_prefix(self, haystack: &Rope) -> bool {
        (&self)._is_prefix(haystack)
    }

    fn _is_suffix(self, haystack: &Rope) -> bool {
        (&self)._is_suffix(haystack)
    }

    fn _convert_to_owning<A>(
        finder: &Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _convert_into_owning<A>(
        finder: Self::FindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }

    fn _rconvert_to_owning<A>(
        finder: &Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.to_owning()
    }

    fn _rconvert_into_owning<A>(
        finder: Self::RFindAllImpl<A>,
    ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
    where
        A: Accessor,
    {
        finder.into_owning()
    }
}

/// Finder implementation for predicates.
pub struct FindPred<A, F, const REV: bool> {
    hayspout: CharIndices<A>,
    pred: std::rc::Rc<F>,
}

impl<A, F, const REV: bool> FindPred<A, F, REV>
where
    F: Fn(char) -> bool,
    A: Accessor,
{
    fn new(haystack: A, pred: F) -> FindPred<A, F, REV> {
        FindPred {
            hayspout: CharIndices::new(haystack),
            pred: std::rc::Rc::new(pred),
        }
    }
}

impl<A, F, const REV: bool> ToOwning for FindPred<A, F, REV>
where
    F: Fn(char) -> bool,
    A: Accessor,
{
    type Owning = FindPred<A::Owning, F, REV>;

    fn to_owning(&self) -> Self::Owning {
        FindPred {
            hayspout: self.hayspout.to_owning(),
            pred: self.pred.to_owning(),
        }
    }
}

impl<A, F, const REV: bool> IntoOwning for FindPred<A, F, REV>
where
    F: Fn(char) -> bool,
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        FindPred {
            hayspout: self.hayspout.into_owning(),
            pred: self.pred.into_owning(),
        }
    }
}

impl<A, F, const REV: bool> Iterator for FindPred<A, F, REV>
where
    A: Accessor,
    F: Fn(char) -> bool,
{
    type Item = (Range<usize>, char);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (i, c) = if REV {
                self.hayspout.next_back()?
            } else {
                self.hayspout.next()?
            };
            if (self.pred)(c) {
                return Some((i..i + c.len_utf8(), c));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, max) = self.hayspout.size_hint();
        (0, max)
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<A, F, const REV: bool> DoubleEndedIterator for FindPred<A, F, REV>
where
    A: Accessor,
    F: Fn(char) -> bool,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (i, c) = if REV {
                self.hayspout.next()?
            } else {
                self.hayspout.next_back()?
            };
            if (self.pred)(c) {
                return Some((i..i + c.len_utf8(), c));
            }
        }
    }
}

impl<A, F, const REV: bool> FusedIterator for FindPred<A, F, REV>
where
    A: Accessor,
    F: Fn(char) -> bool,
{
}

/// Finder implementation for character lists.
pub struct FindChars<A, const REV: bool> {
    inner: FindCharsImpl<A, REV>,
}
enum FindCharsImpl<A, const REV: bool> {
    One(FindMemchr<A, Memchr1Helper, REV>),
    Two(FindMemchr<A, Memchr2Helper, REV>),
    Three(FindMemchr<A, Memchr3Helper, REV>),
    Generic(FindCharsGeneric<A, REV>),
}

/// Finder implementation for characters.
pub struct FindChar<A, const REV: bool> {
    inner: FindCharImpl<A, REV>,
}

enum FindCharImpl<A, const REV: bool> {
    Ascii(FindMemchr<A, Memchr1Helper, REV>),
    Generic(FindCharGeneric<A, REV>),
}

impl<A, const REV: bool> FindChars<A, REV>
where
    A: Accessor,
{
    fn new<N: StaticCow<[char]>>(haystack: A, needle: N) -> FindChars<A, REV> {
        FindChars {
            inner: match &*needle {
                [a] if a.is_ascii() => FindCharsImpl::One(FindMemchr::new(haystack, *a as u8)),
                [a, b] if a.is_ascii() && b.is_ascii() => {
                    FindCharsImpl::Two(FindMemchr::new(haystack, (*a as u8, *b as u8)))
                }
                [a, b, c] if a.is_ascii() && b.is_ascii() && c.is_ascii() => {
                    FindCharsImpl::Three(FindMemchr::new(haystack, (*a as u8, *b as u8, *c as u8)))
                }
                _ => FindCharsImpl::Generic(FindCharsGeneric::new(haystack, needle.into_owned())),
            },
        }
    }
}

impl<A, const REV: bool> ToOwning for FindChars<A, REV>
where
    A: Accessor,
{
    type Owning = FindChars<A::Owning, REV>;

    fn to_owning(&self) -> Self::Owning {
        FindChars {
            inner: match &self.inner {
                FindCharsImpl::One(iter) => FindCharsImpl::One(iter.to_owning()),
                FindCharsImpl::Two(iter) => FindCharsImpl::Two(iter.to_owning()),
                FindCharsImpl::Three(iter) => FindCharsImpl::Three(iter.to_owning()),
                FindCharsImpl::Generic(iter) => FindCharsImpl::Generic(iter.to_owning()),
            },
        }
    }
}

impl<A, const REV: bool> IntoOwning for FindChars<A, REV>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        FindChars {
            inner: match self.inner {
                FindCharsImpl::One(iter) => FindCharsImpl::One(iter.into_owning()),
                FindCharsImpl::Two(iter) => FindCharsImpl::Two(iter.into_owning()),
                FindCharsImpl::Three(iter) => FindCharsImpl::Three(iter.into_owning()),
                FindCharsImpl::Generic(iter) => FindCharsImpl::Generic(iter.into_owning()),
            },
        }
    }
}

impl<A, const REV: bool> Iterator for FindChars<A, REV>
where
    A: Accessor,
{
    type Item = (Range<usize>, char);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            FindCharsImpl::One(iter) => iter.next(),
            FindCharsImpl::Two(iter) => iter.next(),
            FindCharsImpl::Three(iter) => iter.next(),
            FindCharsImpl::Generic(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            FindCharsImpl::One(iter) => iter.size_hint(),
            FindCharsImpl::Two(iter) => iter.size_hint(),
            FindCharsImpl::Three(iter) => iter.size_hint(),
            FindCharsImpl::Generic(iter) => iter.size_hint(),
        }
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<A, const REV: bool> DoubleEndedIterator for FindChars<A, REV>
where
    A: Accessor,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            FindCharsImpl::One(iter) => iter.next_back(),
            FindCharsImpl::Two(iter) => iter.next_back(),
            FindCharsImpl::Three(iter) => iter.next_back(),
            FindCharsImpl::Generic(iter) => iter.next_back(),
        }
    }
}

impl<A, const REV: bool> FusedIterator for FindChars<A, REV> where A: Accessor {}

impl<A, const REV: bool> FindChar<A, REV>
where
    A: Accessor,
{
    fn new(haystack: A, needle: char) -> FindChar<A, REV> {
        FindChar {
            inner: if needle.is_ascii() {
                FindCharImpl::Ascii(FindMemchr::new(haystack, needle as u8))
            } else {
                FindCharImpl::Generic(FindCharGeneric::new(haystack, needle))
            },
        }
    }
}

impl<A, const REV: bool> ToOwning for FindChar<A, REV>
where
    A: Accessor,
{
    type Owning = FindChar<A::Owning, REV>;

    fn to_owning(&self) -> Self::Owning {
        FindChar {
            inner: match &self.inner {
                FindCharImpl::Ascii(iter) => FindCharImpl::Ascii(iter.to_owning()),
                FindCharImpl::Generic(iter) => FindCharImpl::Generic(iter.to_owning()),
            },
        }
    }
}

impl<A, const REV: bool> IntoOwning for FindChar<A, REV>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        FindChar {
            inner: match self.inner {
                FindCharImpl::Ascii(iter) => FindCharImpl::Ascii(iter.into_owning()),
                FindCharImpl::Generic(iter) => FindCharImpl::Generic(iter.into_owning()),
            },
        }
    }
}

impl<A, const REV: bool> Iterator for FindChar<A, REV>
where
    A: Accessor,
{
    type Item = (Range<usize>, char);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            FindCharImpl::Ascii(iter) => iter.next(),
            FindCharImpl::Generic(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            FindCharImpl::Ascii(iter) => iter.size_hint(),
            FindCharImpl::Generic(iter) => iter.size_hint(),
        }
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<A, const REV: bool> DoubleEndedIterator for FindChar<A, REV>
where
    A: Accessor,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            FindCharImpl::Ascii(iter) => iter.next_back(),
            FindCharImpl::Generic(iter) => iter.next_back(),
        }
    }
}

impl<A, const REV: bool> FusedIterator for FindChar<A, REV> where A: Accessor {}

struct FindCharsGeneric<A, const REV: bool> {
    hayspout: CharIndices<A>,
    needle: Vec<char>,
}

impl<A, const REV: bool> FindCharsGeneric<A, REV>
where
    A: Accessor,
{
    fn new(haystack: A, mut needle: Vec<char>) -> FindCharsGeneric<A, REV> {
        needle.sort_unstable();
        FindCharsGeneric {
            hayspout: CharIndices::new(haystack),
            needle,
        }
    }
}

impl<A, const REV: bool> ToOwning for FindCharsGeneric<A, REV>
where
    A: Accessor,
{
    type Owning = FindCharsGeneric<A::Owning, REV>;

    fn to_owning(&self) -> Self::Owning {
        FindCharsGeneric {
            hayspout: self.hayspout.to_owning(),
            needle: self.needle.to_owning(),
        }
    }
}

impl<A, const REV: bool> IntoOwning for FindCharsGeneric<A, REV>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        FindCharsGeneric {
            hayspout: self.hayspout.into_owning(),
            needle: self.needle.into_owning(),
        }
    }
}

impl<A, const REV: bool> Iterator for FindCharsGeneric<A, REV>
where
    A: Accessor,
{
    type Item = (Range<usize>, char);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (i, c) = if REV {
                self.hayspout.next_back()?
            } else {
                self.hayspout.next()?
            };
            if self.needle.binary_search(&c).is_ok() {
                return Some((i..i + c.len_utf8(), c));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, max) = self.hayspout.size_hint();
        (0, max)
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<A, const REV: bool> DoubleEndedIterator for FindCharsGeneric<A, REV>
where
    A: Accessor,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (i, c) = if REV {
                self.hayspout.next()?
            } else {
                self.hayspout.next_back()?
            };
            if self.needle.binary_search(&c).is_ok() {
                return Some((i..i + c.len_utf8(), c));
            }
        }
    }
}

impl<A, const REV: bool> FusedIterator for FindCharsGeneric<A, REV> where A: Accessor {}

struct FindCharGeneric<A, const REV: bool> {
    hayspout: CharIndices<A>,
    needle: char,
}

impl<A, const REV: bool> FindCharGeneric<A, REV>
where
    A: Accessor,
{
    fn new(haystack: A, needle: char) -> FindCharGeneric<A, REV> {
        FindCharGeneric {
            hayspout: CharIndices::new(haystack),
            needle,
        }
    }
}

impl<A, const REV: bool> ToOwning for FindCharGeneric<A, REV>
where
    A: Accessor,
{
    type Owning = FindCharGeneric<A::Owning, REV>;

    fn to_owning(&self) -> Self::Owning {
        FindCharGeneric {
            hayspout: self.hayspout.to_owning(),
            needle: self.needle.to_owning(),
        }
    }
}

impl<A, const REV: bool> IntoOwning for FindCharGeneric<A, REV>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        FindCharGeneric {
            hayspout: self.hayspout.into_owning(),
            needle: self.needle.into_owning(),
        }
    }
}

impl<A, const REV: bool> Iterator for FindCharGeneric<A, REV>
where
    A: Accessor,
{
    type Item = (Range<usize>, char);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (i, c) = if REV {
                self.hayspout.next_back()?
            } else {
                self.hayspout.next()?
            };
            if self.needle == c {
                return Some((i..i + c.len_utf8(), c));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, max) = self.hayspout.size_hint();
        (0, max)
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<A, const REV: bool> DoubleEndedIterator for FindCharGeneric<A, REV>
where
    A: Accessor,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (i, c) = if REV {
                self.hayspout.next()?
            } else {
                self.hayspout.next_back()?
            };
            if self.needle == c {
                return Some((i..i + c.len_utf8(), c));
            }
        }
    }
}

impl<A, const REV: bool> FusedIterator for FindCharGeneric<A, REV> where A: Accessor {}

trait MemchrHelper {
    type Needle: Copy;
    type Output<'a>: Iterator<Item = usize> + DoubleEndedIterator;

    fn build(needle: Self::Needle, haystack: &[u8]) -> Self::Output<'_>;
}

enum Memchr1Helper {}
impl MemchrHelper for Memchr1Helper {
    type Needle = u8;
    type Output<'a> = memchr::Memchr<'a>;

    fn build(needle: Self::Needle, haystack: &[u8]) -> Self::Output<'_> {
        memchr::memchr_iter(needle, haystack)
    }
}

enum Memchr2Helper {}
impl MemchrHelper for Memchr2Helper {
    type Needle = (u8, u8);
    type Output<'a> = memchr::Memchr2<'a>;

    fn build(needle: Self::Needle, haystack: &[u8]) -> Self::Output<'_> {
        memchr::memchr2_iter(needle.0, needle.1, haystack)
    }
}

enum Memchr3Helper {}
impl MemchrHelper for Memchr3Helper {
    type Needle = (u8, u8, u8);
    type Output<'a> = memchr::Memchr3<'a>;

    fn build(needle: Self::Needle, haystack: &[u8]) -> Self::Output<'_> {
        memchr::memchr3_iter(needle.0, needle.1, needle.2, haystack)
    }
}

struct FindMemchr<A, H, const REV: bool>
where
    H: MemchrHelper,
{
    haystack: A,
    needle: H::Needle,
    front_matches: VecDeque<(Range<usize>, char)>,
    back_matches: VecDeque<(Range<usize>, char)>,
}

impl<A, H, const REV: bool> FindMemchr<A, H, REV>
where
    A: Accessor,
    H: MemchrHelper,
{
    fn new(haystack: A, needle: H::Needle) -> FindMemchr<A, H, REV> {
        FindMemchr {
            haystack,
            needle,
            front_matches: VecDeque::with_capacity(64),
            back_matches: VecDeque::with_capacity(64),
        }
    }

    fn forward(&mut self) -> Option<(Range<usize>, char)> {
        while self.front_matches.is_empty() {
            if let Some((range, chunk)) = self.haystack.front_chunk() {
                for i in H::build(self.needle, chunk) {
                    #[allow(clippy::range_plus_one)]
                    self.front_matches
                        .push_back((range.start + i..range.start + i + 1, chunk[i] as char));
                }
            } else {
                break;
            }
        }

        self.front_matches
            .pop_front()
            .or_else(|| self.back_matches.pop_back())
    }

    fn backward(&mut self) -> Option<(Range<usize>, char)> {
        while self.back_matches.is_empty() {
            if let Some((range, chunk)) = self.haystack.back_chunk() {
                for i in H::build(self.needle, chunk).rev() {
                    #[allow(clippy::range_plus_one)]
                    self.back_matches
                        .push_back((range.start + i..range.start + i + 1, chunk[i] as char));
                }
            } else {
                break;
            }
        }

        self.back_matches
            .pop_front()
            .or_else(|| self.front_matches.pop_back())
    }
}

impl<A, H, const REV: bool> ToOwning for FindMemchr<A, H, REV>
where
    A: Accessor,
    H: MemchrHelper,
{
    type Owning = FindMemchr<A::Owning, H, REV>;

    fn to_owning(&self) -> Self::Owning {
        FindMemchr {
            haystack: self.haystack.to_owning(),
            needle: self.needle.to_owning(),
            front_matches: self.front_matches.to_owning(),
            back_matches: self.back_matches.to_owning(),
        }
    }
}

impl<A, H, const REV: bool> IntoOwning for FindMemchr<A, H, REV>
where
    A: Accessor,
    H: MemchrHelper,
{
    fn into_owning(self) -> Self::Owning {
        FindMemchr {
            haystack: self.haystack.into_owning(),
            needle: self.needle.into_owning(),
            front_matches: self.front_matches.into_owning(),
            back_matches: self.back_matches.into_owning(),
        }
    }
}

impl<A, H, const REV: bool> Iterator for FindMemchr<A, H, REV>
where
    A: Accessor,
    H: MemchrHelper,
{
    type Item = (Range<usize>, char);

    fn next(&mut self) -> Option<Self::Item> {
        if REV {
            self.backward()
        } else {
            self.forward()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = self.front_matches.len() + self.back_matches.len();
        let max = min + (self.haystack.back_index() - self.haystack.front_index());
        (min, Some(max))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<A, H, const REV: bool> DoubleEndedIterator for FindMemchr<A, H, REV>
where
    A: Accessor,
    H: MemchrHelper,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if REV {
            self.forward()
        } else {
            self.backward()
        }
    }
}

impl<A, H, const REV: bool> FusedIterator for FindMemchr<A, H, REV>
where
    A: Accessor,
    H: MemchrHelper,
{
}

/// Finder implementation for strings.
pub struct FindStr<'n, A> {
    inner: FindStrImpl<'n, A>,
}

enum FindStrImpl<'n, A> {
    Nonempty(FindNonemptyStr<'n, A>),
    Empty(FindEmpty<A, false>),
}

impl<'n, A> FindStr<'n, A>
where
    A: Accessor,
{
    fn borrowed(haystack: A, needle: &'n str) -> FindStr<'n, A> {
        FindStr {
            inner: if needle.is_empty() {
                FindStrImpl::Empty(FindEmpty::new(haystack))
            } else {
                FindStrImpl::Nonempty(FindNonemptyStr::borrowed(haystack, needle))
            },
        }
    }

    fn owned(haystack: A, needle: &'n str) -> FindStr<'static, A> {
        FindStr {
            inner: if needle.is_empty() {
                FindStrImpl::Empty(FindEmpty::new(haystack))
            } else {
                FindStrImpl::Nonempty(FindNonemptyStr::owned(haystack, needle))
            },
        }
    }
}

impl<'n, A> ToOwning for FindStr<'n, A>
where
    A: Accessor,
{
    type Owning = FindStr<'static, A::Owning>;

    fn to_owning(&self) -> Self::Owning {
        FindStr {
            inner: match &self.inner {
                FindStrImpl::Nonempty(iter) => FindStrImpl::Nonempty(iter.to_owning()),
                FindStrImpl::Empty(iter) => FindStrImpl::Empty(iter.to_owning()),
            },
        }
    }
}

impl<'n, A> IntoOwning for FindStr<'n, A>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        FindStr {
            inner: match self.inner {
                FindStrImpl::Nonempty(iter) => FindStrImpl::Nonempty(iter.into_owning()),
                FindStrImpl::Empty(iter) => FindStrImpl::Empty(iter.into_owning()),
            },
        }
    }
}

impl<'n, A> Iterator for FindStr<'n, A>
where
    A: Accessor,
{
    type Item = (Range<usize>, ());

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            FindStrImpl::Nonempty(iter) => iter.next(),
            FindStrImpl::Empty(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            FindStrImpl::Nonempty(iter) => iter.size_hint(),
            FindStrImpl::Empty(iter) => iter.size_hint(),
        }
    }
}

impl<'n, A> FusedIterator for FindStr<'n, A> where A: Accessor {}

/// Reverse finder implementation for strings.
pub struct RFindStr<'n, A> {
    inner: RFindStrImpl<'n, A>,
}

enum RFindStrImpl<'n, A> {
    Nonempty(RFindNonemptyStr<'n, A>),
    Empty(FindEmpty<A, true>),
}

impl<'n, A> RFindStr<'n, A>
where
    A: Accessor,
{
    fn borrowed(haystack: A, needle: &'n str) -> RFindStr<'n, A> {
        RFindStr {
            inner: if needle.is_empty() {
                RFindStrImpl::Empty(FindEmpty::new(haystack))
            } else {
                RFindStrImpl::Nonempty(RFindNonemptyStr::borrowed(haystack, needle))
            },
        }
    }

    fn owned(haystack: A, needle: &'n str) -> RFindStr<'static, A> {
        RFindStr {
            inner: if needle.is_empty() {
                RFindStrImpl::Empty(FindEmpty::new(haystack))
            } else {
                RFindStrImpl::Nonempty(RFindNonemptyStr::owned(haystack, needle))
            },
        }
    }
}

impl<'n, A> ToOwning for RFindStr<'n, A>
where
    A: Accessor,
{
    type Owning = RFindStr<'static, A::Owning>;

    fn to_owning(&self) -> Self::Owning {
        RFindStr {
            inner: match &self.inner {
                RFindStrImpl::Nonempty(iter) => RFindStrImpl::Nonempty(iter.to_owning()),
                RFindStrImpl::Empty(iter) => RFindStrImpl::Empty(iter.to_owning()),
            },
        }
    }
}

impl<'n, A> IntoOwning for RFindStr<'n, A>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        RFindStr {
            inner: match self.inner {
                RFindStrImpl::Nonempty(iter) => RFindStrImpl::Nonempty(iter.into_owning()),
                RFindStrImpl::Empty(iter) => RFindStrImpl::Empty(iter.into_owning()),
            },
        }
    }
}

impl<'n, A> Iterator for RFindStr<'n, A>
where
    A: Accessor,
{
    type Item = (Range<usize>, ());

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            RFindStrImpl::Nonempty(iter) => iter.next(),
            RFindStrImpl::Empty(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            RFindStrImpl::Nonempty(iter) => iter.size_hint(),
            RFindStrImpl::Empty(iter) => iter.size_hint(),
        }
    }
}

impl<'n, A> FusedIterator for RFindStr<'n, A> where A: Accessor {}

struct FindEmpty<A, const REV: bool> {
    hayspout: CharIndices<A>,
    back: Option<usize>,
}

impl<A, const REV: bool> FindEmpty<A, REV>
where
    A: Accessor,
{
    fn new(haystack: A) -> FindEmpty<A, REV> {
        FindEmpty {
            back: Some(haystack.back_index()),
            hayspout: CharIndices::new(haystack),
        }
    }

    fn forward(&mut self) -> Option<(Range<usize>, ())> {
        if let Some((i, _)) = self.hayspout.next() {
            Some((i..i, ()))
        } else {
            self.back.take().map(|i| (i..i, ()))
        }
    }

    fn backward(&mut self) -> Option<(Range<usize>, ())> {
        if let Some(i) = self.back.take() {
            Some((i..i, ()))
        } else {
            self.hayspout.next_back().map(|(i, _)| (i..i, ()))
        }
    }
}

impl<A, const REV: bool> ToOwning for FindEmpty<A, REV>
where
    A: Accessor,
{
    type Owning = FindEmpty<A::Owning, REV>;

    fn to_owning(&self) -> Self::Owning {
        FindEmpty {
            hayspout: self.hayspout.to_owning(),
            back: self.back.to_owning(),
        }
    }
}

impl<A, const REV: bool> IntoOwning for FindEmpty<A, REV>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        FindEmpty {
            hayspout: self.hayspout.into_owning(),
            back: self.back.into_owning(),
        }
    }
}

impl<A, const REV: bool> Iterator for FindEmpty<A, REV>
where
    A: Accessor,
{
    type Item = (Range<usize>, ());

    fn next(&mut self) -> Option<Self::Item> {
        if REV {
            self.backward()
        } else {
            self.forward()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min, max) = self.hayspout.size_hint();
        let more: usize = self.back.is_some().into();
        (min + more, max.map(|n| n + more))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<A, const REV: bool> DoubleEndedIterator for FindEmpty<A, REV>
where
    A: Accessor,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if REV {
            self.forward()
        } else {
            self.backward()
        }
    }
}

struct FindNonemptyStr<'n, A> {
    haystack: A,
    finder: memmem::Finder<'n>,
    window: VecDeque<u8>,
    matches: VecDeque<(Range<usize>, ())>,
    window_start_index: usize,
}

impl<'n, A> FindNonemptyStr<'n, A>
where
    A: Accessor,
{
    fn borrowed(haystack: A, needle: &'n str) -> FindNonemptyStr<'n, A> {
        let match_capacity = (haystack.back_index() - haystack.front_index()).saturating_add(63);
        let window_capacity =
            match_capacity.saturating_add(haystack.back_index() - haystack.front_index());
        FindNonemptyStr {
            window_start_index: haystack.front_index(),
            haystack,
            finder: memmem::Finder::new(needle),
            window: VecDeque::with_capacity(window_capacity),
            matches: VecDeque::with_capacity(match_capacity),
        }
    }

    fn owned(haystack: A, needle: &'n str) -> FindNonemptyStr<'static, A> {
        let match_capacity = (haystack.back_index() - haystack.front_index()).saturating_add(63);
        let window_capacity =
            match_capacity.saturating_add(haystack.back_index() - haystack.front_index());
        FindNonemptyStr {
            window_start_index: haystack.front_index(),
            haystack,
            finder: memmem::Finder::new(needle).into_owned(),
            window: VecDeque::with_capacity(window_capacity),
            matches: VecDeque::with_capacity(match_capacity),
        }
    }
}

impl<'n, A> ToOwning for FindNonemptyStr<'n, A>
where
    A: Accessor,
{
    type Owning = FindNonemptyStr<'static, A::Owning>;

    fn to_owning(&self) -> Self::Owning {
        FindNonemptyStr {
            haystack: self.haystack.to_owning(),
            finder: self.finder.clone().into_owned(),
            window: self.window.to_owning(),
            matches: self.matches.to_owning(),
            window_start_index: self.window_start_index.to_owning(),
        }
    }
}

impl<'n, A> IntoOwning for FindNonemptyStr<'n, A>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        FindNonemptyStr {
            haystack: self.haystack.into_owning(),
            finder: self.finder.into_owned(),
            window: self.window.into_owning(),
            matches: self.matches.into_owning(),
            window_start_index: self.window_start_index.into_owning(),
        }
    }
}

impl<'n, A> Iterator for FindNonemptyStr<'n, A>
where
    A: Accessor,
{
    type Item = (Range<usize>, ());

    fn next(&mut self) -> Option<Self::Item> {
        let needle_len = self.finder.needle().len();

        while self.matches.is_empty() {
            while self.window.len() / 2 < needle_len {
                match self.haystack.front_chunk() {
                    Some((_, chunk)) => {
                        self.window.extend(chunk);
                    }
                    None => {
                        break;
                    }
                }
            }

            if self.window.len() < needle_len {
                break;
            }

            for i in self.finder.find_iter(self.window.make_contiguous()) {
                self.matches.push_back((
                    self.window_start_index + i..self.window_start_index + i + needle_len,
                    (),
                ));
            }

            let cutoff = self.window.len() - needle_len + 1;
            self.window.drain(..cutoff);
            self.window_start_index += cutoff;
        }

        self.matches.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.matches.len(),
            Some(
                self.matches.len()
                    + (self.window.len()
                        + (self.haystack.back_index() - self.haystack.front_index())
                        + 1)
                    .saturating_sub(self.finder.needle().len()),
            ),
        )
    }
}

impl<'n, A> FusedIterator for FindNonemptyStr<'n, A> where A: Accessor {}

struct RFindNonemptyStr<'n, A> {
    haystack: A,
    finder: memmem::FinderRev<'n>,
    window: VecDeque<u8>,
    matches: VecDeque<(Range<usize>, ())>,
    window_start_index: usize,
}

impl<'n, A> RFindNonemptyStr<'n, A>
where
    A: Accessor,
{
    fn borrowed(haystack: A, needle: &'n str) -> RFindNonemptyStr<'n, A> {
        let match_capacity = (haystack.back_index() - haystack.front_index()).saturating_add(63);
        let window_capacity =
            match_capacity.saturating_add(haystack.back_index() - haystack.front_index());
        RFindNonemptyStr {
            window_start_index: haystack.back_index(),
            haystack,
            finder: memmem::FinderRev::new(needle),
            window: VecDeque::with_capacity(window_capacity),
            matches: VecDeque::with_capacity(match_capacity),
        }
    }

    fn owned(haystack: A, needle: &'n str) -> RFindNonemptyStr<'static, A> {
        let match_capacity = (haystack.back_index() - haystack.front_index()).saturating_add(63);
        let window_capacity =
            match_capacity.saturating_add(haystack.back_index() - haystack.front_index());
        RFindNonemptyStr {
            window_start_index: haystack.back_index(),
            haystack,
            finder: memmem::FinderRev::new(needle).into_owned(),
            window: VecDeque::with_capacity(window_capacity),
            matches: VecDeque::with_capacity(match_capacity),
        }
    }
}

impl<'n, A> ToOwning for RFindNonemptyStr<'n, A>
where
    A: Accessor,
{
    type Owning = RFindNonemptyStr<'static, A::Owning>;

    fn to_owning(&self) -> Self::Owning {
        RFindNonemptyStr {
            haystack: self.haystack.to_owning(),
            finder: self.finder.clone().into_owned(),
            window: self.window.to_owning(),
            matches: self.matches.to_owning(),
            window_start_index: self.window_start_index.to_owning(),
        }
    }
}

impl<'n, A> IntoOwning for RFindNonemptyStr<'n, A>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        RFindNonemptyStr {
            haystack: self.haystack.into_owning(),
            finder: self.finder.into_owned(),
            window: self.window.into_owning(),
            matches: self.matches.into_owning(),
            window_start_index: self.window_start_index.into_owning(),
        }
    }
}

impl<'n, A> Iterator for RFindNonemptyStr<'n, A>
where
    A: Accessor,
{
    type Item = (Range<usize>, ());

    fn next(&mut self) -> Option<Self::Item> {
        let needle_len = self.finder.needle().len();

        while self.matches.is_empty() {
            while self.window.len() / 2 < needle_len {
                match self.haystack.back_chunk() {
                    Some((_, chunk)) => {
                        for byte in chunk.iter().rev() {
                            self.window.push_front(*byte);
                        }
                        self.window_start_index -= chunk.len();
                    }
                    None => {
                        break;
                    }
                }
            }

            if self.window.len() < needle_len {
                break;
            }

            for i in self.finder.rfind_iter(self.window.make_contiguous()) {
                self.matches.push_back((
                    self.window_start_index + i..self.window_start_index + i + needle_len,
                    (),
                ));
            }

            let cutoff = needle_len - 1;
            self.window.truncate(cutoff);
        }

        self.matches.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.matches.len(),
            Some(
                self.matches.len()
                    + (self.window.len()
                        + (self.haystack.back_index() - self.haystack.front_index())
                        + 1)
                    .saturating_sub(self.finder.needle().len()),
            ),
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;
    #[derive(Debug, Copy, Clone)]
    pub(crate) struct CharStrategy;

    impl Strategy for CharStrategy {
        type Tree = prop::char::CharValueTree;

        type Value = char;

        fn new_tree(
            &self,
            runner: &mut prop::test_runner::TestRunner,
        ) -> prop::strategy::NewTree<Self> {
            if runner.rng().gen_bool(0.5) {
                any::<char>().new_tree(runner)
            } else {
                prop::char::range('\0', '\u{7F}').new_tree(runner)
            }
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub(crate) struct CharsetStrategy;

    impl Strategy for CharsetStrategy {
        type Tree = prop::collection::VecValueTree<prop::char::CharValueTree>;

        type Value = Vec<char>;

        fn new_tree(
            &self,
            runner: &mut prop::test_runner::TestRunner,
        ) -> prop::strategy::NewTree<Self> {
            if runner.rng().gen_bool(0.5) {
                prop::collection::vec(any::<char>(), 0..8).new_tree(runner)
            } else {
                prop::collection::vec(prop::char::range('\0', '\u{7F}'), 1..=3).new_tree(runner)
            }
        }
    }

    #[derive(Debug, Clone)]
    pub(crate) enum CharPattern {
        Pred(Vec<char>),
        Char(char),
        Chars(Vec<char>),
        CharsVec(Vec<char>),
        CharsVecRef(Vec<char>),
    }

    impl CharPattern {
        pub(crate) fn to_pred(&self) -> Box<dyn Fn(char) -> bool + 'static> {
            match self {
                Self::Pred(v) | Self::Chars(v) | Self::CharsVec(v) | Self::CharsVecRef(v) => {
                    vec_to_pred(v.clone())
                }
                Self::Char(ch) => {
                    let ch = *ch;
                    Box::new(move |other| other == ch)
                }
            }
        }
    }

    fn vec_to_pred(v: Vec<char>) -> Box<dyn Fn(char) -> bool + 'static> {
        Box::new(move |ch| v.contains(&ch))
    }

    pub(crate) enum CharPatternFinder<A, const REV: bool> {
        Pred(crate::pattern::FindPred<A, Box<dyn Fn(char) -> bool + 'static>, REV>),
        Char(crate::pattern::FindChar<A, REV>),
        Chars(crate::pattern::FindChars<A, REV>),
    }

    impl<A, const REV: bool> ToOwning for CharPatternFinder<A, REV>
    where
        A: Accessor,
    {
        type Owning = CharPatternFinder<A::Owning, REV>;

        fn to_owning(&self) -> Self::Owning {
            match self {
                CharPatternFinder::Pred(p) => CharPatternFinder::Pred(p.to_owning()),
                CharPatternFinder::Char(p) => CharPatternFinder::Char(p.to_owning()),
                CharPatternFinder::Chars(p) => CharPatternFinder::Chars(p.to_owning()),
            }
        }
    }

    impl<A, const REV: bool> IntoOwning for CharPatternFinder<A, REV>
    where
        A: Accessor,
    {
        fn into_owning(self) -> Self::Owning {
            match self {
                CharPatternFinder::Pred(p) => CharPatternFinder::Pred(p.into_owning()),
                CharPatternFinder::Char(p) => CharPatternFinder::Char(p.into_owning()),
                CharPatternFinder::Chars(p) => CharPatternFinder::Chars(p.into_owning()),
            }
        }
    }

    impl<A, const REV: bool> Iterator for CharPatternFinder<A, REV>
    where
        A: Accessor,
    {
        type Item = (Range<usize>, char);

        fn next(&mut self) -> Option<Self::Item> {
            match self {
                CharPatternFinder::Pred(iter) => iter.next(),
                CharPatternFinder::Char(iter) => iter.next(),
                CharPatternFinder::Chars(iter) => iter.next(),
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            match self {
                CharPatternFinder::Pred(iter) => iter.size_hint(),
                CharPatternFinder::Char(iter) => iter.size_hint(),
                CharPatternFinder::Chars(iter) => iter.size_hint(),
            }
        }

        fn last(self) -> Option<Self::Item> {
            match self {
                CharPatternFinder::Pred(iter) => iter.last(),
                CharPatternFinder::Char(iter) => iter.last(),
                CharPatternFinder::Chars(iter) => iter.last(),
            }
        }
    }

    impl<A, const REV: bool> DoubleEndedIterator for CharPatternFinder<A, REV>
    where
        A: Accessor,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            match self {
                CharPatternFinder::Pred(iter) => iter.next_back(),
                CharPatternFinder::Char(iter) => iter.next_back(),
                CharPatternFinder::Chars(iter) => iter.next_back(),
            }
        }
    }

    impl<A, const REV: bool> FusedIterator for CharPatternFinder<A, REV> where A: Accessor {}

    #[sealed::sealed]
    impl Pattern for &CharPattern {
        type Output = char;

        type Owned = Self;

        type FindAllImpl<A> = CharPatternFinder<A, false> where A: Accessor;

        type RFindAllImpl<A> = CharPatternFinder<A, true> where A: Accessor;

        fn _find_all<A>(self, accessor: A) -> Self::FindAllImpl<A>
        where
            A: Accessor,
        {
            match self {
                CharPattern::Pred(v) => {
                    CharPatternFinder::Pred(vec_to_pred(v.clone())._find_all(accessor))
                }
                CharPattern::Char(ch) => CharPatternFinder::Char(ch._find_all(accessor)),
                CharPattern::Chars(v) => CharPatternFinder::Chars(v.as_slice()._find_all(accessor)),
                CharPattern::CharsVec(v) => CharPatternFinder::Chars(v.clone()._find_all(accessor)),
                CharPattern::CharsVecRef(v) => CharPatternFinder::Chars(v._find_all(accessor)),
            }
        }

        fn _rfind_all<A>(self, accessor: A) -> Self::RFindAllImpl<A>
        where
            A: Accessor,
        {
            match self {
                CharPattern::Pred(v) => {
                    CharPatternFinder::Pred(vec_to_pred(v.clone())._rfind_all(accessor))
                }
                CharPattern::Char(ch) => CharPatternFinder::Char(ch._rfind_all(accessor)),
                CharPattern::Chars(v) => {
                    CharPatternFinder::Chars(v.as_slice()._rfind_all(accessor))
                }
                CharPattern::CharsVec(v) => {
                    CharPatternFinder::Chars(v.clone()._rfind_all(accessor))
                }
                CharPattern::CharsVecRef(v) => CharPatternFinder::Chars(v._rfind_all(accessor)),
            }
        }

        fn _is_prefix(self, haystack: &Rope) -> bool {
            match self {
                CharPattern::Pred(v) => vec_to_pred(v.clone())._is_prefix(haystack),
                CharPattern::Char(ch) => ch._is_prefix(haystack),
                CharPattern::Chars(v) => v.as_slice()._is_prefix(haystack),
                CharPattern::CharsVec(v) => v.clone()._is_prefix(haystack),
                CharPattern::CharsVecRef(v) => v._is_prefix(haystack),
            }
        }

        fn _is_suffix(self, haystack: &Rope) -> bool {
            match self {
                CharPattern::Pred(v) => vec_to_pred(v.clone())._is_suffix(haystack),
                CharPattern::Char(ch) => ch._is_suffix(haystack),
                CharPattern::Chars(v) => v.as_slice()._is_suffix(haystack),
                CharPattern::CharsVec(v) => v.clone()._is_suffix(haystack),
                CharPattern::CharsVecRef(v) => v._is_suffix(haystack),
            }
        }

        fn _convert_to_owning<A>(
            finder: &Self::FindAllImpl<A>,
        ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
        where
            A: Accessor,
        {
            match finder {
                CharPatternFinder::Pred(f) => {
                    CharPatternFinder::Pred(
                        Box::<dyn Fn(char) -> bool + 'static>::_convert_to_owning(f),
                    )
                }
                CharPatternFinder::Char(f) => CharPatternFinder::Char(char::_convert_to_owning(f)),
                CharPatternFinder::Chars(f) => {
                    CharPatternFinder::Chars(Vec::<char>::_convert_to_owning(f))
                }
            }
        }

        fn _convert_into_owning<A>(
            finder: Self::FindAllImpl<A>,
        ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
        where
            A: Accessor,
        {
            match finder {
                CharPatternFinder::Pred(f) => {
                    CharPatternFinder::Pred(
                        Box::<dyn Fn(char) -> bool + 'static>::_convert_into_owning(f),
                    )
                }
                CharPatternFinder::Char(f) => {
                    CharPatternFinder::Char(char::_convert_into_owning(f))
                }
                CharPatternFinder::Chars(f) => {
                    CharPatternFinder::Chars(Vec::<char>::_convert_into_owning(f))
                }
            }
        }

        fn _rconvert_to_owning<A>(
            finder: &Self::RFindAllImpl<A>,
        ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
        where
            A: Accessor,
        {
            match finder {
                CharPatternFinder::Pred(f) => {
                    CharPatternFinder::Pred(
                        Box::<dyn Fn(char) -> bool + 'static>::_rconvert_to_owning(f),
                    )
                }
                CharPatternFinder::Char(f) => CharPatternFinder::Char(char::_rconvert_to_owning(f)),
                CharPatternFinder::Chars(f) => {
                    CharPatternFinder::Chars(Vec::<char>::_rconvert_to_owning(f))
                }
            }
        }

        fn _rconvert_into_owning<A>(
            finder: Self::RFindAllImpl<A>,
        ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
        where
            A: Accessor,
        {
            match finder {
                CharPatternFinder::Pred(f) => {
                    CharPatternFinder::Pred(
                        Box::<dyn Fn(char) -> bool + 'static>::_rconvert_into_owning(f),
                    )
                }
                CharPatternFinder::Char(f) => {
                    CharPatternFinder::Char(char::_rconvert_into_owning(f))
                }
                CharPatternFinder::Chars(f) => {
                    CharPatternFinder::Chars(Vec::<char>::_rconvert_into_owning(f))
                }
            }
        }
    }

    pub(crate) enum CharPatternValueTree {
        Pred(prop::collection::VecValueTree<prop::char::CharValueTree>),
        Char(prop::char::CharValueTree),
        Chars(prop::collection::VecValueTree<prop::char::CharValueTree>),
        CharsVec(prop::collection::VecValueTree<prop::char::CharValueTree>),
        CharsVecRef(prop::collection::VecValueTree<prop::char::CharValueTree>),
    }

    impl prop::strategy::ValueTree for CharPatternValueTree {
        type Value = CharPattern;

        fn current(&self) -> Self::Value {
            match self {
                CharPatternValueTree::Pred(t) => CharPattern::Pred(t.current()),
                CharPatternValueTree::Char(t) => CharPattern::Char(t.current()),
                CharPatternValueTree::Chars(t) => CharPattern::Chars(t.current()),
                CharPatternValueTree::CharsVec(t) => CharPattern::CharsVec(t.current()),
                CharPatternValueTree::CharsVecRef(t) => CharPattern::CharsVecRef(t.current()),
            }
        }

        fn simplify(&mut self) -> bool {
            match self {
                CharPatternValueTree::Pred(t)
                | CharPatternValueTree::Chars(t)
                | CharPatternValueTree::CharsVec(t)
                | CharPatternValueTree::CharsVecRef(t) => t.simplify(),
                CharPatternValueTree::Char(t) => t.simplify(),
            }
        }

        fn complicate(&mut self) -> bool {
            match self {
                CharPatternValueTree::Pred(t)
                | CharPatternValueTree::Chars(t)
                | CharPatternValueTree::CharsVec(t)
                | CharPatternValueTree::CharsVecRef(t) => t.complicate(),
                CharPatternValueTree::Char(t) => t.complicate(),
            }
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub(crate) struct CharPatternStrategy;

    impl Strategy for CharPatternStrategy {
        type Tree = CharPatternValueTree;
        type Value = CharPattern;

        fn new_tree(
            &self,
            runner: &mut proptest::test_runner::TestRunner,
        ) -> proptest::strategy::NewTree<Self> {
            let char_strategy = if runner.rng().gen_bool(0.5) {
                any::<char>()
            } else {
                prop::char::range('\0', '\u{7f}')
            };

            match runner.rng().gen_range(0..5) {
                0 => Ok(CharPatternValueTree::Pred(
                    prop::collection::vec(char_strategy, 0..7).new_tree(runner)?,
                )),
                1 => Ok(CharPatternValueTree::Char(char_strategy.new_tree(runner)?)),
                2 => Ok(CharPatternValueTree::Chars(
                    prop::collection::vec(char_strategy, 0..7).new_tree(runner)?,
                )),
                3 => Ok(CharPatternValueTree::CharsVec(
                    prop::collection::vec(char_strategy, 0..7).new_tree(runner)?,
                )),
                4 => Ok(CharPatternValueTree::CharsVecRef(
                    prop::collection::vec(char_strategy, 0..7).new_tree(runner)?,
                )),
                _ => unreachable!(),
            }
        }
    }

    impl Arbitrary for CharPattern {
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            CharPatternStrategy
        }

        type Strategy = CharPatternStrategy;
    }

    #[derive(Debug, Copy, Clone)]
    pub(crate) enum StringHow {
        Str,
        String,
        StringRef,
        Rope,
        RopeRef,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct StringPattern {
        string: String,
        how: StringHow,
    }

    impl std::fmt::Display for StringPattern {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.string.fmt(f)
        }
    }

    #[sealed::sealed]
    impl<'a> Pattern for &'a StringPattern {
        type Output = ();
        type Owned = Self;

        type FindAllImpl<A> = crate::pattern::FindStr<'a, A> where A: Accessor;

        type RFindAllImpl<A> = crate::pattern::RFindStr<'a, A> where A: Accessor;

        fn _find_all<A>(self, accessor: A) -> Self::FindAllImpl<A>
        where
            A: Accessor,
        {
            match self.how {
                StringHow::Str => self.string.as_str()._find_all(accessor),
                StringHow::String => self.string.clone()._find_all(accessor),
                StringHow::StringRef => (&self.string)._find_all(accessor),
                StringHow::Rope => Rope::from(&self.string)._find_all(accessor),
                StringHow::RopeRef => (&Rope::from(&self.string))._find_all(accessor),
            }
        }

        fn _rfind_all<A>(self, accessor: A) -> Self::RFindAllImpl<A>
        where
            A: Accessor,
        {
            match self.how {
                StringHow::Str => self.string.as_str()._rfind_all(accessor),
                StringHow::String => self.string.clone()._rfind_all(accessor),
                StringHow::StringRef => (&self.string)._rfind_all(accessor),
                StringHow::Rope => Rope::from(&self.string)._rfind_all(accessor),
                StringHow::RopeRef => (&Rope::from(&self.string))._rfind_all(accessor),
            }
        }

        fn _is_prefix(self, haystack: &Rope) -> bool {
            match self.how {
                StringHow::Str => self.string.as_str()._is_prefix(haystack),
                StringHow::String => self.string.clone()._is_prefix(haystack),
                StringHow::StringRef => (&self.string)._is_prefix(haystack),
                StringHow::Rope => Rope::from(&self.string)._is_prefix(haystack),
                StringHow::RopeRef => (&Rope::from(&self.string))._is_prefix(haystack),
            }
        }

        fn _is_suffix(self, haystack: &Rope) -> bool {
            match self.how {
                StringHow::Str => self.string.as_str()._is_suffix(haystack),
                StringHow::String => self.string.clone()._is_suffix(haystack),
                StringHow::StringRef => (&self.string)._is_suffix(haystack),
                StringHow::Rope => Rope::from(&self.string)._is_suffix(haystack),
                StringHow::RopeRef => (&Rope::from(&self.string))._is_suffix(haystack),
            }
        }

        fn _convert_to_owning<A>(
            finder: &Self::FindAllImpl<A>,
        ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
        where
            A: Accessor,
        {
            finder.to_owning()
        }

        fn _convert_into_owning<A>(
            finder: Self::FindAllImpl<A>,
        ) -> <Self::Owned as Pattern>::FindAllImpl<OwningAccessor>
        where
            A: Accessor,
        {
            finder.into_owning()
        }

        fn _rconvert_to_owning<A>(
            finder: &Self::RFindAllImpl<A>,
        ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
        where
            A: Accessor,
        {
            finder.to_owning()
        }

        fn _rconvert_into_owning<A>(
            finder: Self::RFindAllImpl<A>,
        ) -> <Self::Owned as Pattern>::RFindAllImpl<OwningAccessor>
        where
            A: Accessor,
        {
            finder.into_owning()
        }
    }

    pub(crate) struct StringPatternValueTree {
        tree: prop::string::RegexGeneratorValueTree<String>,
        how: StringHow,
    }

    impl prop::strategy::ValueTree for StringPatternValueTree {
        type Value = StringPattern;

        fn current(&self) -> Self::Value {
            StringPattern {
                string: self.tree.current(),
                how: self.how,
            }
        }

        fn simplify(&mut self) -> bool {
            self.tree.simplify()
        }

        fn complicate(&mut self) -> bool {
            self.tree.complicate()
        }
    }

    #[derive(Debug, Clone)]
    pub(crate) struct StringPatternStrategy(String);

    impl Strategy for StringPatternStrategy {
        type Tree = StringPatternValueTree;

        type Value = StringPattern;

        fn new_tree(
            &self,
            runner: &mut proptest::test_runner::TestRunner,
        ) -> proptest::strategy::NewTree<Self> {
            let how = match runner.rng().gen_range(0..5) {
                0 => StringHow::Str,
                1 => StringHow::String,
                2 => StringHow::StringRef,
                3 => StringHow::Rope,
                4 => StringHow::RopeRef,
                _ => unreachable!(),
            };

            Ok(StringPatternValueTree {
                tree: prop::string::string_regex(self.0.as_str())
                    .unwrap()
                    .new_tree(runner)?,
                how,
            })
        }
    }

    pub(crate) fn string_pattern(s: &str) -> StringPatternStrategy {
        StringPatternStrategy(s.to_owned())
    }
}
