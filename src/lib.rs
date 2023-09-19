#![warn(missing_docs)]
#![warn(clippy::pedantic)]
/*! A Unicode string backed by an RRB vector.
 *
 * Similarly to the standard library [`String`] type, a [`Rope`] owns its
 * storage and guarantees that its contents are valid Unicode. Unlike
 * a [`String`], it is backed not by a [`Vec<u8>`] but by an
 * [`im::Vector<u8>`]. These in turn are backed by a balanced tree structure
 * known as an an RRB tree, which makes a wide variety of operations asymptotically
 * efficient. In particular, ropes can be cloned in constant time, and can be
 * split or concatenated in logarithmic time.
 */

pub mod accessor;
pub mod pattern;
mod validations;

#[cfg(test)]
mod test_utils;
#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "proptest"))]
pub mod proptest;

use accessor::{Accessor, BorrowingAccessor, OwningAccessor, PopVecBytes};
use im::vector;
use im::vector::Vector;
use static_cow::{IntoOwning, ToOwning};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::iter::{DoubleEndedIterator, FusedIterator, Iterator};
use std::ops::{Deref, DerefMut, Range, RangeBounds};
use std::panic::{AssertUnwindSafe, RefUnwindSafe};
use validations::{
    ends_on_utf8_boundary, next_code_point, next_code_point_reverse, run_utf8_validation,
    starts_on_utf8_boundary, utf8_char_width, utf8_is_first_byte,
};

use pattern::Pattern;
pub use validations::Utf8Error;

/// Guards against panics during unsafe rope mutation.
///
/// `VectorGuard` is returned from [`Rope::as_mut_vector`]. It implements
/// `DerefMut<Target=Vector<u8>>`. If a `VectorGuard` is dropped by panicking,
/// it will clear the vector to prevent observation of a rope containing invalid
/// UTF-8 after catching the panic.
#[derive(Debug)]
pub struct VectorGuard<'a>(&'a mut Vector<u8>);

impl<'a> Deref for VectorGuard<'a> {
    type Target = Vector<u8>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> DerefMut for VectorGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

impl<'a> Drop for VectorGuard<'a> {
    fn drop(&mut self) {
        if std::thread::panicking()
            && std::panic::catch_unwind(AssertUnwindSafe(|| self.0.clear())).is_err()
        {
            // It shouldn't be possible for clearing a `Vector<u8>` to panic,
            // but if it happens then the only safe thing to do is abort.
            std::process::abort()
        }
    }
}

/** A Unicode string backed by an RRB vector.
 *
 * See top-level crate documentation for a full introduction.
 */
#[repr(transparent)]
#[derive(Clone, Default)]
pub struct Rope {
    inner: Vector<u8>,
}

impl Rope {
    /// Constructs an empty rope.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Rope {
            inner: Vector::new(),
        }
    }

    /// Constructs a rope from a `Vector<u8>` without validating it.
    ///
    /// If you are looking for a checked version of this method, use
    /// the `TryFrom<Vector<u8>>` trait implementation.
    ///
    /// # Complexity
    /// O(1) time and space, or O(log N) if debug assertions are enabled.
    ///
    /// # Safety
    /// `v` must be valid UTF-8.
    #[must_use]
    #[inline]
    pub unsafe fn from_vector_unchecked(v: Vector<u8>) -> Rope {
        // Validating the whole rope and turning an O(1) call into O(N) is too
        // much even for debug mode, but we can check the ends and this will hit
        // a lot of common mistakes.
        debug_assert!(starts_on_utf8_boundary(&v));
        debug_assert!(ends_on_utf8_boundary(&v));
        Rope { inner: v }
    }

    #[must_use]
    #[inline]
    /// Returns a guarded, mutable reference to the rope's underlying
    /// `Vector<u8>`.
    ///
    /// # Complexity
    /// O(1) time and space.
    ///
    /// # Safety
    /// When the `VectorGuard` is dropped, the contents of the vector must be
    /// valid UTF-8. The guard will clear the vector if it is dropped by
    /// panicking, but the caller is still responsible for validity upon normal
    /// return.
    ///
    /// # Examples
    /// ```
    /// # use im_rope::{Rope,VectorGuard};
    /// # use im::vector::Vector;
    /// # use std::ops::DerefMut;
    /// # use std::panic::{catch_unwind,AssertUnwindSafe};
    /// // Add a 😀, one byte at a time.
    /// fn add_smiley(v: &mut Vector<u8>) {
    ///     v.push_back(0xf0);
    ///     v.push_back(0x9f);
    ///     v.push_back(0x98);
    ///     panic!("💩!");
    ///     v.push_back(0x80);
    /// }
    ///
    /// fn main() {
    ///     let mut rope = Rope::from("My mood today is: ");
    ///     match catch_unwind(AssertUnwindSafe(|| unsafe {
    ///         add_smiley(rope.as_mut_vector().deref_mut());
    ///     })) {
    ///         Ok(_) => {
    ///             unreachable!("We won't get here because we panicked.");
    ///             // But if we hadn't, then we'd see:
    ///             assert_eq!(rope, Rope::from("My mood today is: 😀"))
    ///         },
    ///         Err(_) => {
    ///             // Phew! The guard saved us from being able to observe the invalid UTF-8,
    ///             // and just gave us an empty rope instead. This is a weird result, but it's safe,
    ///             // and we signed ourselves up for weirdness by using `AssertUnwindSafe`.
    ///             assert_eq!(rope, Rope::new());
    ///         }
    ///     }
    /// }
    pub unsafe fn as_mut_vector(&mut self) -> VectorGuard<'_> {
        VectorGuard(&mut self.inner)
    }

    /// Gets the length of a rope, in bytes.
    ///
    /// # Complexity
    /// O(1) time and space.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_ref().len()
    }

    /// Tests whether a rope is empty.
    ///
    /// # Complexity
    /// O(1) time and space.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    /// Tests whether two ropes refer to the same content in memory.
    ///
    /// # Complexity
    /// O(1) time and space.
    #[must_use]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.as_ref().ptr_eq(&other.inner)
    }

    /// Converts a rope into its underlying `Vector<u8>`.
    ///
    /// # Complexity
    /// O(1) time and space.
    #[must_use]
    pub fn into_inner(self) -> Vector<u8> {
        self.inner
    }

    /// Clears the rope, making it empty.
    ///
    /// # Complexity
    /// O(N) time and space.
    #[inline]
    pub fn clear(&mut self) {
        // SAFETY: an empty rope is valid UTF-8.
        unsafe {
            self.as_mut_vector().clear();
        }
    }

    /// Gets an iterator over the chars of a rope.
    ///
    /// # Complexity
    ///
    /// O(1) time and space to construct the iterator. Each call to `next()` is
    /// O(1) space and amortized O(1) time, with a worst case of
    /// O(log N).
    ///
    /// # Examples
    /// ```
    /// # use im_rope::Rope;
    /// let hello = Rope::from("🦆🦢🦤");
    /// let mut chars = hello.chars();
    /// assert_eq!(chars.next(), Some('🦆'));
    /// assert_eq!(chars.next(), Some('🦢'));
    /// assert_eq!(chars.next(), Some('🦤'));
    /// assert_eq!(chars.next(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn chars(&self) -> Chars<BorrowingAccessor<'_>> {
        Chars::borrowed(self)
    }

    /// Converts a rope into an iterator over its chars.
    ///
    /// # Complexity
    ///
    /// O(1) time and space to construct the iterator. Each call to `next()` is
    /// O(1) space and amortized O(1) time, with a worst case of
    /// O(log N).
    ///
    /// # Examples
    /// ```
    /// # use im_rope::Rope;
    /// let mut chars = Rope::from("🦨🦝🦊").into_chars();
    /// assert_eq!(chars.next(), Some('🦨'));
    /// assert_eq!(chars.next(), Some('🦝'));
    /// assert_eq!(chars.next(), Some('🦊'));
    /// assert_eq!(chars.next(), None);
    /// ```

    #[must_use]
    #[inline]
    pub fn into_chars(self) -> Chars<OwningAccessor> {
        Chars::owned(self)
    }

    /// Gets an iterator over the bytes of a rope.
    ///
    /// # Complexity
    ///
    /// O(1) time and space to construct the iterator. Each call to `next()` is
    /// O(1) space and amortized O(1) time, with a worst case of O(log N).
    #[must_use]
    #[inline]
    pub fn bytes(&self) -> Bytes<BorrowingAccessor<'_>> {
        Bytes::borrowed(self)
    }

    /// Converts a rope into an iterator over its chars.
    ///
    /// # Complexity
    ///
    /// O(1) time and space to construct the iterator. Each call to `next()` is
    /// O(1) space and amortized O(1) time, with a worst case of O(log N).
    #[must_use]
    #[inline]
    pub fn into_bytes(self) -> Bytes<OwningAccessor> {
        Bytes::owned(self)
    }

    /// Gets an iterator over the chars of a rope and their indices.
    ///
    /// # Complexity
    ///
    /// O(1) time and space to construct the iterator. Each call to `next()` is
    /// O(1) space and amortized O(1) time, with a worst case of O(log N).
    /// # Examples
    /// ```
    /// # use im_rope::Rope;
    /// let hello = Rope::from("🐎🦓🦄");
    /// let mut chars = hello.char_indices();
    /// assert_eq!(chars.next(), Some((0, '🐎')));
    /// assert_eq!(chars.next(), Some((4, '🦓')));
    /// assert_eq!(chars.next(), Some((8, '🦄')));
    /// assert_eq!(chars.next(), None);
    /// ```

    #[must_use]
    #[inline]
    pub fn char_indices(&self) -> CharIndices<BorrowingAccessor<'_>> {
        CharIndices::borrowed(self)
    }

    /// Converts a rope into an iterator over its chars and their indices.
    ///
    /// # Complexity
    ///
    /// O(1) time and space to construct the iterator. Each call to `next()` is
    /// O(1) space and amortized O(1) time, with a worst case of O(log N).
    /// # Examples
    /// ```
    /// # use im_rope::Rope;
    /// let hello = Rope::from("🐎🦓🦄");
    /// let mut chars = hello.into_char_indices();
    /// assert_eq!(chars.next(), Some((0, '🐎')));
    /// assert_eq!(chars.next(), Some((4, '🦓')));
    /// assert_eq!(chars.next(), Some((8, '🦄')));
    /// assert_eq!(chars.next(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn into_char_indices(self) -> CharIndices<OwningAccessor> {
        CharIndices::owned(self)
    }

    /// Gets an iterator over contiguous chunks of a rope.
    ///
    /// The content of a rope is kept internally in 64-byte chunks. Each
    /// [`Chunk`] provided by the iterator returned from this function contains
    /// either a `&str` or a `char`. A returned `&str` is a string of all the
    /// complete characters stored contiguously in a single chunk. A returned
    /// `char` is a character whose UTF-8 encoded representation straddles two
    /// chunks.
    ///
    /// # Complexity
    ///
    /// O(1) time and space to construct the iterator. Each call to `next()` is
    /// O(1) space and amortized O(1) time, with a worst case of O(log N).
    ///
    /// # Examples
    ///
    /// In this example, 'x' takes one byte to encode while '🦀' takes four, so
    /// the entire string takes up two chunks of storage, but the 16th crab is
    /// split across the two.
    ///
    /// ```
    /// # use im_rope::{Rope,Chunk};
    /// let xx_31crabs_xx = "xx🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀xx";
    /// let rope = Rope::from(xx_31crabs_xx);
    /// let mut chunks = rope.chunks();
    ///
    /// assert_eq!(chunks.next(), Some(Chunk::Str("xx🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀")));
    /// assert_eq!(chunks.next(), Some(Chunk::Char('🦀')));
    /// assert_eq!(chunks.next(), Some(Chunk::Str("🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀xx")));
    /// assert_eq!(chunks.next(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn chunks(&self) -> Chunks<'_> {
        Chunks {
            inner: self.as_ref().leaves(),
            unconsumed_fwd: &[],
            unconsumed_back: &[],
        }
    }

    /// Checks that `index`-th byte is the first byte in a UTF-8 code point
    /// sequence or the end of the rope.
    ///
    /// The start and end of the rope (when `index == self.len()`) are
    /// considered to be boundaries.
    ///
    /// Returns `false` if `index` is greater than `self.len()`.
    ///
    /// # Complexity
    /// O(log N) time, O(1) space.
    ///
    /// # Examples
    ///
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("x💩x");
    ///
    /// assert!(rope.is_char_boundary(0));
    /// assert!(rope.is_char_boundary(1));
    /// assert!(! rope.is_char_boundary(2));
    /// assert!(! rope.is_char_boundary(3));
    /// assert!(! rope.is_char_boundary(4));
    /// assert!(rope.is_char_boundary(5));
    /// assert!(rope.is_char_boundary(6));
    /// assert!(! rope.is_char_boundary(42));
    /// ```
    #[must_use]
    pub fn is_char_boundary(&self, index: usize) -> bool {
        if index == 0 {
            return true;
        }

        match self.inner.get(index) {
            None => index == self.len(),
            Some(&b) => utf8_is_first_byte(b),
        }
    }

    /// Returns the first character of the rope, or `None` if the rope is empty.
    #[must_use]
    #[inline]
    pub fn front(&self) -> Option<char> {
        // SAFETY: we rely on `self` being valid UTF-8 so that
        // `next_code_point` will return a valid codepoint to pass to
        // `char_from_u32_debug`.
        unsafe {
            next_code_point(&mut self.as_ref().iter().copied()).map(|c| char_from_u32_debug(c))
        }
    }

    /// Removes and returns the first character of the rope, or returns `None`
    /// if the rope is empty.
    #[inline]
    pub fn pop_front(&mut self) -> Option<char> {
        // SAFETY: we rely on the invariant that `self` contains valid UTF-8.
        // `next_code_point` will keep popping bytes until it has read an
        // entire character, so if the invariant holds before this call it also
        // holds after.
        unsafe {
            let mut v = self.as_mut_vector();
            next_code_point(&mut PopVecBytes(&mut v)).map(|c| char_from_u32_debug(c))
        }
    }

    /// Prepends the character `ch` to the rope.
    #[inline]
    pub fn push_front(&mut self, ch: char) {
        let mut buf: [u8; 4] = [0; 4];
        let str = ch.encode_utf8(&mut buf);
        // SAFETY: we must prepend valid UTF-8, which we trust
        // `char::encode_utf8` to supply.
        unsafe {
            let mut v = self.as_mut_vector();
            for byte in str.bytes().rev() {
                v.push_front(byte);
            }
        }
    }

    /// Returns the last character of the rope, or `None` if the rope is empty.
    #[must_use]
    #[inline]
    pub fn back(&self) -> Option<char> {
        // SAFETY: we rely on `self` being valid UTF-8 so that
        // `next_code_point_reverse` will return a valid codepoint to pass to
        // `char_from_u32_debug`.
        unsafe {
            next_code_point_reverse(&mut self.as_ref().iter().copied())
                .map(|c| char_from_u32_debug(c))
        }
    }

    /// Removes and returns the last character of the rope, or returns `None`
    /// if the rope is empty.
    #[inline]
    pub fn pop_back(&mut self) -> Option<char> {
        // SAFETY: we rely on the invariant that `self` contains valid UTF-8.
        // `next_code_point_reverse` will keep popping bytes until it has read an
        // entire character, so if the invariant holds before this call it also
        // holds after.
        unsafe {
            let mut v = self.as_mut_vector();
            next_code_point_reverse(&mut PopVecBytes(&mut v)).map(|c| char_from_u32_debug(c))
        }
    }

    /// Appends the character `ch` to the end of the rope.
    #[inline]
    pub fn push_back(&mut self, ch: char) {
        let mut buf: [u8; 4] = [0; 4];
        let str = ch.encode_utf8(&mut buf);
        // SAFETY: we must append valid UTF-8, which we trust
        // `char::encode_utf8` to supply.
        unsafe {
            let mut v = self.as_mut_vector();
            for byte in str.bytes() {
                v.push_back(byte);
            }
        }
    }

    /// Divides one rope into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string. It must also be on a UTF-8 character boundary.
    ///
    /// The two ropes returned go from the start of the rope to `mid`, and from
    /// `mid` to the end of the rope.
    ///
    /// # Complexity
    /// O(log N) time and space.
    ///
    /// # Safety
    /// `mid` must be on a UTF-8 character boundary.
    #[must_use]
    pub unsafe fn split_at_unchecked(&self, mid: usize) -> (Rope, Rope) {
        let (a, b) = self.as_ref().clone().split_at(mid);
        debug_assert!(starts_on_utf8_boundary(&b));
        (
            Self::from_vector_unchecked(a),
            Self::from_vector_unchecked(b),
        )
    }

    /// Tries dividing one rope into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string.
    ///
    /// On success, the two ropes returned go from the start of the rope to
    /// `mid`, and from `mid` to the end of the rope.
    ///
    /// # Complexity
    /// O(log N) time and space.
    ///
    /// # Errors
    /// Errors if `mid` is not on a UTF-8 character boundary.
    ///
    /// # Examples
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("🟥🟩🟦");
    /// assert_eq!(rope.try_split_at(0), Ok(("".into(),"🟥🟩🟦".into())));
    /// assert_eq!(rope.try_split_at(4), Ok(("🟥".into(),"🟩🟦".into())));
    /// assert_eq!(rope.try_split_at(8), Ok(("🟥🟩".into(),"🟦".into())));
    /// assert_eq!(rope.try_split_at(12), Ok(("🟥🟩🟦".into(),"".into())));
    ///
    /// // Fails because it falls in middle of 🟥's encoding.
    /// assert!(rope.try_split_at(2).is_err());
    /// ```
    pub fn try_split_at(&self, mid: usize) -> Result<(Rope, Rope), Utf8BoundaryError> {
        if mid > self.len() {
            return Err(Utf8BoundaryError(mid));
        }

        let (x, y) = self.as_ref().clone().split_at(mid);

        if starts_on_utf8_boundary(&y) {
            // SAFETY: `x` and `y` must each be valid UTF-8. Per the `Rope`
            // invariant, the original string was valid, and we just verified
            // that we've split it at a character boundary. Therefore, both
            // subropes are also valid.
            unsafe {
                Ok((
                    Self::from_vector_unchecked(x),
                    Self::from_vector_unchecked(y),
                ))
            }
        } else {
            Err(Utf8BoundaryError(mid))
        }
    }

    /// Divides one rope into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string.
    ///
    /// The two ropes returned go from the start of the rope to `mid`, and from
    /// `mid` to the end of the rope.
    ///
    /// # Complexity
    /// O(log N) time and space.
    ///
    /// # Panics
    /// Panics if `mid` is not on a UTF-8 character boundary.
    ///
    /// # Examples
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("🟥🟩🟦");
    /// assert_eq!(rope.split_at(0), ("".into(), "🟥🟩🟦".into()));
    /// assert_eq!(rope.split_at(4), ("🟥".into(), "🟩🟦".into()));
    /// assert_eq!(rope.split_at(8), ("🟥🟩".into(), "🟦".into()));
    /// assert_eq!(rope.split_at(12), ("🟥🟩🟦".into(),"".into()));
    /// ```

    #[must_use]
    #[inline]
    pub fn split_at(&self, mid: usize) -> (Rope, Rope) {
        self.try_split_at(mid).unwrap()
    }

    /// Returns a subrope over the given range of bytes.
    ///
    /// # Complexity
    /// O(log N) time and space.
    ///
    /// # Safety
    /// Both sides of `range` must be UTF-8 character boundaries.
    #[must_use]
    pub unsafe fn subrope_unchecked<R: RangeBounds<usize>>(&self, range: R) -> Rope {
        let (start, end) = to_range_tuple(&range, self.len());
        let mut v = self.as_ref().skip(start);
        if cfg!(debug_assertions) {
            let junk = v.split_off(end - start);
            debug_assert!(starts_on_utf8_boundary(&junk));
        } else {
            // As of im-15.1 `truncate` isn't actually any faster than
            // `split_off`, but maybe this will get fixed at some point.
            v.truncate(end - start);
        }

        Self::from_vector_unchecked(v)
    }

    /// Tries returning a subrope over the given range of bytes.
    ///
    /// # Complexity
    /// O(log N) time and space.
    ///
    /// # Errors
    /// Errors if either side of `range` is not on a UTF-8 character boundary.
    pub fn try_subrope<R: RangeBounds<usize>>(&self, range: R) -> Result<Rope, Utf8BoundaryError> {
        let (start, end) = to_range_tuple(&range, self.len());

        if start > self.len() {
            Err(Utf8BoundaryError(start))
        } else if end > self.len() {
            Err(Utf8BoundaryError(end))
        } else if start >= end {
            Ok(Rope::new())
        } else {
            let mut v = if start > 0 {
                let v = self.as_ref().skip(start);
                if !starts_on_utf8_boundary(&v) {
                    return Err(Utf8BoundaryError(start));
                }
                v
            } else {
                self.as_ref().clone()
            };

            let sublen = end - start;

            if sublen == v.len() {
                // SAFETY: `v` must be valid UTF-8. Per the `Rope` invariant,
                // `v` is cut from a valid rope. It has the same ending as the
                // original rope, and we've just verified that it begins on a
                // character boundary, so it is therefore valid.
                unsafe { Ok(Self::from_vector_unchecked(v)) }
            } else {
                v.truncate(sublen + 1);
                // SAFETY: We truncated `v` to `sublen + 1` which must be
                // positive since `sublen` is unsigned. `v` is now in fact
                // `sublen + 1` bytes long, because the `truncate` call would
                // have panicked if the original vector were shorter than that.
                // (In fact this panic is impossible because sublen is `end -
                // start` and we already validated at the start of this method
                // that `end` is not greater than the length of the rope).
                // Therefore, `v` is non-empty an `pop_back` will never return
                // `None`.
                let back = unsafe { v.pop_back().unwrap_unchecked() };
                if utf8_is_first_byte(back) {
                    // SAFETY: `v` must be valid UTF-8. Per the `Rope`
                    // invariant, `v` is cut from a valid rope. Previously, we
                    // verified that `v` begins on a character boundary, and now
                    // we've verified that it ends on one as well, so it is
                    // therefore valid.
                    unsafe { Ok(Self::from_vector_unchecked(v)) }
                } else {
                    Err(Utf8BoundaryError(end))
                }
            }
        }
    }

    /// Returns a subrope over the given range of bytes.
    ///
    /// # Complexity
    /// O(log N) time and space.
    ///
    /// # Panics
    /// Panics if either side of `range` is not a UTF-8 character boundary.
    #[must_use]
    #[inline]
    pub fn subrope<R: RangeBounds<usize>>(&self, range: R) -> Rope {
        self.try_subrope(range)
            .expect("Both sides of `range` must be character boundaries")
    }

    /// Removes the subrope given by `range` from `self` and returns it.
    ///
    /// # Complexity
    /// O(log N) time and space.
    ///
    /// # Safety
    /// Both ends of `range` must fall on UTF-8 character boundaries.
    #[allow(clippy::return_self_not_must_use)]
    pub unsafe fn extract_unchecked<R: RangeBounds<usize>>(&mut self, range: R) -> Rope {
        let (start, end) = to_range_tuple(&range, self.len());
        let mut v = self.as_mut_vector();

        if start >= end {
            Rope::new()
        } else if end == v.len() {
            let w = v.split_off(start);
            Rope::from_vector_unchecked(w)
        } else if start == 0 {
            let mut w = v.split_off(end);
            debug_assert!(starts_on_utf8_boundary(&w));
            std::mem::swap(&mut *v, &mut w);
            Rope::from_vector_unchecked(w)
        } else {
            let mut w = v.split_off(start);
            debug_assert!(starts_on_utf8_boundary(&w));
            let u = w.split_off(end - start);
            debug_assert!(starts_on_utf8_boundary(&u));
            v.append(u);
            Rope::from_vector_unchecked(w)
        }
    }

    /// Tries removing the subrope given by `range` from `self`, and returns it.
    ///
    /// # Complexity
    /// O(log N) time and space.
    ///
    /// # Errors
    /// Errors if either end of `range` does not fall on a UTF-8 character boundary.
    /// On error, `self` is left unchanged.
    pub fn try_extract<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<Rope, Utf8BoundaryError> {
        let (start, end) = to_range_tuple(&range, self.len());

        if start > self.len() {
            Err(Utf8BoundaryError(start))
        } else if end > self.len() {
            Err(Utf8BoundaryError(end))
        } else if start >= end {
            Ok(Rope::new())
        } else if end == self.len() {
            // SAFETY: On exit from this block `v` must be valid UTF-8, and on
            // successful return `w` must be as well. The `Rope` invariant
            // assures that `v` is initially valid. After removing `w`, we check
            // whether it begins on a UTF-8 boundary; if so, this suffices that
            // both `v` and `w` are valid. If not, then `w` is re-appended to
            // `v`, restoring the original state which is valid by hypothesis.
            //
            // This technique provides a faster happy path than validating prior
            // to splitting, because we only need do one lookup rather than two
            // on the index we're splitting on.
            unsafe {
                let mut v = self.as_mut_vector();
                let w = v.split_off(start);
                if starts_on_utf8_boundary(&w) {
                    Ok(Rope::from_vector_unchecked(w))
                } else {
                    v.append(w);
                    Err(Utf8BoundaryError(start))
                }
            }
        } else if start == 0 {
            // SAFETY: same trick as above.
            unsafe {
                let mut v = self.as_mut_vector();
                let mut w = v.split_off(end);

                if starts_on_utf8_boundary(&w) {
                    std::mem::swap(&mut *v, &mut w);
                    Ok(Rope::from_vector_unchecked(w))
                } else {
                    v.append(w);
                    Err(Utf8BoundaryError(end))
                }
            }
        } else {
            // SAFETY: same as above but now we have to do it on both boundaries.
            unsafe {
                let mut v = self.as_mut_vector();
                let mut w = v.split_off(start);

                if starts_on_utf8_boundary(&w) {
                    let x = w.split_off(end - start);
                    if starts_on_utf8_boundary(&x) {
                        v.append(x);
                        Ok(Rope::from_vector_unchecked(w))
                    } else {
                        w.append(x);
                        v.append(w);
                        Err(Utf8BoundaryError(end))
                    }
                } else {
                    v.append(w);
                    Err(Utf8BoundaryError(start))
                }
            }
        }
    }

    /// Removes the subrope given by `range` from `self` and returns it.
    ///
    /// # Complexity
    /// O(log N) time and space.
    ///
    /// # Panics
    /// Panics if either end of `range` does not fall on a UTF-8 character boundary.
    ///
    /// # Examples
    /// ```
    /// # use im_rope::Rope;
    /// let mut rope = Rope::from("🟥🟩🟦");
    /// let extracted = rope.extract(4..8);
    /// assert_eq!(extracted, "🟩");
    /// assert_eq!(rope, "🟥🟦");
    /// ```
    #[inline]
    #[allow(clippy::return_self_not_must_use)]
    pub fn extract<R: RangeBounds<usize>>(&mut self, range: R) -> Rope {
        self.try_extract(range).unwrap()
    }

    /// Appends `other` to the end of `self`.
    ///
    /// # Complexity
    ///
    /// If `other` is a `Rope`, O(log (N + M)) time and space. In general, O(M +
    /// log N) time and space. Here `M` is `other.len()` and `N` is `self.len()`.
    #[inline]
    pub fn append<A: StrLike>(&mut self, other: A) {
        // SAFETY: Upon exit from this block, `v` must be valid UTF-8. The
        // `Rope` invariant assures that `v` is initially valid, and the
        // `StrLike` invariant assures that we're apppending something valid.
        // The concatenation of two valid ropes is valid.
        unsafe {
            let mut v = self.as_mut_vector();
            v.append(other.into_vector());
        }
    }

    /// Appends `other` to the end of `self`.
    ///
    /// # Complexity
    ///
    /// If `other` is a `Rope`, O(log (N + M)) time and space. In general, O(M +
    /// log N) time and space. Here `M` is `other.len()` and `N` is `self.len()`.
    ///
    /// # Safety
    ///
    /// `other` must be valid UTF-8. Note that this function is only useful for
    /// when you have an `other` which is [`BytesLike`] but not [`StrLike`]. If
    /// you already have a `StrLike` type, there is no additional cost to using
    /// the safe version of this function.
    #[inline]
    pub unsafe fn append_unchecked<O: BytesLike>(&mut self, other: O) {
        let mut v = self.as_mut_vector();
        v.append(other.into_vector());
    }

    /// Prepends `other` to the start of `self`.
    ///
    /// # Complexity
    ///
    /// If `other` is a `Rope`, O(log (N + M)) time and space. In general, O(M +
    /// log N) time and space. Here `M` is `other.len()` and `N` is `self.len()`.
    #[inline]
    pub fn prepend<A: StrLike>(&mut self, other: A) {
        let mut o = other.into_rope();
        std::mem::swap(self, &mut o);
        self.append(o);
    }

    /// Prepends `other` to the start of `self`.
    ///
    /// # Complexity
    ///
    /// If `other` is a `Rope`, O(log (N + M)) time and space. In general, O(M +
    /// log N) time and space. Here `M` is `other.len()` and `N` is `self.len()`.
    ///
    /// # Safety
    ///
    /// `other` must be valid UTF-8. Note that this function is only useful for
    /// when you have an `other` which is [`BytesLike`] but not [`StrLike`]. If
    /// you already have a `StrLike` type, there is no additional cost to using
    /// the safe version of this function.
    #[inline]
    pub unsafe fn prepend_unchecked<O: BytesLike>(&mut self, other: O) {
        let mut o = Rope::from_vector_unchecked(other.into_vector());
        std::mem::swap(self, &mut o);
        self.append(o);
    }

    /// Inserts `other` into `self` at position `at`.
    ///
    /// # Complexity
    ///
    /// If `other` is a `Rope`, O(log (N + M)) time and space. In general, O(M +
    /// log N) time and space. Here `M` is `other.len()` and `N` is `self.len()`.
    ///
    /// # Safety
    ///
    /// `other` must be valid UTF-8, and `at` must be a UTF-8 character boundary.
    #[inline]
    pub unsafe fn insert_unchecked<O: BytesLike>(&mut self, at: usize, other: O) {
        let mut v = self.as_mut_vector();
        let w = v.split_off(at);
        debug_assert!(starts_on_utf8_boundary(&w));
        v.append(other.into_vector());
        v.append(w);
    }

    /// Tries inserting `other` into `self` at position `at`.
    ///
    /// # Complexity
    ///
    /// If `other` is a `Rope`, O(log (N + M)) time and space. In general, O(M +
    /// log N) time and space. Here `M` is `other.len()` and `N` is `self.len()`.
    ///
    /// # Errors
    /// Errors if `at` is not on a UTF-8 character boundary. On error, `self`
    /// is left unchanged.
    pub fn try_insert<A: StrLike>(&mut self, at: usize, other: A) -> Result<(), Utf8BoundaryError> {
        // SAFETY: On exit from this block `v` must be valid UTF-8. The `Rope`
        // invariant assures that `v` is initially valid. After splitting `w`,
        // we check whether it begins on a UTF-8 boundary; if so, this suffices
        // that both `v` and `w` are valid. If not, then `w` is re-appended to
        // `v`, restoring the original state which is valid by hypothesis. In
        // the sucessful case, we first append `other`, which is valid by the
        // `StrLike` invariant, then we append `w`. Appending valid UTF-8
        // strings to each other gives vaild UTF-8.
        //
        // This technique provides a faster happy path than validating prior to
        // splitting, because we only need do one lookup rather than two on the
        // index we're splitting on.
        unsafe {
            let mut v = self.as_mut_vector();

            let w = v.split_off(at);
            if starts_on_utf8_boundary(&w) {
                v.append(other.into_vector());
                v.append(w);
                Ok(())
            } else {
                v.append(w);
                Err(Utf8BoundaryError(at))
            }
        }
    }

    /// Inserts `other` into `self` at position `at`.
    ///
    /// # Complexity
    ///
    /// If `other` is a `Rope`, O(log (N + M)) time and space. In general, O(M +
    /// log N) time and space. Here `M` is `other.len()` and `N` is `self.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `at` is not a UTF-8 character boundary.
    ///
    /// # Examples
    /// ```
    /// # use im_rope::Rope;
    /// # let mut rope = Rope::from("one three");
    /// rope.insert(4, "two ");
    /// assert_eq!(rope, Rope::from("one two three"));
    /// ```
    #[inline]
    pub fn insert<A: StrLike>(&mut self, at: usize, other: A) {
        self.try_insert(at, other).unwrap();
    }

    /// Searches forward for all occurences of `needle` within `self`.
    ///
    /// Returns an iterator over [`Range<usize>`], giving the locations of
    /// matches.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    #[inline]
    #[must_use]
    pub fn find_all<P>(&self, needle: P) -> FindAll<BorrowingAccessor<'_>, P>
    where
        P: Pattern,
    {
        FindAll {
            inner: needle._find_all(BorrowingAccessor::new(self.as_ref())),
        }
    }

    /// Searches backward for all occurences of `needle` within `self`.
    ///
    /// Returns an iterator over [`Range<usize>`], giving the locations of
    /// matches.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    #[inline]
    #[must_use]
    pub fn rfind_all<P>(&self, needle: P) -> RFindAll<BorrowingAccessor<'_>, P>
    where
        P: Pattern,
    {
        RFindAll {
            inner: needle._rfind_all(BorrowingAccessor::new(self.as_ref())),
        }
    }

    /// Searches forward for the first occurence of `needle` within
    /// `self`.
    ///
    /// If any is found, returns a [`Range<usize>`] giving its location.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    #[inline]
    #[must_use]
    pub fn find<P>(&self, needle: P) -> Option<(Range<usize>, P::Output)>
    where
        P: Pattern,
    {
        self.find_all(needle).next()
    }

    /// Searches backward for the first occurence of `needle` within
    /// `self`.
    ///
    /// If any is found, returns a [`Range<usize>`] giving its location.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    #[inline]
    #[must_use]
    pub fn rfind<P>(&self, needle: P) -> Option<(Range<usize>, P::Output)>
    where
        P: Pattern,
    {
        self.rfind_all(needle).next()
    }

    /// Returns `true` iff `self` the beginning of `self` matches `needle`.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    #[inline]
    #[must_use]
    pub fn starts_with<P>(&self, needle: P) -> bool
    where
        P: Pattern,
    {
        needle._is_prefix(self)
    }

    /// Returns `true` iff `self` the end of `self` matches `needle`.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    #[inline]
    #[must_use]
    pub fn ends_with<P>(&self, needle: P) -> bool
    where
        P: Pattern,
    {
        needle._is_suffix(self)
    }

    /// Returns a forward iterator over subropes of `self` separated by
    /// `delimiter`.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("one,two,three");
    /// let mut tokens = rope.split(',');
    /// assert_eq!(tokens.next(), Some("one".into()));
    /// assert_eq!(tokens.next(), Some("two".into()));
    /// assert_eq!(tokens.next(), Some("three".into()));
    /// assert_eq!(tokens.next(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn split<P>(&self, delimiter: P) -> Split<BorrowingAccessor<'_>, P>
    where
        P: Pattern,
    {
        Split::new(self, delimiter, 0)
    }

    /// Returns a forward iterator over the first `limit` subropes of
    /// `self` separated by `delimiter`.
    ///
    /// After `limit` delimiters have been encountered, the remainder of the
    /// rope will be returned as a single subrope.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("one,two,three,four");
    /// let mut tokens = rope.splitn(3, ',');
    /// assert_eq!(tokens.next(), Some("one".into()));
    /// assert_eq!(tokens.next(), Some("two".into()));
    /// assert_eq!(tokens.next(), Some("three,four".into()));
    /// assert_eq!(tokens.next(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn splitn<P>(&self, limit: usize, delimiter: P) -> SplitN<BorrowingAccessor<'_>, P>
    where
        P: Pattern,
    {
        SplitN::new(self, delimiter, limit)
    }

    /// Returns a forward iterator over subropes of `self` separated or
    /// terminated by `terminator`.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use im_rope::Rope;
    /// let rope =  Rope::from("one;two;three;");
    /// let mut tokens = rope.split_terminator(';');
    /// assert_eq!(tokens.next(), Some("one".into()));
    /// assert_eq!(tokens.next(), Some("two".into()));
    /// assert_eq!(tokens.next(), Some("three".into()));
    /// assert_eq!(tokens.next(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn split_terminator<P>(&self, terminator: P) -> SplitTerminator<BorrowingAccessor<'_>, P>
    where
        P: Pattern,
    {
        SplitTerminator::new(self, terminator, 0)
    }

    /// Returns a forward iterator over subropes of `self` separated or
    /// terminated by `delimiter`. The delimiter will be included in the
    /// returned subropes.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("one;two;three;");
    /// let mut tokens = rope.split_inclusive(';');
    /// assert_eq!(tokens.next(), Some("one;".into()));
    /// assert_eq!(tokens.next(), Some("two;".into()));
    /// assert_eq!(tokens.next(), Some("three;".into()));
    /// assert_eq!(tokens.next(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn split_inclusive<P>(&self, delimiter: P) -> SplitInclusive<BorrowingAccessor<'_>, P>
    where
        P: Pattern,
    {
        SplitInclusive::new(self, delimiter, 0)
    }

    /// Returns a backward iterator over subropes of `self` separated by
    /// `delimiter`.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    /// # Examples
    ///
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("one,two,three");
    /// let mut tokens = rope.rsplit(',');
    /// assert_eq!(tokens.next(), Some("three".into()));
    /// assert_eq!(tokens.next(), Some("two".into()));
    /// assert_eq!(tokens.next(), Some("one".into()));
    /// assert_eq!(tokens.next(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn rsplit<P>(&self, delimiter: P) -> RSplit<BorrowingAccessor<'_>, P>
    where
        P: Pattern,
    {
        RSplit::new(self, delimiter, 0)
    }

    /// Returns a backward iterator over the first `limit` subropes of
    /// `self` separated by `delimiter`.
    ///
    /// After `limit` delimiters have been encountered, the remainder of the
    /// rope will be returned as a single subrope.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("one,two,three,four");
    /// let mut tokens = rope.rsplitn(3, ',');
    /// assert_eq!(tokens.next(), Some("four".into()));
    /// assert_eq!(tokens.next(), Some("three".into()));
    /// assert_eq!(tokens.next(), Some("one,two".into()));
    /// assert_eq!(tokens.next(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn rsplitn<P>(&self, limit: usize, delimiter: P) -> RSplitN<BorrowingAccessor<'_>, P>
    where
        P: Pattern,
    {
        RSplitN::new(self, delimiter, limit)
    }

    /// Returns a backward iterator over subropes of `self` separated or
    /// terminated by `terminator`.
    ///
    /// # Complexity
    /// Dependent on the type of [`Pattern`] used; see its documentation for
    /// more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("one;two;three;");
    /// let mut tokens = rope.rsplit_terminator(';');
    /// assert_eq!(tokens.next(), Some("three".into()));
    /// assert_eq!(tokens.next(), Some("two".into()));
    /// assert_eq!(tokens.next(), Some("one".into()));
    /// assert_eq!(tokens.next(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn rsplit_terminator<P>(&self, terminator: P) -> RSplitTerminator<BorrowingAccessor<'_>, P>
    where
        P: Pattern,
    {
        RSplitTerminator::new(self, terminator, 0)
    }

    /// Splits `self` into lines.
    ///
    /// Lines are ended with either a newline (\n) or a carriage return with a
    /// line feed (\r\n).
    ///
    /// The final line ending is optional. A string that ends with a final line
    /// ending will return the same lines as an otherwise identical string
    /// without a final line ending.
    ///
    /// # Complexity
    /// O(N log N) time for the search, O(log M) space to construct each line,
    /// where M is the length of the line.
    ///
    /// # Examples
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("one\ntwo\r\nthree");
    /// let mut lines = rope.lines();
    /// assert_eq!(lines.next(), Some("one".into()));
    /// assert_eq!(lines.next(), Some("two".into()));
    /// assert_eq!(lines.next(), Some("three".into()));
    /// assert_eq!(lines.next(), None);
    /// ```
    ///
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("one\ntwo\r\nthree\n");
    /// let mut lines = rope.lines();
    /// assert_eq!(lines.next(), Some("one".into()));
    /// assert_eq!(lines.next(), Some("two".into()));
    /// assert_eq!(lines.next(), Some("three".into()));
    /// assert_eq!(lines.next(), None);
    /// ```
    ///
    /// A bare carriage return at the end of the line will not be stripped. This
    /// is contrast to the behavior [`std::str::Lines`], which as of Rust 1.66
    /// still has this as a bug (which is expeccted to be fixed).
    /// ```
    /// # use im_rope::Rope;
    /// let rope = Rope::from("one\ntwo\r\nthree\r");
    /// let mut lines = rope.lines();
    /// assert_eq!(lines.next(), Some("one".into()));
    /// assert_eq!(lines.next(), Some("two".into()));
    /// assert_eq!(lines.next(), Some("three\r".into()));
    /// assert_eq!(lines.next(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn lines(&self) -> Lines<BorrowingAccessor<'_>> {
        Lines::borrowed(self)
    }
}

macro_rules! reverse_cmp {
    ($ty:ty) => {
        impl PartialEq<Rope> for $ty {
            fn eq(&self, other: &Rope) -> bool {
                other.eq(self)
            }
        }

        impl PartialOrd<Rope> for $ty {
            fn partial_cmp(&self, other: &Rope) -> Option<Ordering> {
                other.partial_cmp(self).map(|o| o.reverse())
            }
        }
    };
}

impl PartialEq<[u8]> for Rope {
    fn eq(&self, mut other: &[u8]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for chunk in self.as_ref().leaves() {
            if chunk.ne(&other[..chunk.len()]) {
                return false;
            }
            other = &other[chunk.len()..];
        }

        true
    }
}

impl PartialOrd<[u8]> for Rope {
    #[allow(clippy::redundant_else)]
    fn partial_cmp(&self, mut other: &[u8]) -> Option<Ordering> {
        for chunk in self.inner.leaves() {
            if chunk.len() > other.len() {
                match chunk[..other.len()].cmp(other) {
                    Ordering::Equal => return Some(Ordering::Greater),
                    ord => return Some(ord),
                }
            } else {
                match chunk.cmp(&other[..chunk.len()]) {
                    Ordering::Equal => {
                        other = &other[chunk.len()..];
                    }
                    ord => return Some(ord),
                }
            }
        }
        if other.is_empty() {
            Some(Ordering::Equal)
        } else {
            Some(Ordering::Less)
        }
    }
}

reverse_cmp!([u8]);

impl PartialEq<str> for Rope {
    fn eq(&self, other: &str) -> bool {
        self.eq(other.as_bytes())
    }
}

impl PartialOrd<str> for Rope {
    fn partial_cmp(&self, other: &str) -> Option<Ordering> {
        self.partial_cmp(other.as_bytes())
    }
}

reverse_cmp!(str);

impl PartialEq<&str> for Rope {
    fn eq(&self, other: &&str) -> bool {
        self.eq(other.as_bytes())
    }
}

impl PartialOrd<&str> for Rope {
    fn partial_cmp(&self, other: &&str) -> Option<Ordering> {
        self.partial_cmp(other.as_bytes())
    }
}

reverse_cmp!(&str);

impl PartialEq<Vec<u8>> for Rope {
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.eq(other.as_slice())
    }
}

impl PartialOrd<Vec<u8>> for Rope {
    fn partial_cmp(&self, other: &Vec<u8>) -> Option<Ordering> {
        self.partial_cmp(other.as_slice())
    }
}

reverse_cmp!(Vec<u8>);

impl PartialEq<String> for Rope {
    fn eq(&self, other: &String) -> bool {
        self.eq(other.as_bytes())
    }
}

impl PartialOrd<String> for Rope {
    fn partial_cmp(&self, other: &String) -> Option<Ordering> {
        self.partial_cmp(other.as_bytes())
    }
}

reverse_cmp!(String);

// The PartialOrd and PartialEq implementations for Vector<A> just iterate
// over each element. Because we're specialized on u8, we can speed things
// up by comparing chunks at a time.
impl PartialOrd<Vector<u8>> for Rope {
    #[allow(clippy::redundant_else)]
    fn partial_cmp(&self, other: &Vector<u8>) -> Option<Ordering> {
        let mut self_iter = self.as_ref().leaves();
        let mut other_iter = other.leaves();

        let mut self_chunk: &[u8] = &[];
        let mut other_chunk: &[u8] = &[];

        loop {
            match self_chunk.len().cmp(&other_chunk.len()) {
                Ordering::Less => match self_chunk.cmp(&other_chunk[..self_chunk.len()]) {
                    Ordering::Equal => {
                        let self_len = self_chunk.len();
                        self_chunk = match next_nonempty(&mut self_iter) {
                            None => return Some(Ordering::Less),
                            Some(chunk) => chunk,
                        };
                        other_chunk = &other_chunk[self_len..];
                    }
                    ord => return Some(ord),
                },
                Ordering::Equal => match self_chunk.cmp(other_chunk) {
                    Ordering::Equal => {
                        self_chunk = match next_nonempty(&mut self_iter) {
                            None => {
                                if next_nonempty(&mut other_iter).is_some() {
                                    return Some(Ordering::Less);
                                } else {
                                    return Some(Ordering::Equal);
                                }
                            }
                            Some(chunk) => chunk,
                        };

                        other_chunk = match next_nonempty(&mut other_iter) {
                            None => return Some(Ordering::Greater),
                            Some(chunk) => chunk,
                        }
                    }

                    ord => return Some(ord),
                },

                Ordering::Greater => match self_chunk[..other_chunk.len()].cmp(other_chunk) {
                    Ordering::Equal => {
                        self_chunk = &self_chunk[other_chunk.len()..];
                        other_chunk = match next_nonempty(&mut other_iter) {
                            None => return Some(Ordering::Greater),
                            Some(chunk) => chunk,
                        }
                    }
                    ord => return Some(ord),
                },
            }
        }
    }
}

impl PartialEq<Vector<u8>> for Rope {
    fn eq(&self, other: &Vector<u8>) -> bool {
        if self.inner.ptr_eq(other) {
            true
        } else if self.inner.len() != other.len() {
            false
        } else {
            self.partial_cmp(other).unwrap() == Ordering::Equal
        }
    }
}

reverse_cmp!(Vector<u8>);

impl PartialEq<Rope> for Rope {
    fn eq(&self, other: &Rope) -> bool {
        self.eq(&other.inner)
    }
}

impl Eq for Rope {}

impl PartialOrd<Rope> for Rope {
    fn partial_cmp(&self, other: &Rope) -> Option<Ordering> {
        self.partial_cmp(&other.inner)
    }
}

impl Ord for Rope {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl std::hash::Hash for Rope {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        for b in self.bytes() {
            state.write_u8(b);
        }
    }
}

impl From<String> for Rope {
    fn from(s: String) -> Self {
        //SAFETY: `String`s guarantee the UTF-8 validity of their contents.
        unsafe { Self::from_vector_unchecked(s.into_bytes().into()) }
    }
}

impl From<&str> for Rope {
    fn from(s: &str) -> Self {
        //SAFETY: `str`s guarantee the UTF-8 validity of their contents.
        unsafe { Self::from_vector_unchecked(s.as_bytes().into()) }
    }
}

impl From<&String> for Rope {
    fn from(s: &String) -> Self {
        //SAFETY: `String`s guarantee the UTF-8 validity of their contents.
        unsafe { Self::from_vector_unchecked(s.as_bytes().into()) }
    }
}

impl From<char> for Rope {
    fn from(ch: char) -> Self {
        let mut buf: [u8; 4] = Default::default();
        let str = ch.encode_utf8(&mut buf);
        Rope::from(str as &str)
    }
}

impl From<&char> for Rope {
    fn from(ch: &char) -> Self {
        Self::from(*ch)
    }
}

impl TryFrom<Vector<u8>> for Rope {
    type Error = FromUtf8Error;
    fn try_from(v: Vector<u8>) -> Result<Self, Self::Error> {
        match run_utf8_validation(&v) {
            Ok(()) => unsafe {
                // SAFETY: `v` must be valid UTF-8, which we just finished
                // checking.
                Ok(Self::from_vector_unchecked(v))
            },
            Err(e) => Err(FromUtf8Error {
                vector: v,
                error: e,
            }),
        }
    }
}

impl TryFrom<&Vector<u8>> for Rope {
    type Error = FromUtf8Error;
    fn try_from(v: &Vector<u8>) -> Result<Self, Self::Error> {
        Self::try_from(v.clone())
    }
}

impl TryFrom<Vec<u8>> for Rope {
    type Error = std::string::FromUtf8Error;
    fn try_from(v: Vec<u8>) -> Result<Self, Self::Error> {
        Ok(Self::from(String::from_utf8(v)?))
    }
}

impl TryFrom<&Vec<u8>> for Rope {
    type Error = Utf8Error;
    fn try_from(v: &Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(v.as_slice())
    }
}

impl TryFrom<&[u8]> for Rope {
    type Error = Utf8Error;
    fn try_from(v: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self::from(std::str::from_utf8(v)?))
    }
}

impl From<&Rope> for Vec<u8> {
    fn from(t: &Rope) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::with_capacity(t.len());
        for chunk in t.as_ref().leaves() {
            v.extend_from_slice(chunk);
        }
        v
    }
}

impl From<Rope> for Vec<u8> {
    fn from(t: Rope) -> Self {
        (&t).into()
    }
}

impl From<&Rope> for Vector<u8> {
    fn from(value: &Rope) -> Self {
        value.clone().into_inner()
    }
}

impl From<Rope> for Vector<u8> {
    fn from(value: Rope) -> Self {
        value.into_inner()
    }
}

impl From<&Rope> for String {
    fn from(t: &Rope) -> String {
        // SAFETY: The `Rope` invariant guarantees that we have valid UTF-8.
        unsafe { string_from_utf8_debug(t.into()) }
    }
}

impl From<Rope> for String {
    fn from(t: Rope) -> String {
        String::from(&t)
    }
}

impl AsRef<Vector<u8>> for Rope {
    fn as_ref(&self) -> &Vector<u8> {
        &self.inner
    }
}

impl Borrow<Vector<u8>> for Rope {
    fn borrow(&self) -> &Vector<u8> {
        &self.inner
    }
}

impl<A> Extend<A> for Rope
where
    A: StrLike,
{
    fn extend<T: IntoIterator<Item = A>>(&mut self, iter: T) {
        for item in iter {
            self.append(item);
        }
    }
}

impl<A> FromIterator<A> for Rope
where
    A: StrLike,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let mut rope = Rope::new();
        rope.extend(iter);
        rope
    }
}

impl std::ops::Add<Rope> for Rope {
    type Output = Rope;

    fn add(mut self, rhs: Rope) -> Self::Output {
        self.append(rhs);
        self
    }
}

impl<'a> std::ops::Add<&'a Rope> for Rope {
    type Output = Rope;

    fn add(mut self, rhs: &Rope) -> Self::Output {
        self.append(rhs);
        self
    }
}

impl<'a> std::ops::Add<Rope> for &'a Rope {
    type Output = Rope;

    fn add(self, rhs: Rope) -> Self::Output {
        let mut out = self.clone();
        out.append(rhs);
        out
    }
}

impl<'a> std::ops::Add<&'a Rope> for &'a Rope {
    type Output = Rope;

    fn add(self, rhs: &Rope) -> Self::Output {
        let mut out = self.clone();
        out.append(rhs);
        out
    }
}

impl<T> std::ops::AddAssign<T> for Rope
where
    T: StrLike,
{
    fn add_assign(&mut self, rhs: T) {
        self.append(rhs);
    }
}

impl std::fmt::Debug for Rope {
    // See <https://github.com/rust-lang/rust/issues/107035> for why this
    // method is annoyingly complicated.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("\"")?;

        for chunk in self.chunks() {
            match chunk {
                Chunk::Str(str) => {
                    for ch in str.chars() {
                        if ch == '\'' {
                            f.write_str("'")?;
                        } else {
                            std::fmt::Display::fmt(&ch.escape_debug(), f)?;
                        }
                    }
                }
                Chunk::Char(ch) => {
                    if ch == '\'' {
                        f.write_str("'")?;
                    } else {
                        std::fmt::Display::fmt(&ch.escape_debug(), f)?;
                    }
                }
            }
        }

        f.write_str("\"")
    }
}

impl std::fmt::Display for Rope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for chunk in self.chunks() {
            match chunk {
                Chunk::Str(str) => str.fmt(f)?,
                Chunk::Char(c) => c.fmt(f)?,
            }
        }
        Ok(())
    }
}

impl std::fmt::Write for Rope {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.append(s);
        Ok(())
    }
}

#[cfg(any(test, feature = "serde"))]
impl serde::Serialize for Rope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

#[cfg(any(test, feature = "serde"))]
impl<'de> serde::Deserialize<'de> for Rope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct StrVisitor;

        impl<'de> serde::de::Visitor<'de> for StrVisitor {
            type Value = Rope;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a string")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Rope::from(v))
            }
        }

        deserializer.deserialize_str(StrVisitor)
    }
}

#[cfg(any(test, feature = "proptest"))]
impl ::proptest::arbitrary::Arbitrary for Rope {
    type Parameters = self::proptest::RopeParam;
    type Strategy = self::proptest::RopeStrategy;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        self::proptest::RopeStrategy(args)
    }
}

/// A trait for types that can be converted into a [`Vector<u8>`].
///
/// This trait acts as a bound for certain unsafe functions that construct or
/// insert into `Rope`s.
pub trait BytesLike: RefUnwindSafe + Sized {
    /// Convert `self` to a [`Vector<u8>`].
    fn into_vector(self) -> Vector<u8>;
}

impl BytesLike for Vector<u8> {
    fn into_vector(self) -> Vector<u8> {
        self
    }
}

impl<'a> BytesLike for &'a Vector<u8> {
    fn into_vector(self) -> Vector<u8> {
        self.clone()
    }
}

impl BytesLike for Rope {
    fn into_vector(self) -> Vector<u8> {
        self.into_inner()
    }
}

impl BytesLike for &Rope {
    fn into_vector(self) -> Vector<u8> {
        self.as_ref().clone()
    }
}

impl<'a> BytesLike for &'a [u8] {
    fn into_vector(self) -> Vector<u8> {
        self.into()
    }
}

impl BytesLike for Vec<u8> {
    fn into_vector(self) -> Vector<u8> {
        self.into()
    }
}

impl BytesLike for &Vec<u8> {
    fn into_vector(self) -> Vector<u8> {
        self.into()
    }
}

impl<'a> BytesLike for &'a str {
    fn into_vector(self) -> Vector<u8> {
        self.as_bytes().into_vector()
    }
}

impl BytesLike for String {
    fn into_vector(self) -> Vector<u8> {
        self.as_bytes().into_vector()
    }
}

impl<'a> BytesLike for &'a String {
    fn into_vector(self) -> Vector<u8> {
        self.as_bytes().into_vector()
    }
}

impl BytesLike for char {
    fn into_vector(self) -> Vector<u8> {
        let mut buf: [u8; 4] = Default::default();
        self.encode_utf8(&mut buf).into_vector()
    }
}

impl<'a> BytesLike for &'a char {
    fn into_vector(self) -> Vector<u8> {
        (*self).into_vector()
    }
}
/// A trait for types that can be converted into a [`Vector<u8>`] containing
/// valid UTF-8.
///
/// This trait acts as a bound for certain safe functions that construct or
/// insert into [`Rope`]s.
///
/// # Safety
/// This trait is `unsafe` because implementors must guarantee that their
/// implementation of [`BytesLike::into_vector`] always returns valid UTF-8.
pub unsafe trait StrLike: BytesLike {
    /// Convert `self` to a `Rope`.
    fn into_rope(self) -> Rope {
        unsafe { Rope::from_vector_unchecked(self.into_vector()) }
    }
}

// SAFETY: each of these types guarantees the UTF-8 validity of its contents.
unsafe impl StrLike for Rope {}
unsafe impl<'a> StrLike for &'a Rope {}
unsafe impl<'a> StrLike for &'a str {}
unsafe impl StrLike for String {}
unsafe impl<'a> StrLike for &'a String {}
unsafe impl StrLike for char {}
unsafe impl<'a> StrLike for &'a char {}

/// A possible error value when converting a [`Vector<u8>`] into a [`Rope`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FromUtf8Error {
    vector: Vector<u8>,
    error: Utf8Error,
}

impl FromUtf8Error {
    /// Returns a reference to the vector that was being converted.
    #[inline]
    #[must_use]
    pub fn as_vector(&self) -> &Vector<u8> {
        &self.vector
    }

    /// Returns the vector that was being converted.
    #[inline]
    #[must_use]
    pub fn into_vector(self) -> Vector<u8> {
        self.vector
    }

    /// Returns the details of the conversion error.
    #[inline]
    #[must_use]
    pub fn utf8_error(&self) -> Utf8Error {
        self.error
    }
}

impl std::fmt::Display for FromUtf8Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.error, f)
    }
}

impl std::error::Error for FromUtf8Error {}

/// An error returned when attempting to divide a [`Rope`] at a location that is
/// not a UTF-8 character boundary.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Utf8BoundaryError(usize);

impl Utf8BoundaryError {
    /// Returns the location where the split was attempted.
    #[inline]
    #[must_use]
    pub fn location(&self) -> usize {
        self.0
    }
}

impl std::fmt::Display for Utf8BoundaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Index {} is not at a UTF-8 character boundary", self.0)
    }
}

impl std::error::Error for Utf8BoundaryError {}

///An iterator over the bytes of a rope.
///
/// Call [`bytes`](Rope::bytes) or [`into_bytes`](Rope::into_bytes) to construct
/// one.
pub struct Bytes<A> {
    inner: A,
}

impl<'a> Bytes<BorrowingAccessor<'a>> {
    fn borrowed(rope: &'a Rope) -> Bytes<BorrowingAccessor<'a>> {
        Bytes {
            inner: BorrowingAccessor::new(rope.as_ref()),
        }
    }
}

impl Bytes<OwningAccessor> {
    fn owned(rope: Rope) -> Bytes<OwningAccessor> {
        Bytes {
            inner: OwningAccessor::new(rope.into()),
        }
    }
}

impl<A> ToOwning for Bytes<A>
where
    A: Accessor,
{
    type Owning = Bytes<A::Owning>;

    fn to_owning(&self) -> Self::Owning {
        Bytes {
            inner: self.inner.to_owning(),
        }
    }
}

impl<A> IntoOwning for Bytes<A>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        Bytes {
            inner: self.inner.into_owning(),
        }
    }
}

impl<A> Iterator for Bytes<A>
where
    A: Accessor,
{
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.inner.front_byte()?.1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.back_index() - self.inner.front_index();
        (len, Some(len))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<A> DoubleEndedIterator for Bytes<A>
where
    A: Accessor,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.inner.back_byte()?.1)
    }
}

impl<A> FusedIterator for Bytes<A> where A: Accessor {}

/** An iterator over the [`char`]s of a [`Rope`].
 *
 * Call [`Rope::chars`] to construct one. */
pub struct Chars<A> {
    inner: A,
}

impl<'a> Chars<BorrowingAccessor<'a>> {
    fn borrowed(rope: &'a Rope) -> Chars<BorrowingAccessor<'a>> {
        Chars {
            inner: BorrowingAccessor::new(rope.as_ref()),
        }
    }
}

impl Chars<OwningAccessor> {
    fn owned(rope: Rope) -> Chars<OwningAccessor> {
        Chars {
            inner: OwningAccessor::new(rope.into()),
        }
    }
}

impl<A> ToOwning for Chars<A>
where
    A: Accessor,
{
    type Owning = Chars<A::Owning>;

    fn to_owning(&self) -> Self::Owning {
        Chars {
            inner: self.inner.to_owning(),
        }
    }
}

impl<A> IntoOwning for Chars<A>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        Chars {
            inner: self.inner.into_owning(),
        }
    }
}

impl<A> Iterator for Chars<A>
where
    A: Accessor,
{
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        // SAFETY: `Rope` invariant says iteration is over a valid UTF-8 string and
        // the resulting `ch` is a valid Unicode Scalar Value.
        unsafe { next_code_point(&mut self.inner.byte_iter()).map(|ch| char_from_u32_debug(ch)) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.back_index() - self.inner.front_index();
        (len.saturating_add(3) / 4, Some(len))
    }

    #[inline]
    fn last(mut self) -> Option<char> {
        // No need to go through the entire string.
        self.next_back()
    }
}

impl<A> DoubleEndedIterator for Chars<A>
where
    A: Accessor,
{
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        // SAFETY: `Rope` invariant says iteration is over a valid UTF-8 string and
        // the resulting `ch` is a valid Unicode Scalar Value.
        unsafe {
            next_code_point_reverse(&mut self.inner.byte_iter()).map(|ch| char_from_u32_debug(ch))
        }
    }
}

impl<A> FusedIterator for Chars<A> where A: Accessor {}
/** An iterator over the [`char`]s of a [`Rope`] and their indices.
 *
 * Call [`Rope::char_indices`] to construct one.
 */
pub struct CharIndices<A> {
    inner: A,
}

impl<A> CharIndices<A>
where
    A: Accessor,
{
    fn new(accessor: A) -> CharIndices<A> {
        CharIndices { inner: accessor }
    }
}

impl<'a> CharIndices<BorrowingAccessor<'a>> {
    fn borrowed(rope: &'a Rope) -> CharIndices<BorrowingAccessor<'a>> {
        CharIndices {
            inner: BorrowingAccessor::new(rope.as_ref()),
        }
    }
}

impl CharIndices<OwningAccessor> {
    fn owned(rope: Rope) -> CharIndices<OwningAccessor> {
        CharIndices {
            inner: OwningAccessor::new(rope.into()),
        }
    }
}

impl<A> ToOwning for CharIndices<A>
where
    A: Accessor,
{
    type Owning = CharIndices<A::Owning>;

    fn to_owning(&self) -> Self::Owning {
        CharIndices {
            inner: self.inner.to_owning(),
        }
    }
}

impl<A> IntoOwning for CharIndices<A>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        CharIndices {
            inner: self.inner.into_owning(),
        }
    }
}

impl<A> Iterator for CharIndices<A>
where
    A: Accessor,
{
    type Item = (usize, char);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.inner.front_index();
        // SAFETY: `Rope` invariant says iteration is over a valid UTF-8 string and
        // the resulting `ch` is a valid Unicode Scalar Value.
        unsafe {
            next_code_point(&mut self.inner.byte_iter()).map(|ch| (i, char_from_u32_debug(ch)))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.back_index() - self.inner.front_index();
        (len.saturating_add(3) / 4, Some(len))
    }

    #[inline]
    fn last(mut self) -> Option<(usize, char)> {
        // No need to go through the entire string.
        self.next_back()
    }
}

impl<A> DoubleEndedIterator for CharIndices<A>
where
    A: Accessor,
{
    #[inline]
    fn next_back(&mut self) -> Option<(usize, char)> {
        // SAFETY: `Rope` invariant says iteration is over a valid UTF-8 string and
        // the resulting `ch` is a valid Unicode Scalar Value.
        unsafe {
            next_code_point_reverse(&mut self.inner.byte_iter())
                .map(|ch| (self.inner.back_index(), char_from_u32_debug(ch)))
        }
    }
}

impl<A> FusedIterator for CharIndices<A> where A: Accessor {}

/** A chunk of a [`Rope`].
 *
 * These are returned from the [`Chunks`] iterator. Each chunk is either a substring
 * stored contiguously in a single leaf of the underlying RRB vector, or a single
 * character whose encoded representation straddles two leaves.
 */
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Chunk<'a> {
    /// A contiguously-stored substring
    Str(&'a str),
    /// A non-contiguously stored character
    Char(char),
}
/** An iterator over chunks of a [`Rope`].
 *
 * Call [`chunks`](Rope::chunks) to construct one.
 */
pub struct Chunks<'a> {
    /* Invariants:
    1. We are iterating over a Vector containing already-validated UTF-8.
    2. Anything already read from `self.inner` but not yet returned to the caller
       is stored in `self.unconsumed_fwd` or `self.unconsumed_back`.
    3. `self.unconsumed_fwd` always begins on a character boundary, but does not
       necessarily end on one.
    4. `self.unconsumed_back` always ends on a character boundary, but does not
       necessarily begin on one.
    */
    inner: vector::Chunks<'a, u8>,
    unconsumed_fwd: &'a [u8],
    unconsumed_back: &'a [u8],
}

impl<'a> Iterator for Chunks<'a> {
    type Item = Chunk<'a>;

    #[allow(clippy::redundant_else)]
    fn next(&mut self) -> Option<Self::Item> {
        while self.unconsumed_fwd.is_empty() {
            if let Some(unconsumed) = self.inner.next() {
                self.unconsumed_fwd = unconsumed;
            } else {
                if self.unconsumed_back.is_empty() {
                    return None;
                }
                self.unconsumed_fwd = self.unconsumed_back;
                self.unconsumed_back = &[];
            }
        }

        let mut unconsumed = self.unconsumed_fwd;

        let start_of_last_char_option =
            (0..unconsumed.len()).rfind(|&i| utf8_is_first_byte(unconsumed[i]));

        let start_of_last_char = unsafe {
            // SAFETY: Per the invaraiant, unconsumed always begins on a
            // character boundary, and just above we ensured that unconsumed is
            // non-empty. Therefore, it must always contain a first byte, so the
            // search must always succeed.
            start_of_last_char_option.unwrap_unchecked()
        };

        let last_char_width = utf8_char_width(unconsumed[start_of_last_char]);

        if start_of_last_char + last_char_width == unconsumed.len() {
            // `unconsumed` ends on a character boundary, so we can return the
            // entire thing.
            let ret = unsafe {
                // SAFETY: Per the invariant, we have valid UTF-8 that begins on
                // a character boundary, and we've just ensured that it ends on
                // one as well.
                str_from_utf8_debug(unconsumed)
            };
            self.unconsumed_fwd = &[];
            return Some(Chunk::Str(ret));
        } else if start_of_last_char > 0 {
            // The last character is incomplete, but we have some complete ones
            // before it, so return those and leave the last character
            // unconsumed.
            let ret = unsafe {
                // SAFETY: Per the invariant, we have valid UTF-8 that begins on
                // a character boundary, and we're slicing it so that it ends on
                // one as well.
                str_from_utf8_debug(&unconsumed[..start_of_last_char])
            };
            self.unconsumed_fwd = &unconsumed[start_of_last_char..];
            return Some(Chunk::Str(ret));
        } else {
            // We don't have any complete character, so we need to keep reading
            // until we do.
            let mut bytes_available = unconsumed.len();
            // 4 is the maximum UTF-8 encoded character length
            let mut buf: [u8; 4] = Default::default();

            buf[..bytes_available].copy_from_slice(unconsumed);
            unconsumed = &[];

            while bytes_available < last_char_width {
                while unconsumed.is_empty() {
                    unconsumed = self.inner.next().unwrap_or_else(|| {
                        let ret = self.unconsumed_back;
                        self.unconsumed_back = &[];
                        ret
                    });
                }
                buf[bytes_available] = unconsumed[0];
                unconsumed = &unconsumed[1..];
                bytes_available += 1;
            }

            // We now have a complete charater in `buf`. Save the unconsumed
            // output and then decode and return that character.
            self.unconsumed_fwd = unconsumed;

            // SAFETY: `buf` now contains exactly one valid UTF-8 character.
            unsafe {
                let c = next_code_point(&mut buf.iter().copied()).unwrap_unchecked();
                return Some(Chunk::Char(char_from_u32_debug(c)));
            }
        }
    }
}

impl<'a> DoubleEndedIterator for Chunks<'a> {
    #[allow(clippy::redundant_else)]
    fn next_back(&mut self) -> Option<Self::Item> {
        while self.unconsumed_back.is_empty() {
            if let Some(unconsumed) = self.inner.next_back() {
                self.unconsumed_back = unconsumed;
            } else {
                if self.unconsumed_fwd.is_empty() {
                    return None;
                }
                self.unconsumed_back = self.unconsumed_fwd;
                self.unconsumed_fwd = &[];
            }
        }

        let mut unconsumed = self.unconsumed_back;

        let start_of_first_char = (0..unconsumed.len())
            .find(|&i| utf8_is_first_byte(unconsumed[i]))
            .unwrap_or(unconsumed.len());

        if start_of_first_char == 0 {
            // `unconsumed` starts on a character boundary, so we can return the
            // entire thing.
            let ret = unsafe {
                // SAFETY: Per the invariant, we have valid UTF-8 that ends on
                // a character boundary, and we've just ensured that it begins on
                // one as well.
                str_from_utf8_debug(unconsumed)
            };
            self.unconsumed_back = &[];
            return Some(Chunk::Str(ret));
        } else if start_of_first_char < unconsumed.len() {
            // The buffer begins with an incomplete character, but we have some
            // complete ones after it, so return those and leave the beginning
            // unconsumed.

            let ret = unsafe {
                // SAFETY: Per the invariant, we have valid UTF-8 that begins on
                // a character boundary, and we're slicing it so that it ends on
                // one as well.
                str_from_utf8_debug(&unconsumed[start_of_first_char..])
            };
            self.unconsumed_back = &unconsumed[..start_of_first_char];
            return Some(Chunk::Str(ret));
        } else {
            // We don't have any complete character, so we need to keep reading
            // until we do.
            let mut bytes_available = unconsumed.len();
            // 4 is the maximum UTF-8 encoded character length
            let mut buf: [u8; 4] = Default::default();

            buf[4 - bytes_available..].copy_from_slice(unconsumed);
            unconsumed = &[];

            while !utf8_is_first_byte(buf[4 - bytes_available]) {
                while unconsumed.is_empty() {
                    unconsumed = self.inner.next_back().unwrap_or_else(|| {
                        let ret = self.unconsumed_fwd;
                        self.unconsumed_fwd = &[];
                        ret
                    });
                }

                buf[3 - bytes_available] = unconsumed[unconsumed.len() - 1];
                unconsumed = &unconsumed[..unconsumed.len() - 1];
                bytes_available += 1;
            }

            // We now have a complete charater in `buf`. Save the unconsumed
            // output and then decode and return that character.
            self.unconsumed_back = unconsumed;
            // SAFETY: `buf` now contains exactly one valid UTF-8 character.
            unsafe {
                let c = next_code_point_reverse(&mut buf.iter().copied()).unwrap_unchecked();
                return Some(Chunk::Char(char_from_u32_debug(c)));
            }
        }
    }
}

impl<'a> FusedIterator for Chunks<'a> {}
///An iterator returned by [`find_all`](Rope::rfind_all).
pub struct FindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
    inner: P::FindAllImpl<A>,
}

impl<A, P> FindAll<A, P>
where
    A: Accessor,
    P: Pattern,
{
    fn new(inner: P::FindAllImpl<A>) -> FindAll<A, P> {
        FindAll { inner }
    }
}

impl<A, P> ToOwning for FindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
    type Owning = FindAll<OwningAccessor, P::Owned>;

    fn to_owning(&self) -> Self::Owning {
        FindAll {
            inner: P::_convert_to_owning(&self.inner),
        }
    }
}

impl<A, P> IntoOwning for FindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        FindAll {
            inner: P::_convert_into_owning(self.inner),
        }
    }
}

impl<A, P> Iterator for FindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
    type Item = (Range<usize>, P::Output);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<A, P> FusedIterator for FindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
}

impl<A, P> DoubleEndedIterator for FindAll<A, P>
where
    P: Pattern,
    A: Accessor,
    P::FindAllImpl<A>: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

///An iterator returned by [`rfind_all`](Rope::rfind_all).
pub struct RFindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
    inner: P::RFindAllImpl<A>,
}

impl<A, P> RFindAll<A, P>
where
    A: Accessor,
    P: Pattern,
{
    fn new(inner: P::RFindAllImpl<A>) -> RFindAll<A, P> {
        RFindAll { inner }
    }
}

impl<A, P> ToOwning for RFindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
    type Owning = RFindAll<OwningAccessor, P::Owned>;

    fn to_owning(&self) -> Self::Owning {
        RFindAll {
            inner: P::_rconvert_to_owning(&self.inner),
        }
    }
}

impl<A, P> IntoOwning for RFindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        RFindAll {
            inner: P::_rconvert_into_owning(self.inner),
        }
    }
}

impl<A, P> Iterator for RFindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
    type Item = (Range<usize>, P::Output);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<A, P> DoubleEndedIterator for RFindAll<A, P>
where
    P: Pattern,
    A: Accessor,
    P::RFindAllImpl<A>: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<A, P> FusedIterator for RFindAll<A, P>
where
    P: Pattern,
    A: Accessor,
{
}

struct SplitImpl<A, M, const LIMITED: bool, const TERMINATED: bool, const INCLUSIVE: bool> {
    haystack: A,
    matcher: M,
    limit: usize,
    empty_at_back: bool,
}

impl<A, P, const LIMITED: bool, const TERMINATED: bool, const INCLUSIVE: bool>
    SplitImpl<A, FindAll<A, P>, LIMITED, TERMINATED, INCLUSIVE>
where
    A: Accessor,
    P: Pattern,
{
    fn new_forward(
        haystack: A,
        needle: P,
        limit: usize,
    ) -> SplitImpl<A, FindAll<A, P>, LIMITED, TERMINATED, INCLUSIVE> {
        SplitImpl {
            matcher: FindAll::new(needle._find_all(haystack.shallow_clone())),
            haystack,
            limit,
            empty_at_back: !TERMINATED,
        }
    }
}

impl<A, P, const LIMITED: bool, const TERMINATED: bool, const INCLUSIVE: bool>
    SplitImpl<A, RFindAll<A, P>, LIMITED, TERMINATED, INCLUSIVE>
where
    A: Accessor,
    P: Pattern,
{
    fn new_backward(
        haystack: A,
        needle: P,
        limit: usize,
    ) -> SplitImpl<A, RFindAll<A, P>, LIMITED, TERMINATED, INCLUSIVE> {
        SplitImpl {
            matcher: RFindAll::new(needle._rfind_all(haystack.shallow_clone())),
            haystack,
            limit,
            empty_at_back: !TERMINATED,
        }
    }
}

impl<A, M, const LIMITED: bool, const TERMINATED: bool, const INCLUSIVE: bool>
    SplitImpl<A, M, LIMITED, TERMINATED, INCLUSIVE>
where
    A: Accessor,
{
    // SAFETY: ranges returned by `next_match` must have both ends on
    // UTF-8 character boundaries.
    unsafe fn forward<F: FnMut(&mut Self) -> Option<Range<usize>>>(
        &mut self,
        next_match: &mut F,
    ) -> Option<Rope> {
        if LIMITED && self.limit == 0 {
            return None;
        }
        match next_match(self) {
            Some(range) if !LIMITED || self.limit > 1 => {
                let mut v = self.haystack.take_front(range.end);

                if !INCLUSIVE {
                    v.truncate(v.len() - (range.end - range.start));
                }

                if LIMITED {
                    self.limit -= 1;
                }
                Some(Rope::from_vector_unchecked(v))
            }
            _ => {
                let ret = Rope::from_vector_unchecked(
                    self.haystack.take_back(self.haystack.front_index()),
                );

                let empty_at_back = self.empty_at_back;
                self.empty_at_back = false;

                if empty_at_back || !ret.is_empty() {
                    if LIMITED {
                        self.limit -= 1;
                    }
                    Some(ret)
                } else {
                    None
                }
            }
        }
    }

    // SAFETY: ranges returned by `next_match` must have both ends on
    // UTF-8 character boundaries.
    unsafe fn backward<F: FnMut(&mut Self) -> Option<Range<usize>>>(
        &mut self,
        next_match: &mut F,
    ) -> Option<Rope> {
        if LIMITED && self.limit == 0 {
            return None;
        }

        match next_match(self) {
            Some(range) if !LIMITED || self.limit > 1 => {
                let rope = Rope::from_vector_unchecked(self.haystack.take_back(range.end));

                if !INCLUSIVE {
                    self.haystack.take_back(range.start);
                }

                let empty_at_back = self.empty_at_back;
                self.empty_at_back = true;

                if rope.is_empty() && !empty_at_back {
                    self.backward(next_match)
                } else {
                    if LIMITED {
                        self.limit -= 1;
                    }
                    Some(rope)
                }
            }
            _ => {
                let rope = Rope::from_vector_unchecked(
                    self.haystack.take_back(self.haystack.front_index()),
                );

                if rope.is_empty() && !self.empty_at_back {
                    None
                } else {
                    self.empty_at_back = false;
                    if LIMITED {
                        self.limit -= 1;
                    }
                    Some(rope)
                }
            }
        }
    }
}

impl<A, M, const LIMITED: bool, const TERMINATED: bool, const INCLUSIVE: bool> ToOwning
    for SplitImpl<A, M, LIMITED, TERMINATED, INCLUSIVE>
where
    A: Accessor,
    M: ToOwning,
{
    type Owning = SplitImpl<A::Owning, M::Owning, LIMITED, TERMINATED, INCLUSIVE>;

    fn to_owning(&self) -> Self::Owning {
        SplitImpl {
            haystack: self.haystack.to_owning(),
            matcher: self.matcher.to_owning(),
            limit: self.limit.to_owning(),
            empty_at_back: self.empty_at_back.to_owning(),
        }
    }
}

impl<A, M, const LIMITED: bool, const TERMINATED: bool, const INCLUSIVE: bool> IntoOwning
    for SplitImpl<A, M, LIMITED, TERMINATED, INCLUSIVE>
where
    A: Accessor,
    M: IntoOwning,
{
    fn into_owning(self) -> Self::Owning {
        SplitImpl {
            haystack: self.haystack.into_owning(),
            matcher: self.matcher.into_owning(),
            limit: self.limit.into_owning(),
            empty_at_back: self.empty_at_back.into_owning(),
        }
    }
}

impl<A, P, const LIMITED: bool, const TERMINATED: bool, const INCLUSIVE: bool> Iterator
    for SplitImpl<A, FindAll<A, P>, LIMITED, TERMINATED, INCLUSIVE>
where
    A: Accessor,
    P: Pattern,
{
    type Item = Rope;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next_match = |s: &mut Self| {
            let (range, _) = s.matcher.next()?;
            Some(range)
        };

        // SAFETY: next_match must yield ranges whose ends fall on UTF-8
        // character boundaries. These ranges come from the pattern's
        // `_find_all` method. All `Pattern` implementations in this crate assure
        // that this holds, and since `Pattern` is a sealed trait, the user
        // cannot supply other implementations.
        unsafe { self.forward(&mut next_match) }
    }
}

impl<A, P, const TERMINATED: bool, const INCLUSIVE: bool> DoubleEndedIterator
    for SplitImpl<A, FindAll<A, P>, false, TERMINATED, INCLUSIVE>
where
    A: Accessor,
    P: Pattern,
    P::FindAllImpl<A>: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut next_match = |s: &mut Self| {
            let (range, _) = s.matcher.next_back()?;
            Some(range)
        };

        // SAFETY: next_match must yield ranges whose ends fall on UTF-8
        // character boundaries. These ranges come from the pattern's
        // `_find_all` method. All `Pattern` implementations in this crate assure
        // that this holds, and since `Pattern` is a sealed trait, the user
        // cannot supply other implementations.
        unsafe { self.backward(&mut next_match) }
    }
}

impl<A, P, const LIMITED: bool, const TERMINATED: bool, const INCLUSIVE: bool> Iterator
    for SplitImpl<A, RFindAll<A, P>, LIMITED, TERMINATED, INCLUSIVE>
where
    A: Accessor,
    P: Pattern,
{
    type Item = Rope;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next_match = |s: &mut Self| {
            let (range, _) = s.matcher.next()?;
            Some(range)
        };

        // SAFETY: next_match must yield ranges whose ends fall on UTF-8
        // character boundaries. These ranges come from the pattern's
        // `_rfind_all` method. All `Pattern` implementations in this crate assure
        // that this holds, and since `Pattern` is a sealed trait, the user
        // cannot supply other implementations.
        unsafe { self.backward(&mut next_match) }
    }
}

impl<A, P, const TERMINATED: bool, const INCLUSIVE: bool> DoubleEndedIterator
    for SplitImpl<A, RFindAll<A, P>, false, TERMINATED, INCLUSIVE>
where
    A: Accessor,
    P: Pattern,
    P::RFindAllImpl<A>: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut next_match = |s: &mut Self| {
            let (range, _) = s.matcher.next_back()?;
            Some(range)
        };

        // SAFETY: next_match must yield ranges whose ends fall on UTF-8
        // character boundaries. These ranges come from the pattern's
        // `_rfind_all` method. All `Pattern` implementations in this crate assure
        // that this holds, and since `Pattern` is a sealed trait, the user
        // cannot supply other implementations.
        unsafe { self.forward(&mut next_match) }
    }
}

macro_rules! def_split {
    ($name:ident, $doc: literal, $limited:expr, $terminated:expr, $inclusive:expr) => {
        #[doc = $doc]
        pub struct $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
            inner: SplitImpl<A, FindAll<A, P>, $limited, $terminated, $inclusive>,
        }

        impl<'h, P> $name<BorrowingAccessor<'h>, P>
        where
            P: Pattern,
        {
            fn new(haystack: &'h Rope, needle: P, limit: usize) -> $name<BorrowingAccessor<'h>, P> {
                let accessor = BorrowingAccessor::new(haystack.as_ref());
                $name {
                    inner: SplitImpl::new_forward(accessor, needle, limit),
                }
            }
        }

        impl<A, P> ToOwning for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
            type Owning = $name<A::Owning, P::Owned>;

            fn to_owning(&self) -> Self::Owning {
                $name {
                    inner: self.inner.to_owning(),
                }
            }
        }

        impl<A, P> IntoOwning for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
            fn into_owning(self) -> Self::Owning {
                $name {
                    inner: self.inner.into_owning(),
                }
            }
        }

        impl<A, P> Iterator for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
            type Item = Rope;

            fn next(&mut self) -> Option<Self::Item> {
                self.inner.next()
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        impl<A, P> FusedIterator for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
        }
    };
}

macro_rules! def_split_double {
    ($name:ident) => {
        impl<A, P> DoubleEndedIterator for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
            <P as Pattern>::FindAllImpl<A>: DoubleEndedIterator,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.inner.next_back()
            }
        }
    };
}

def_split!(
    Split,
    "An iterator returned by [`split`](Rope::split).",
    false,
    false,
    false
);
def_split!(
    SplitN,
    "An iterator returned by [`splitn`](Rope::splitn).",
    true,
    false,
    false
);
def_split!(
    SplitTerminator,
    "An iterator returned by [`split_terminator`](Rope::split_terminator).",
    false,
    true,
    false
);
def_split!(
    SplitInclusive,
    "An iterator returned by [`split_inclusive`](Rope::split_inclusive).",
    false,
    true,
    true
);
def_split_double!(Split);
def_split_double!(SplitTerminator);
def_split_double!(SplitInclusive);

macro_rules! def_rsplit {
    ($name:ident, $doc: literal, $limited:expr, $terminated:expr, $inclusive:expr) => {
        #[doc = $doc]
        pub struct $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
            inner: SplitImpl<A, RFindAll<A, P>, $limited, $terminated, $inclusive>,
        }

        impl<'h, P> $name<BorrowingAccessor<'h>, P>
        where
            P: Pattern,
        {
            fn new(haystack: &'h Rope, needle: P, limit: usize) -> $name<BorrowingAccessor<'h>, P> {
                let accessor = BorrowingAccessor::new(haystack.as_ref());
                $name {
                    inner: SplitImpl::new_backward(accessor, needle, limit),
                }
            }
        }

        impl<A, P> ToOwning for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
            type Owning = $name<A::Owning, P::Owned>;

            fn to_owning(&self) -> Self::Owning {
                $name {
                    inner: self.inner.to_owning(),
                }
            }
        }

        impl<A, P> IntoOwning for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
            fn into_owning(self) -> Self::Owning {
                $name {
                    inner: self.inner.into_owning(),
                }
            }
        }

        impl<A, P> Iterator for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
            type Item = Rope;

            fn next(&mut self) -> Option<Self::Item> {
                self.inner.next()
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        impl<A, P> FusedIterator for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
        {
        }
    };
}

macro_rules! def_rsplit_double {
    ($name:ident) => {
        impl<A, P> DoubleEndedIterator for $name<A, P>
        where
            A: Accessor,
            P: Pattern,
            <P as Pattern>::RFindAllImpl<A>: DoubleEndedIterator,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.inner.next_back()
            }
        }
    };
}

def_rsplit!(
    RSplit,
    "An iterator returned by [`rsplit`](Rope::rsplit).",
    false,
    false,
    false
);
def_rsplit!(
    RSplitN,
    "An iterator returned by [`rsplitn`](Rope::rsplitn).",
    true,
    false,
    false
);
def_rsplit!(
    RSplitTerminator,
    "An iterator returned by [`rsplit_terminator`](Rope::rsplit_terminator).",
    false,
    true,
    false
);

def_rsplit_double!(RSplit);
def_rsplit_double!(RSplitTerminator);

/// An iterator returned by [`lines`](Rope::lines).
pub struct Lines<A>
where
    A: Accessor,
{
    inner: SplitInclusive<A, char>,
}

impl<'h> Lines<BorrowingAccessor<'h>> {
    fn borrowed(haystack: &'h Rope) -> Lines<BorrowingAccessor<'h>> {
        Lines {
            inner: haystack.split_inclusive('\n'),
        }
    }
}

impl<A> Lines<A>
where
    A: Accessor,
{
    fn strip_ending(line: &mut Rope) {
        if line.back() == Some('\n') {
            line.pop_back();
            if line.back() == Some('\r') {
                line.pop_back();
            }
        }
    }
}

impl<A> ToOwning for Lines<A>
where
    A: Accessor,
{
    type Owning = Lines<A::Owning>;

    fn to_owning(&self) -> Self::Owning {
        Lines {
            inner: self.inner.to_owning(),
        }
    }
}

impl<A> IntoOwning for Lines<A>
where
    A: Accessor,
{
    fn into_owning(self) -> Self::Owning {
        Lines {
            inner: self.inner.into_owning(),
        }
    }
}

impl<A> Iterator for Lines<A>
where
    A: Accessor,
{
    type Item = Rope;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = self.inner.next()?;
        Self::strip_ending(&mut line);
        Some(line)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<A> DoubleEndedIterator for Lines<A>
where
    A: Accessor,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut line = self.inner.next_back()?;
        Self::strip_ending(&mut line);
        Some(line)
    }
}

impl<A> FusedIterator for Lines<A> where A: Accessor {}

#[inline]
fn to_range_tuple<R: RangeBounds<usize>>(r: &R, len: usize) -> (usize, usize) {
    (
        match r.start_bound() {
            std::ops::Bound::Included(&i) => i,
            std::ops::Bound::Excluded(&i) => i + 1,
            std::ops::Bound::Unbounded => 0,
        },
        match r.end_bound() {
            std::ops::Bound::Included(&i) => i + 1,
            std::ops::Bound::Excluded(&i) => i,
            std::ops::Bound::Unbounded => len,
        },
    )
}

// Probably this function is unnecessary? I don't think vectors ever really
// return empty chunks, but there's no documented guarantee of this.
fn next_nonempty<'a, I, A>(iter: &mut I) -> Option<&'a [A]>
where
    I: Iterator<Item = &'a [A]>,
{
    loop {
        if let Some(chunk) = iter.next() {
            if chunk.is_empty() {
                continue;
            }
            return Some(chunk);
        }
        return None;
    }
}

// SAFETY: `ch` must be a valid Unicode codepoint.
unsafe fn char_from_u32_debug(ch: u32) -> char {
    if cfg!(debug_assertions) {
        char::from_u32(ch).unwrap()
    } else {
        char::from_u32_unchecked(ch)
    }
}

// SAFETY: `bytes` must be valid UTF-8.
unsafe fn str_from_utf8_debug(bytes: &[u8]) -> &str {
    if cfg!(debug_assertions) {
        std::str::from_utf8(bytes).unwrap()
    } else {
        std::str::from_utf8_unchecked(bytes)
    }
}

// SAFETY: `bytes` must be valid UTF-8.
unsafe fn string_from_utf8_debug(bytes: Vec<u8>) -> String {
    if cfg!(debug_assertions) {
        String::from_utf8(bytes).unwrap()
    } else {
        String::from_utf8_unchecked(bytes)
    }
}
