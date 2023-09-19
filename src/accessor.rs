//! Lending iterators over vectors, with flexible ownership.
//!
//! Accessors are what make it possible for all of [`Rope`](super::Rope)'s
//! various iterators to implement [`IntoOwning`]. They are mostly an
//! implementation detail, but they do show up as generic arguments in type
//! signatures, and so this module is exposed and documented. You cannot
//! directly construct or use an `Accessor` through any of `im-rope`'s public
//! APIs, but you can tell whether an iterator borrows or owns its underlying
//! rope based on whether it has a `BorrowingAccessor<'a>` or an
//! `OwningAccessor` in its type.

use im::vector;
use im::vector::Vector;
use sealed::sealed;
use static_cow::{IntoOwning, ToOwning};
use std::iter::FusedIterator;
use std::mem::MaybeUninit;
use std::ops::Range;

/// Owns a `Vector<u8>` along with a `Focus` which references it.

// SAFETY invariants:
// 1. `vector` is valid.
// 2. `focus` is initialized with something that can be
//    safely transmuted into `Focus<'a, u8>` where `Self: 'a`.

struct OwningFocus {
    vector: *mut Vector<u8>,
    focus: MaybeUninit<vector::Focus<'static, u8>>,
}

impl OwningFocus {
    fn new(vector: Box<Vector<u8>>) -> OwningFocus {
        let vector_ref = Box::leak(vector);
        let vector_ptr = vector_ref as *mut Vector<u8>;
        let focus = MaybeUninit::new(vector_ref.focus());

        // SAFETY:
        // 1. `vector` is valid.
        // 2. `focus` is initialized with a focus borrowed from `vector`,
        //    so it's safe to give it a lifetime bounded by self.
        OwningFocus {
            vector: vector_ptr,
            focus,
        }
    }

    /// Access the underlying vector.
    fn as_vector(&self) -> &Vector<u8> {
        // SAFETY: per the first invariant.
        unsafe { &*self.vector }
    }

    fn as_focus(&mut self) -> &mut vector::Focus<'_, u8> {
        // SAFETY: per the second invariant.
        unsafe { std::mem::transmute(self.focus.assume_init_mut()) }
    }

    fn update<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Vector<u8>) -> R,
    {
        unsafe {
            // SAFETY: the invariants assure that these two calls are safe, but
            // afterward `focus` is uninitialized which breaks the second
            // invariant. We need to make sure it gets reinitialized before we
            // return.
            let call_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.focus.assume_init_drop();
                f(&mut *self.vector)
            }));

            // Here, whether or not there was a panic, `vector` is still in some
            // valid (though perhaps weird) state, while `focus` presumptively
            // needs to be reinitialized.
            let focus_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.focus.write((*self.vector).focus());
            }));

            if focus_result.is_err() {
                // Safety necessitates that `focus` be initialized when we exit
                // from this method, so if we fail at initializing it then the
                // only safe course is to abort.
                std::process::abort();
            };

            // Now the invariant is restored and we can either return or resume
            // panicking.
            match call_result {
                Ok(r) => r,
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
    }

    /// Access the byte at `index`.
    fn index(&mut self, index: usize) -> u8 {
        *self.as_focus().index(index)
    }

    /// Access the chunk at `index`.
    fn chunk_at(&mut self, index: usize) -> (Range<usize>, &[u8]) {
        self.as_focus().chunk_at(index)
    }
}

impl Clone for OwningFocus {
    fn clone(&self) -> Self {
        OwningFocus::new(Box::new(self.as_vector().clone()))
    }
}

impl Drop for OwningFocus {
    fn drop(&mut self) {
        // SAFETY: `vector` is valid per the first invariant, and `focus` is
        // initialized per the second invariant.
        unsafe {
            self.focus.assume_init_drop();
            std::mem::drop(Box::from_raw(self.vector));
        }
    }
}

/// An [`Accessor`] which borrows its underlying `Vector<u8>`.
#[allow(clippy::module_name_repetitions)]
pub struct BorrowingAccessor<'a> {
    vector: &'a Vector<u8>,
    focus: vector::Focus<'a, u8>,
    front_index: usize,
    back_index: usize,
}

impl<'a> BorrowingAccessor<'a> {
    pub(crate) fn new(vector: &'a Vector<u8>) -> BorrowingAccessor<'a> {
        BorrowingAccessor {
            vector,
            focus: vector.focus(),
            front_index: 0,
            back_index: vector.len(),
        }
    }
}

impl<'a> ToOwning for BorrowingAccessor<'a> {
    type Owning = OwningAccessor;

    fn to_owning(&self) -> Self::Owning {
        OwningAccessor::from_borrowed(self.vector, self.front_index, self.back_index)
    }
}

impl<'a> IntoOwning for BorrowingAccessor<'a> {
    fn into_owning(self) -> Self::Owning {
        self.to_owning()
    }
}

/// An [`Accessor`] which owns its underlying `Vector<u8>`.
///
/// An `OwningAccessor` will, at exponential intervals, "garbage collect" the
/// portion of the vector which it has already returned. An `OwningAccessor`'s
/// worst-case space consumption is therefore big-O linear in the unconsumed
/// portion of the vector, rather than linear in the size of the entire vector.
#[derive(Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct OwningAccessor {
    focus: OwningFocus,
    /// The index of the next byte to be returned from the front, relative to
    /// the original vector this accessor was constructed from.
    proper_front_index: usize,
    /// The index+1 of the next byte to be returned from the back, relative to
    /// the *original* vector this accessor was constructed from.
    proper_back_index: usize,
    /// The index, relative to the original vector this accessor was constructed
    /// from, of the first byte still held by `focus`.
    focal_front_index: usize,
    /// The index+1, relative to the original vector this accessor was constructed
    /// from, of the last byte still held by `focus`.
    focal_back_index: usize,
}

impl OwningAccessor {
    pub(crate) fn new(vector: Vector<u8>) -> OwningAccessor {
        let len = vector.len();
        OwningAccessor {
            focus: OwningFocus::new(Box::new(vector)),
            proper_front_index: 0,
            proper_back_index: len,
            focal_front_index: 0,
            focal_back_index: len,
        }
    }

    fn from_borrowed(vector: &Vector<u8>, front_index: usize, back_index: usize) -> OwningAccessor {
        let mut owning = OwningAccessor {
            focus: OwningFocus::new(Box::new(vector.clone())),
            proper_front_index: front_index,
            proper_back_index: back_index,
            focal_front_index: 0,
            focal_back_index: vector.len(),
        };

        owning.gc();
        owning
    }

    fn gc(&mut self) {
        let cur_len = self.proper_back_index - self.proper_front_index;
        let orig_len = self.focal_back_index - self.focal_front_index;

        if orig_len > 256 && orig_len / 2 >= cur_len {
            self.focus.update(|vector| {
                *vector = vector.split_off(self.proper_front_index - self.focal_front_index);
                vector.truncate(self.proper_back_index - self.proper_front_index);
            });
            self.focal_front_index = self.proper_front_index;
            self.focal_back_index = self.proper_back_index;
        }
    }
}

/// An iterator over the bytes of an `Accessor`.
pub struct ByteIter<'a, A>(&'a mut A);

/// A double-ended lending iterator over a `Vector<u8>`.
#[sealed]
pub trait Accessor: IntoOwning<Owning = OwningAccessor> {
    /// Consumes the frontmost byte and returns it along with its index.
    fn front_byte(&mut self) -> Option<(usize, u8)>;
    /// Consumes the backmost byte and returns it along with its index.
    fn back_byte(&mut self) -> Option<(usize, u8)>;
    /// Consumes the frontmost chunk and returns a reference to it along with
    /// the range it covers.
    fn front_chunk(&mut self) -> Option<(Range<usize>, &[u8])>;
    /// Consumes the backmost chunk and returns a refernence to it along with
    /// the range it covers.
    fn back_chunk(&mut self) -> Option<(Range<usize>, &[u8])>;

    /// Consumes everything from the first unconsumed byte up to `index` and returns
    /// it as a new vector.
    fn take_front(&mut self, index: usize) -> Vector<u8>;
    /// Consumes everything from `index` to the backmost unconsumed byte and returns
    /// it as a new vector.
    fn take_back(&mut self, index: usize) -> Vector<u8>;

    /// Returns the index of the frontmost unconsumed byte.
    #[must_use]
    fn front_index(&self) -> usize;
    /// Returns the index+1 of the backmost unconsumed byte.
    #[must_use]
    fn back_index(&self) -> usize;

    /// Returns a consuming iterator over the accessor's bytes.
    fn byte_iter(&mut self) -> ByteIter<'_, Self> {
        ByteIter(self)
    }

    /// Clones the accessor.
    ///
    /// This is really just `clone`. It's "shallow" in the sense that it just
    /// won't turn a borrowing accessor into an owning one, and just copies the
    /// reference. Due to a limitation of [`static_cow`] and Rust's coherence
    /// rules, it isn't possible to have both a `Clone` implementation and a
    /// non-trivial `IntoOwning` implementation.
    #[must_use]
    fn shallow_clone(&self) -> Self;
}

impl<'a, A> Iterator for ByteIter<'a, A>
where
    A: Accessor,
{
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.front_byte()?.1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.back_index() - self.0.front_index();
        (len, Some(len))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, A> DoubleEndedIterator for ByteIter<'a, A>
where
    A: Accessor,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.0.back_byte()?.1)
    }
}

impl<'a, A> FusedIterator for ByteIter<'a, A> where A: Accessor {}

impl<'a, A> ExactSizeIterator for ByteIter<'a, A> where A: Accessor {}

#[sealed]
impl<'a> Accessor for BorrowingAccessor<'a> {
    fn front_byte(&mut self) -> Option<(usize, u8)> {
        if self.front_index == self.back_index {
            None
        } else {
            let byte = *self.focus.index(self.front_index);
            let index = self.front_index;
            self.front_index += 1;
            Some((index, byte))
        }
    }

    fn back_byte(&mut self) -> Option<(usize, u8)> {
        if self.front_index == self.back_index {
            None
        } else {
            self.back_index -= 1;
            Some((self.back_index, *self.focus.index(self.back_index)))
        }
    }

    fn front_chunk(&mut self) -> Option<(Range<usize>, &[u8])> {
        if self.front_index == self.back_index {
            None
        } else {
            let (unclamped_range, unclamped_chunk) = self.focus.chunk_at(self.front_index);
            let clamped_range = std::cmp::max(unclamped_range.start, self.front_index)
                ..std::cmp::min(unclamped_range.end, self.back_index);
            let cut_range = (clamped_range.start - unclamped_range.start)
                ..(unclamped_chunk.len() - (unclamped_range.end - clamped_range.end));
            self.front_index = clamped_range.end;
            Some((clamped_range, &unclamped_chunk[cut_range]))
        }
    }

    fn back_chunk(&mut self) -> Option<(Range<usize>, &[u8])> {
        if self.front_index == self.back_index {
            None
        } else {
            let (unclamped_range, unclamped_chunk) = self.focus.chunk_at(self.back_index - 1);
            let clamped_range = std::cmp::max(unclamped_range.start, self.front_index)
                ..std::cmp::min(unclamped_range.end, self.back_index);
            let cut_range = (clamped_range.start - unclamped_range.start)
                ..(unclamped_chunk.len() - (unclamped_range.end - clamped_range.end));
            self.back_index = clamped_range.start;
            Some((clamped_range, &unclamped_chunk[cut_range]))
        }
    }

    fn take_front(&mut self, index: usize) -> Vector<u8> {
        let mut vector = self.vector.skip(self.front_index);
        vector.truncate(index - self.front_index);
        self.front_index = index;
        vector
    }

    fn take_back(&mut self, index: usize) -> Vector<u8> {
        let mut vector = self.vector.skip(index);
        vector.truncate(self.back_index - index);
        self.back_index = index;
        vector
    }

    fn front_index(&self) -> usize {
        self.front_index
    }

    fn back_index(&self) -> usize {
        self.back_index
    }

    fn shallow_clone(&self) -> Self {
        BorrowingAccessor {
            vector: self.vector,
            focus: self.focus.clone(),
            front_index: self.front_index,
            back_index: self.back_index,
        }
    }
}

#[sealed]
impl Accessor for OwningAccessor {
    fn front_byte(&mut self) -> Option<(usize, u8)> {
        if self.proper_front_index == self.proper_back_index {
            None
        } else {
            let byte = self
                .focus
                .index(self.proper_front_index - self.focal_front_index);
            let index = self.proper_front_index;
            self.proper_front_index += 1;
            self.gc();
            Some((index, byte))
        }
    }

    fn back_byte(&mut self) -> Option<(usize, u8)> {
        if self.proper_front_index == self.proper_back_index {
            None
        } else {
            self.proper_back_index -= 1;
            let byte = self
                .focus
                .index(self.proper_back_index - self.focal_front_index);
            self.gc();
            Some((self.proper_back_index, byte))
        }
    }

    fn front_chunk(&mut self) -> Option<(Range<usize>, &[u8])> {
        if self.proper_front_index == self.proper_back_index {
            None
        } else {
            self.gc();
            let (focal_range, chunk) = self
                .focus
                .chunk_at(self.proper_front_index - self.focal_front_index);
            let unclamped_proper_range = focal_range.start + self.focal_front_index
                ..focal_range.end + self.focal_front_index;
            let clamped_proper_range =
                std::cmp::max(unclamped_proper_range.start, self.proper_front_index)
                    ..std::cmp::min(unclamped_proper_range.end, self.proper_back_index);
            let cut_range = (clamped_proper_range.start - unclamped_proper_range.start)
                ..(focal_range.len() - (unclamped_proper_range.end - clamped_proper_range.end));

            self.proper_front_index = clamped_proper_range.end;
            Some((clamped_proper_range, &chunk[cut_range]))
        }
    }

    fn back_chunk(&mut self) -> Option<(Range<usize>, &[u8])> {
        if self.proper_front_index == self.proper_back_index {
            None
        } else {
            self.gc();
            let (focal_range, chunk) = self
                .focus
                .chunk_at(self.proper_back_index - self.focal_front_index - 1);
            let unclamped_proper_range = focal_range.start + self.focal_front_index
                ..focal_range.end + self.focal_front_index;
            let clamped_proper_range =
                std::cmp::max(unclamped_proper_range.start, self.proper_front_index)
                    ..std::cmp::min(unclamped_proper_range.end, self.proper_back_index);
            let cut_range = (clamped_proper_range.start - unclamped_proper_range.start)
                ..(focal_range.len() - (unclamped_proper_range.end - clamped_proper_range.end));

            self.proper_back_index = clamped_proper_range.start;
            Some((clamped_proper_range, &chunk[cut_range]))
        }
    }

    fn take_front(&mut self, index: usize) -> Vector<u8> {
        let mut vector = self
            .focus
            .as_vector()
            .skip(self.proper_front_index - self.focal_front_index);
        vector.truncate(index - self.proper_front_index);
        self.proper_front_index = index;
        self.gc();
        vector
    }

    fn take_back(&mut self, index: usize) -> Vector<u8> {
        let mut vector = self.focus.as_vector().skip(index - self.focal_front_index);
        vector.truncate(self.proper_back_index - index);
        self.proper_back_index = index;
        self.gc();
        vector
    }

    fn front_index(&self) -> usize {
        self.proper_front_index
    }

    fn back_index(&self) -> usize {
        self.proper_back_index
    }

    fn shallow_clone(&self) -> Self {
        self.clone()
    }
}

pub(crate) struct PopVecBytes<'a>(pub(crate) &'a mut Vector<u8>);

impl<'a> Iterator for PopVecBytes<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len(), Some(self.0.len()))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a> DoubleEndedIterator for PopVecBytes<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.pop_back()
    }
}

impl<'a> FusedIterator for PopVecBytes<'a> {}
impl<'a> ExactSizeIterator for PopVecBytes<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::StreamStrategy;
    use proptest::prelude::*;
    use proptest_derive::Arbitrary;

    #[derive(Debug, Copy, Clone, Arbitrary)]
    enum Instruction {
        FrontByte,
        BackByte,
        FrontChunk,
        BackChunk,
        TakeFront,
        TakeBack,
    }

    proptest! {
        #[test]
        fn accessors_agree(
            vec in prop::collection::vec(u8::arbitrary(), 0..1024),
            ratio in 0.0f64 .. 1.0,
            mut stream in StreamStrategy(Instruction::arbitrary()))
        {
            let start_size = vec.len();
            #[allow(clippy::cast_possible_truncation,clippy::cast_precision_loss,clippy::cast_sign_loss)]
            let end_size = (ratio * (start_size as f64)).round() as usize;
            assert!(end_size <= start_size);

            let v = Vector::from(vec);

            let mut borrowing = BorrowingAccessor::new(&v);
            while borrowing.back_index() - borrowing.front_index() > end_size {
                match stream.gen() {
                    Instruction::FrontByte => {
                        borrowing.front_byte();
                    }
                    Instruction::BackByte => {
                        borrowing.back_byte();
                    }
                    Instruction::FrontChunk => {
                        borrowing.front_chunk();
                    }
                    Instruction::BackChunk => {
                        borrowing.back_chunk();
                    }
                    Instruction::TakeFront => {
                        let i = std::cmp::min(borrowing.front_index() + 32, borrowing.back_index());
                        borrowing.take_front(i);
                    }
                    Instruction::TakeBack => {
                        let i = std::cmp::max(
                            borrowing.back_index().saturating_sub(32),
                            borrowing.front_index());

                        borrowing.take_back(i);
                    }
                }
            }

            let mut owning = borrowing.to_owning();

            while borrowing.front_index() != borrowing.back_index() {
                prop_assert_eq!(borrowing.front_index(), owning.front_index());
                prop_assert_eq!(borrowing.back_index(), owning.back_index());

                match stream.gen() {
                    Instruction::FrontByte => {
                        prop_assert_eq!(borrowing.front_byte(), owning.front_byte());
                    }
                    Instruction::BackByte => {
                        prop_assert_eq!(borrowing.back_byte(), owning.back_byte());
                    }
                    Instruction::FrontChunk => {
                        prop_assert_eq!(borrowing.front_chunk(), owning.front_chunk());
                    }
                    Instruction::BackChunk => {
                        prop_assert_eq!(borrowing.back_chunk(), owning.back_chunk());
                    }
                    Instruction::TakeFront => {
                        let i = std::cmp::min(borrowing.front_index() + 32, borrowing.back_index());
                        prop_assert_eq!(borrowing.take_front(i), owning.take_front(i));
                    }
                    Instruction::TakeBack => {
                        let i = std::cmp::max(
                            borrowing.back_index().saturating_sub(32),
                            borrowing.front_index(),
                        );
                        prop_assert_eq!(borrowing.take_back(i), owning.take_back(i));
                    }
                }
            }
        }
    }
}
