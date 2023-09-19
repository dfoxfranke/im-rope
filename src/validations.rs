//! Operations related to UTF-8 validation.
//!
//! Lightly modified from the standard library's implementation
//! in order to operate on [`im::Vector`]s.

use im::vector::Vector;
use std::mem;

/// Errors which can occur when attempting to interpret a sequence of [`u8`]
/// as a string.
///
/// This structure exactly duplicates [`std::str::Utf8Error`] to work around
/// the fact that its constructors are private.
#[derive(Copy, Eq, PartialEq, Clone, Debug)]
pub struct Utf8Error {
    valid_up_to: usize,
    error_len: Option<u8>,
}

impl Utf8Error {
    /// Returns the index in the given string up to which valid UTF-8 was
    /// verified.
    #[must_use]
    #[inline]
    pub const fn valid_up_to(&self) -> usize {
        self.valid_up_to
    }

    /// Provides more information about the failure:
    ///
    /// * `None`: the end of the input was reached unexpectedly.
    ///   `self.valid_up_to()` is 1 to 3 bytes from the end of the input.
    ///   If a byte stream (such as a file or a network socket) is being decoded incrementally,
    ///   this could be a valid `char` whose UTF-8 byte sequence is spanning multiple chunks.
    ///
    /// * `Some(len)`: an unexpected byte was encountered.
    ///   The length provided is that of the invalid byte sequence
    ///   that starts at the index given by `valid_up_to()`.
    #[must_use]
    #[inline]
    pub const fn error_len(&self) -> Option<usize> {
        // FIXME: This should become `map` again, once it's `const`
        match self.error_len {
            Some(len) => Some(len as usize),
            None => None,
        }
    }
}

impl std::fmt::Display for Utf8Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(error_len) = self.error_len {
            write!(
                f,
                "invalid utf-8 sequence of {} bytes from index {}",
                error_len, self.valid_up_to
            )
        } else {
            write!(
                f,
                "incomplete utf-8 byte sequence from index {}",
                self.valid_up_to
            )
        }
    }
}

impl std::error::Error for Utf8Error {}

impl From<std::str::Utf8Error> for Utf8Error {
    fn from(e: std::str::Utf8Error) -> Self {
        Utf8Error {
            valid_up_to: e.valid_up_to(),
            error_len: e.error_len().map(|l| l.try_into().unwrap()),
        }
    }
}

/// Checks whether the byte is a UTF-8 initial byte
#[allow(clippy::cast_possible_wrap)]
pub(super) const fn utf8_is_first_byte(byte: u8) -> bool {
    byte as i8 >= -0x40
}

/// Returns the initial codepoint accumulator for the first byte.
/// The first byte is special, only want bottom 5 bits for width 2, 4 bits
/// for width 3, and 3 bits for width 4.
#[inline]
const fn utf8_first_byte(byte: u8, width: u32) -> u32 {
    (byte & (0x7F >> width)) as u32
}

/// Returns the value of `ch` updated with continuation byte `byte`.
#[inline]
const fn utf8_acc_cont_byte(ch: u32, byte: u8) -> u32 {
    (ch << 6) | (byte & CONT_MASK) as u32
}

/// Checks whether the byte is a UTF-8 continuation byte (i.e., starts with the
/// bits `10`).
#[inline]
#[allow(clippy::cast_possible_wrap)]
pub(super) const fn utf8_is_cont_byte(byte: u8) -> bool {
    (byte as i8) < -0x40
}

#[inline]
pub(super) fn starts_on_utf8_boundary(v: &Vector<u8>) -> bool {
    v.front().map_or(true, |&b| utf8_is_first_byte(b))
}

#[inline]
pub(super) fn ends_on_utf8_boundary(v: &Vector<u8>) -> bool {
    if v.back().map_or(true, |&b| b < 128) {
        return true;
    }

    let mut w = v.clone();
    w.pop_back(); // We've already ruled out the ASCII case

    for expected_length in 2usize..=4 {
        let Some(ch) = w.pop_back() else { return false };
        
        if utf8_is_first_byte(ch) {
            return utf8_char_width(ch) == expected_length;
        }
    }

    false
}

/// Reads the next code point out of a byte iterator (assuming a
/// UTF-8-like encoding).
///
/// # Safety
///
/// `bytes` must produce a valid UTF-8-like (UTF-8 or WTF-8) string
#[inline]
pub(super) unsafe fn next_code_point<I: Iterator<Item = u8>>(bytes: &mut I) -> Option<u32> {
    // Decode UTF-8
    let x = bytes.next()?;
    if x < 128 {
        return Some(x.into());
    }

    // Multibyte case follows
    // Decode from a byte combination out of: [[[x y] z] w]
    // NOTE: Performance is sensitive to the exact formulation here
    let init = utf8_first_byte(x, 2);
    // SAFETY: `bytes` produces an UTF-8-like string,
    // so the iterator must produce a value here.
    let y = unsafe { unwrap_debug(bytes.next()) };
    let mut ch = utf8_acc_cont_byte(init, y);
    if x >= 0xE0 {
        // [[x y z] w] case
        // 5th bit in 0xE0 .. 0xEF is always clear, so `init` is still valid
        // SAFETY: `bytes` produces an UTF-8-like string,
        // so the iterator must produce a value here.
        let z = unsafe { unwrap_debug(bytes.next()) };
        let y_z = utf8_acc_cont_byte((y & CONT_MASK).into(), z);
        ch = init << 12 | y_z;
        if x >= 0xF0 {
            // [x y z w] case
            // use only the lower 3 bits of `init`
            // SAFETY: `bytes` produces an UTF-8-like string,
            // so the iterator must produce a value here.
            let w = unsafe { unwrap_debug(bytes.next()) };
            ch = (init & 7) << 18 | utf8_acc_cont_byte(y_z, w);
        }
    }

    Some(ch)
}

/// Reads the last code point out of a byte iterator (assuming a
/// UTF-8-like encoding).
///
/// # Safety
///
/// `bytes` must produce a valid UTF-8-like (UTF-8 or WTF-8) string
#[inline]
pub(super) unsafe fn next_code_point_reverse<I>(bytes: &mut I) -> Option<u32>
where
    I: DoubleEndedIterator<Item = u8>,
{
    // Decode UTF-8
    let w = match bytes.next_back()? {
        next_byte if next_byte < 128 => return Some(next_byte.into()),
        back_byte => back_byte,
    };

    // Multibyte case follows
    // Decode from a byte combination out of: [x [y [z w]]]

    // SAFETY: `bytes` produces an UTF-8-like string,
    // so the iterator must produce a value here.
    let z = unsafe { unwrap_debug(bytes.next_back()) };
    let mut ch = utf8_first_byte(z, 2);
    if utf8_is_cont_byte(z) {
        // SAFETY: `bytes` produces an UTF-8-like string,
        // so the iterator must produce a value here.
        let y = unsafe { unwrap_debug(bytes.next_back()) };
        ch = utf8_first_byte(y, 3);
        if utf8_is_cont_byte(y) {
            // SAFETY: `bytes` produces an UTF-8-like string,
            // so the iterator must produce a value here.
            let x = unsafe { unwrap_debug(bytes.next_back()) };
            ch = utf8_first_byte(x, 4);
            ch = utf8_acc_cont_byte(ch, y);
        }
        ch = utf8_acc_cont_byte(ch, z);
    }
    ch = utf8_acc_cont_byte(ch, w);

    Some(ch)
}

const NONASCII_MASK: usize = usize::from_ne_bytes([0x80; mem::size_of::<usize>()]);

/// Returns `true` if any byte in the word `x` is nonascii (>= 128).
#[inline]
const fn contains_nonascii(x: usize) -> bool {
    (x & NONASCII_MASK) != 0
}

/// Walks through `v` checking that it's a valid UTF-8 sequence,
/// returning `Ok(())` in that case, or, if it is invalid, `Err(err)`.
#[inline]
#[allow(clippy::too_many_lines)]
pub(super) fn run_utf8_validation(v: &Vector<u8>) -> Result<(), Utf8Error> {
    let usize_bytes = mem::size_of::<usize>();
    let ascii_block_size = 2 * usize_bytes;

    let mut success_offset: usize = 0;
    let mut chunk_iter = v.leaves();

    let mut chunk: &[u8] = &[];
    let mut index: usize = 0;
    let mut len: usize = 0;
    let mut blocks_end: usize = 0;
    let mut align: usize = 0;

    macro_rules! err {
        ($error_len: expr) => {
            Err(Utf8Error {
                valid_up_to: success_offset,
                error_len: $error_len,
            })
        };
    }

    macro_rules! advance {
        ($result: expr) => {{
            chunk = match chunk_iter.next() {
                Some(chunk) => chunk,
                None => return $result,
            };
            index = 0;
            len = chunk.len();
            blocks_end = if len >= ascii_block_size {
                len - ascii_block_size + 1
            } else {
                0
            };
            align = chunk.as_ptr().align_offset(usize_bytes);
        }};
    }

    macro_rules! next {
        () => {{
            while (index == len) {
                advance!(err!(None))
            }
            index += 1;
            chunk[index - 1]
        }};
    }

    loop {
        while index == len {
            advance!(Ok(()));
        }

        let first = chunk[index];

        if first >= 128 {
            index += 1;
            let w = utf8_char_width(first);
            // 2-byte encoding is for codepoints  \u{0080} to  \u{07ff}
            //        first  C2 80        last DF BF
            // 3-byte encoding is for codepoints  \u{0800} to  \u{ffff}
            //        first  E0 A0 80     last EF BF BF
            //   excluding surrogates codepoints  \u{d800} to  \u{dfff}
            //               ED A0 80 to       ED BF BF
            // 4-byte encoding is for codepoints \u{1000}0 to \u{10ff}ff
            //        first  F0 90 80 80  last F4 8F BF BF
            //
            // Use the UTF-8 syntax from the RFC
            //
            // https://tools.ietf.org/html/rfc3629
            // UTF8-1      = %x00-7F
            // UTF8-2      = %xC2-DF UTF8-tail
            // UTF8-3      = %xE0 %xA0-BF UTF8-tail / %xE1-EC 2( UTF8-tail ) /
            //               %xED %x80-9F UTF8-tail / %xEE-EF 2( UTF8-tail )
            // UTF8-4      = %xF0 %x90-BF 2( UTF8-tail ) / %xF1-F3 3( UTF8-tail ) /
            //               %xF4 %x80-8F 2( UTF8-tail )
            match w {
                2 => {
                    #[allow(clippy::cast_possible_wrap)]
                    if next!() as i8 >= -64 {
                        return err!(Some(1));
                    }
                    success_offset += 2;
                }
                3 => {
                    #[allow(clippy::unnested_or_patterns)]
                    match (first, next!()) {
                        (0xE0, 0xA0..=0xBF)
                        | (0xE1..=0xEC, 0x80..=0xBF)
                        | (0xED, 0x80..=0x9F)
                        | (0xEE..=0xEF, 0x80..=0xBF) => {}
                        _ => return err!(Some(1)),
                    }

                    if utf8_is_first_byte(next!()) {
                        return err!(Some(2));
                    }
                    success_offset += 3;
                }
                4 => {
                    match (first, next!()) {
                        (0xF0, 0x90..=0xBF) | (0xF1..=0xF3, 0x80..=0xBF) | (0xF4, 0x80..=0x8F) => {}
                        _ => return err!(Some(1)),
                    }
                    if utf8_is_first_byte(next!()) {
                        return err!(Some(2));
                    }
                    if utf8_is_first_byte(next!()) {
                        return err!(Some(3));
                    }
                    success_offset += 4;
                }
                _ => return err!(Some(1)),
            }
        } else {
            // Ascii case, try to skip forward quickly.
            // When the pointer is aligned, read 2 words of data per iteration
            // until we find a word containing a non-ascii byte.
            if align != usize::MAX && align.wrapping_sub(index) % usize_bytes == 0 {
                let ptr = chunk.as_ptr();
                while index < blocks_end {
                    // SAFETY: since `align - index` and `ascii_block_size` are
                    // multiples of `usize_bytes`, `block = ptr.add(index)` is
                    // always aligned with a `usize` so it's safe to dereference
                    // both `block` and `block.add(1)`.
                    unsafe {
                        #[allow(clippy::cast_ptr_alignment)]
                        let block = ptr.add(index).cast::<usize>();
                        // break if there is a nonascii byte
                        let zu = contains_nonascii(*block);
                        let zv = contains_nonascii(*block.add(1));
                        if zu || zv {
                            break;
                        }
                    }
                    index += ascii_block_size;
                    success_offset += ascii_block_size;
                }
                // step from the point where the wordwise loop stopped
                while index < len && chunk[index] < 128 {
                    index += 1;
                    success_offset += 1;
                }
            } else {
                index += 1;
                success_offset += 1;
            }
        }
    }
}

// https://tools.ietf.org/html/rfc3629
const UTF8_CHAR_WIDTH: &[u8; 256] = &[
    // 1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 3
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 4
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 5
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 6
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 7
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 8
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // B
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // D
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // E
    4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // F
];

/// Given a first byte, determines how many bytes are in this UTF-8 character.
#[must_use]
#[inline]
pub(super) const fn utf8_char_width(b: u8) -> usize {
    UTF8_CHAR_WIDTH[b as usize] as usize
}

/// Mask of the value bits of a continuation byte.
const CONT_MASK: u8 = 0b0011_1111;

#[inline]
unsafe fn unwrap_debug<A>(v: Option<A>) -> A {
    debug_assert!(
        v.is_some(),
        "Encountered end-of-iteration on a UTF-8 character non-boundary"
    );
    v.unwrap_unchecked()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::StreamStrategy;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 4096, .. ProptestConfig::default()
          })]
        #[test]
        fn validation_agrees_with_std(s in ".{0,128}", flips in proptest::collection::vec(any::<usize>(), 1usize..3)) {
            let mut bytes = Vec::from(s.as_bytes());
            if !s.is_empty() {
                for flip in flips {
                    let flip_bit = flip % 8;
                    let flip_byte = (flip / 8) % s.len();
                    bytes[flip_byte] ^= 1 << flip_bit;
                }
            }
            let std_result = std::str::from_utf8(bytes.as_slice());
            let my_result = run_utf8_validation(&Vector::from(bytes.as_slice()));
            match std_result {
                Ok(_) => {
                    prop_assert_eq!(my_result, Ok(()));
                },
                Err(std_e) => {
                    prop_assert!(my_result.is_err());
                    let my_e = my_result.unwrap_err();
                    std::mem::drop(format!("{}, {:?}", &my_e, &my_e)); // Just to exercise these implementations
                    prop_assert_eq!(std_e.valid_up_to(), my_e.valid_up_to());
                    prop_assert_eq!(std_e.error_len(), my_e.error_len());
                    prop_assert_eq!(my_e, std_e.into());
                }
            }
        }
    }

    proptest! {
        #[test]
        fn iteration_agrees_with_std(s in ".{0,128}", mut direction in StreamStrategy(any::<bool>())) {
            let mut byte_iter = Vector::from(s.as_bytes()).into_iter();
            let mut char_iter = s.chars();
            loop {
                if direction.gen() {
                    let mine = unsafe { next_code_point(&mut byte_iter).map(|ch| char::from_u32(ch).unwrap()) };
                    let theirs = char_iter.next();
                    prop_assert_eq!(mine, theirs);
                    if mine.is_none() { break; }
                } else {
                    let mine = unsafe { next_code_point_reverse(&mut byte_iter).map(|ch| char::from_u32(ch).unwrap()) };
                    let theirs = char_iter.next_back();
                    prop_assert_eq!(mine, theirs);
                    if mine.is_none() { break; }
                }
            }
        }
    }
}
