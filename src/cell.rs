use std::fmt::{Debug, Formatter};

macro_rules! cell {
    ($cell:ident, $signed:ty, $unsigned:ty) => {
        #[derive(Copy, Clone)]
        pub union $cell {
            signed: $signed,
            unsigned: $unsigned,
        }

        impl Default for $cell {
            fn default() -> Self {
                Self { signed: 0 }
            }
        }

        impl $cell {
            pub fn signed(&self) -> $signed {
                unsafe { self.signed }
            }

            pub fn unsigned(&self) -> $unsigned {
                unsafe { self.unsigned }
            }
        }

        impl From<$signed> for $cell {
            fn from(value: $signed) -> Self {
                Self { signed: value }
            }
        }

        impl From<$unsigned> for $cell {
            fn from(value: $unsigned) -> Self {
                Self { unsigned: value }
            }
        }

        impl PartialEq for $cell {
            fn eq(&self, other: &Self) -> bool {
                self.unsigned() == other.unsigned()
            }
        }

        impl Eq for $cell {}

        impl Debug for $cell {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                <$signed as Debug>::fmt(&self.signed(), f)
            }
        }
    };
}

macro_rules! convert_cell {
    ($small:ident ($small_unsigned:ty) * $factor:literal = $big:ident ($big_unsigned:ty), $($part:literal at $index:literal),* $(,)?) => {
        impl From<[$small; $factor]> for $big {
            fn from(value: [$small; $factor]) -> Self {
                Self {
                    unsigned: $((value[$part].unsigned() as $big_unsigned) << $index)|*
                }
            }
        }

        impl TryFrom<&[$small]> for $big {
            type Error = ();

            fn try_from(value: &[$small]) -> Result<Self, Self::Error> {
                if value.len() == $factor {
                    Ok(Self::from([$(value[$part]),*]))
                } else {
                    Err(())
                }
            }
        }

        impl From<$big> for [$small; $factor] {
            fn from(value: $big) -> Self {
                [$($small::from((value.unsigned() >> $index) as $small_unsigned)),*]
            }
        }
    };
}

cell!(Cell8, i8, u8);
cell!(Cell16, i16, u16);
cell!(Cell32, i32, u32);

convert_cell!(
    Cell8 (u8) * 2 = Cell16 (u16),
    0 at 0,
    1 at 8,
);

convert_cell!(
    Cell8 (u8) * 4 = Cell32 (u32),
    0 at 0,
    1 at 8,
    2 at 16,
    3 at 24,
);

convert_cell!(
    Cell16 (u16) * 2 = Cell32 (u32),
    0 at 0,
    1 at 16,
);
