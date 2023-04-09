pub fn boxed_array<T: Clone, const SIZE: usize>(initial_value: T) -> Box<[T; SIZE]> {
    let array = vec![initial_value; SIZE];
    // The only way to make this work is to use raw pointers to subvert Rust's type system.
    // It is about as ugly as it gets but there is no better way that I am aware of.
    let array_pointer = Box::into_raw(array.into_boxed_slice()) as *mut [T; SIZE];
    unsafe { Box::from_raw(array_pointer) }
}
