pub fn one_hot_encode<T>(data: &[T], alphabet: &[T]) -> Vec<Vec<f64>>
where
    T: Eq,
{
    data.iter()
        .map(|value| {
            alphabet
                .iter()
                .map(|letter| if letter == value { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

pub fn flatten<T>(data: Vec<Vec<T>>) -> Vec<T> {
    data.into_iter().flatten().collect()
}

pub fn pad_end(data: Vec<f64>, length: usize) -> Vec<f64> {
    let mut padded = vec![0.0; length];
    padded[..data.len()].copy_from_slice(&data);
    padded
}
