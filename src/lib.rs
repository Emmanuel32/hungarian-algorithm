extern crate bit_vec;
extern crate ndarray;
extern crate rand;

use bit_vec::*;
use ndarray::*;

/// Preforms the hungarian algorithum on a matrix of costs to find
/// the assignment with the minimal overall cost.
pub fn hungarian_algorithum(mut data: Array2<i32>) -> Vec<usize> {
    let (tmp_rows, tmp_cols) = data.dim();
    // Resape the matrix so cols >= rows
    if tmp_cols < tmp_rows {
        // to optimize maybe impliment a real transpose function
        data = data.reversed_axes();
    }
    let (rows, cols) = data.dim();
    // Reduce each row by it's minimum value
    for mut row in data.axis_iter_mut(Axis(0)) {
        let minimum = match row.iter().min() {
            Some(min) => *min,
            None => 0,
        };
        row -= minimum;
    }
    // If it's a square matrix also reduce each collum by it's minimum value
    if rows == cols {
        for mut col in data.axis_iter_mut(Axis(1)) {
            let minimum = match col.iter().min() {
                Some(x) => *x,
                None => 0,
            };
            col -= minimum;
        }
    }

    let mut assigned_zeros = vec![cols; rows];
    let mut primed_zeros = vec![cols; rows];
    let mut covered_rows = BitVec::from_elem(rows, false);
    let mut covered_cols = BitVec::from_elem(cols, false);

    // Assign as many zeros as possible using a basic search
    for ((row, col), &element) in data.indexed_iter() {
        if element == 0 && !covered_cols[col] && (assigned_zeros[row] == cols) {
            assigned_zeros[row] = col;
            covered_cols.set(col, true);
        }
    }

    // Check if compleated
    let mut assignments_left = rows - covered_cols.iter().filter(|x| *x).count();
    if assignments_left == 0 {
        return assigned_zeros;
    }

    let mut did_change = false;
    'main_loop: loop {
        // Prime an non covered zero
        // To optimize only search uncovered rows
        for ((row, col), &element) in data.indexed_iter() {
            if element == 0 && !covered_rows[row] && !covered_cols[col] {
                did_change = true;
                if assigned_zeros[row] == cols {
                    // Follow path and reassign zeros
                    let mut assigned_zeros_by_col = vec![rows; cols];
                    for (row, &col) in assigned_zeros.iter().enumerate() {
                        if col != cols {
                            assigned_zeros_by_col[col] = row;
                        }
                    }
                    let mut prime_row = row;
                    let mut prime_col = col;
                    let mut path = Vec::new();
                    'path_loop: loop {
                        path.push((prime_row, prime_col));
                        if assigned_zeros_by_col[prime_col] == rows {
                            break;
                        } else {
                            prime_row = assigned_zeros_by_col[prime_col];
                            prime_col = primed_zeros[prime_row];
                        }
                    }
                    // this can be optimized away
                    for (row, col) in path {
                        assigned_zeros[row] = col;
                    }

                    // Erase all primes
                    primed_zeros = vec![cols; rows];
                    // Clear all covers
                    covered_cols.clear();
                    covered_rows.clear();

                    // Cover all collums containing an assigned_zeros
                    for &col in assigned_zeros.iter() {
                        if col != cols {
                            covered_cols.set(col, true)
                        }
                    }
                    // Check if compleated
                    assignments_left -= 1;
                    if assignments_left == 0 {
                        return assigned_zeros;
                    }
                    continue 'main_loop;
                } else {
                    primed_zeros[row] = col;
                    covered_rows.set(row, true);
                    covered_cols.set(assigned_zeros[row], false);
                }
            }
        }
        // If there are no uncovered zeros reduce all uncovered values
        // by the minimum uncovered value and add the minimum uncovered
        // value to all the elements in both a covered row and collum
        if !did_change {
            reduce(&mut data, (&covered_rows, &covered_cols));
            // keep all assigned_zeros covers and primes the same
            // to optimize only loop over new zeros added
        }
        did_change = false;
    }
}

fn reduce(data: &mut Array2<i32>, (covered_rows, covered_cols): (&BitVec, &BitVec)) {
    let mut minimum: i32 = std::i32::MAX;
    for ((row, col), &element) in data.indexed_iter() {
        if !covered_rows[row] && !covered_cols[col] && (element < minimum) {
            minimum = element;
        }
    }

    if minimum == 0 || minimum == std::i32::MAX {
        panic!("{:?},{:?}", covered_rows, covered_cols);
    }

    let add_to_assigned_rows: Array1<i32> = covered_cols
        .iter()
        .map(|x| if x { minimum } else { 0 })
        .collect();
    let add_to_unassigned_rows: Array1<i32> = &add_to_assigned_rows - minimum;
    for (i, mut row) in data.outer_iter_mut().enumerate() {
        // could be done with a zip with covered_rows
        if covered_rows[i] == true {
            row += &add_to_assigned_rows;
        } else {
            row += &add_to_unassigned_rows;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_cost(data: &Array2<i32>, assigned_cells: Vec<usize>) -> i32 {
        let mut cost: i32 = 0;
        for (row, &col) in assigned_cells.iter().enumerate() {
            cost += data[[row, col]];
        }
        cost
    }

    #[test]
    fn large_test() {
        let test_data: Array2<i32> = array![
            [59, 34, 46, 36, 41, 89, 20, 71, 15, 50],
            [78, 12, 44, 9, 61, 13, 62, 43, 84, 97],
            [12, 11, 50, 49, 29, 52, 46, 59, 94, 58],
            [40, 57, 75, 50, 14, 47, 19, 25, 98, 8],
            [24, 96, 98, 57, 5, 85, 96, 67, 85, 84],
            [54, 4, 49, 27, 19, 11, 80, 82, 78, 69],
            [77, 77, 72, 44, 35, 94, 66, 25, 3, 26],
            [48, 81, 36, 50, 82, 55, 93, 7, 25, 34],
            [22, 46, 18, 50, 33, 51, 25, 40, 74, 59],
            [53, 62, 53, 48, 70, 26, 11, 73, 72, 97]
        ];
        let expected_result = 112;
        let test_result = get_cost(&test_data, hungarian_algorithum(test_data.clone()));
        assert_eq!(expected_result, test_result);
    }

    #[test]
    fn test_massive() {
        let test_data = Array2::from_shape_fn([1000, 1000], |_| rand::random::<u16>() as i32);
        let mut result = hungarian_algorithum(test_data);
        result.sort();
        println!("{:?}", result);
    }
}
