package sortby

import (
	"testing"
)

var cases = []int{10, 1, 6, 3, 4, 9, 8, 7, 2, 0, 5}
var vals = []float64{1.0, 0.1, 0.6, 0.3, 0.4, 0.9, 0.8, 0.7, 0.2, 0.0, 0.5, -1.0}
var binvals = []int{1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0}

func TestShortSort(t *testing.T) {

	cases := []int{0, 4, 3, 1, 2}
	vals := []float64{0.1, 10.1, 10.2, 0, 0}

	SortBy(&cases, &vals)

	for i := 1; i < len(cases); i++ {

		if vals[i] < vals[i-1] {
			t.Errorf("Vals weren't sorted \n%v\n%v ", cases, vals)
		}
	}
}

func TestSortBy(t *testing.T) {

	if len(cases) != 11 || len(vals) != 12 {
		t.Errorf("Cases and vals had wrong length before sort %v and %v not 11 and 12", len(cases), len(vals))
	}

	SortBy(&cases, &vals)

	if len(cases) != 11 || len(vals) != 12 {
		t.Errorf("Cases and vals had wrong length after sort %v and %v not 11 and 12", len(cases), len(vals))
	}

	for i := 1; i < len(cases); i++ {

		if cases[i] < cases[i-1] {
			t.Errorf("Cases weren't sorted: \n%v\n%v ", cases, vals)
		}
		if vals[i] < vals[i-1] {
			t.Errorf("Vals weren't sorted \n%v\n%v ", cases, vals)
		}
		if int(10.0*vals[i]) != cases[i] {
			t.Errorf("Cases and val's don't match at pos %v : %v and %v", i, vals[i], cases[i])
		}
	}

	if vals[11] != -1.0 {
		t.Error("Value in vals beyond the range of cases was not left untouched.")
	}

	SortBy(&cases, &vals)

	for i := 1; i < len(cases); i++ {

		if vals[i] < vals[i-1] {
			t.Errorf("Vals weren't sorted \n%v\n%v ", cases, vals)
		}
	}

}

// func TestHeapSort(t *testing.T) {
// 	heapsort(&cases, &vals, 0, 11)

// 	if len(cases) != 11 || len(vals) != 12 {
// 		t.Errorf("Cases and vals had wrong length after sort %v and %v not 11 and 12", len(cases), len(vals))
// 	}

// 	for i := 1; i < len(cases); i++ {

// 		if cases[i] < cases[i-1] {
// 			t.Errorf("Cases weren't sorted: \n%v\n%v ", cases, vals)
// 		}
// 		if vals[i] < vals[i-1] {
// 			t.Errorf("Vals weren't sorted \n%v\n%v ", cases, vals)
// 		}
// 		if int(10.0*vals[i]) != cases[i] {
// 			t.Errorf("Cases and val's don't match at pos %v : %v and %v", i, vals[i], cases[i])
// 		}
// 	}

// 	if vals[11] != -1.0 {
// 		t.Error("Value in vals beyond the range of cases was not left untouched.")
// 	}
// }

// func TestQuickSort(t *testing.T) {
// 	introsort(&cases, &vals, 0, 11, 11)

// 	if len(cases) != 11 || len(vals) != 12 {
// 		t.Errorf("Cases and vals had wrong length after sort %v and %v not 11 and 12", len(cases), len(vals))
// 	}

// 	for i := 1; i < len(cases); i++ {

// 		if cases[i] < cases[i-1] {
// 			t.Errorf("Cases weren't sorted: \n%v\n%v ", cases, vals)
// 		}
// 		if vals[i] < vals[i-1] {
// 			t.Errorf("Vals weren't sorted \n%v\n%v ", cases, vals)
// 		}
// 		if int(10.0*vals[i]) != cases[i] {
// 			t.Errorf("Cases and val's don't match at pos %v : %v and %v", i, vals[i], cases[i])
// 		}
// 	}

// 	if vals[11] != -1.0 {
// 		t.Error("Value in vals beyond the range of cases was not left untouched.")
// 	}
// }

// func TestIntroSort(t *testing.T) {

// 	introsort(&cases, &vals, 0, 11, 2)

// 	if len(cases) != 11 || len(vals) != 12 {
// 		t.Errorf("Cases and vals had wrong length after sort %v and %v not 11 and 12", len(cases), len(vals))
// 	}

// 	for i := 1; i < len(cases); i++ {

// 		if cases[i] < cases[i-1] {
// 			t.Errorf("Cases weren't sorted: \n%v\n%v ", cases, vals)
// 		}
// 		if vals[i] < vals[i-1] {
// 			t.Errorf("Vals weren't sorted \n%v\n%v ", cases, vals)
// 		}
// 		if int(10.0*vals[i]) != cases[i] {
// 			t.Errorf("Cases and val's don't match at pos %v : %v and %v", i, vals[i], cases[i])
// 		}
// 	}

// 	if vals[11] != -1.0 {
// 		t.Error("Value in vals beyond the range of cases was not left untouched.")
// 	}
// }
