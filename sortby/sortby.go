/*Package sortby is a  hybrid, non stable sort based on go's standard sort but with
all less function and many swaps inlined to sort a list of ints by an acompanying list
of floats as needed in random forest training. It is about 30-40% faster then the
standard sort.*/
package sortby

import ()

//SortBy will sort the values in cases and vals by the values in vals in increasing order.
//If vals is longer then cases only the coresponding section will be sorted.
func SortBy(cases *[]int, vals *[]float64) {
	n := len(*cases)
	// Switch to heapsort if depth of 2*ceil(lg(n+1)) is reached.
	maxDepth := 0
	for i := n; i > 0; i >>= 1 {
		maxDepth++
	}
	maxDepth *= 2
	quickSort(cases, vals, 0, n, maxDepth)
	//introsort(cases, vals, 0, n, maxd)
	//heapsort(cases, vals, 0, n)
}

//Swap exchanges the ith and jth cases.
func swap(cases *[]int, vals *[]float64, i int, j int) {
	//swap(cases, vals,
	c := *cases
	v := c[i]
	c[i] = c[j]
	c[j] = v
	vs := *vals
	w := vs[i]
	vs[i] = vs[j]
	vs[j] = w

}

func quickSort(cases *[]int, vals *[]float64, a, b, maxDepth int) {
	for b-a > 7 {
		if maxDepth == 0 {
			heapSort(cases, vals, a, b)
			return
		}
		maxDepth--
		mlo, mhi := doPivot(cases, vals, a, b)
		// Avoiding recursion on the larger subproblem guarantees
		// a stack depth of at most lg(b-a).
		if mlo-a < b-mhi {
			quickSort(cases, vals, a, mlo, maxDepth)
			a = mhi // i.e., quickSort(cases, vals, mhi, b)
		} else {
			quickSort(cases, vals, mhi, b, maxDepth)
			b = mlo // i.e., quickSort(cases, vals, a, mlo)
		}
	}
	if b-a > 1 {
		insertionSort(cases, vals, a, b)
	}
}

func doPivot(cases *[]int, vals *[]float64, lo, hi int) (midlo, midhi int) {
	cs := *cases
	vs := *vals
	var swapi int
	var swapf float64
	m := lo + (hi-lo)/2 // Written like this to avoid integer overflow.
	if hi-lo > 40 {
		// Tukey's ``Ninther,'' median of three medians of three.
		s := (hi - lo) / 8
		medianOfThree(cases, vals, lo, lo+s, lo+2*s)
		medianOfThree(cases, vals, m, m-s, m+s)
		medianOfThree(cases, vals, hi-1, hi-1-s, hi-1-2*s)
	}
	medianOfThree(cases, vals, lo, m, hi-1)

	// Invariants are:
	//	data[lo] = pivot (set up by ChoosePivot)
	//	data[lo <= i < a] = pivot
	//	data[a <= i < b] < pivot
	//	data[b <= i < c] is unexamined
	//	data[c <= i < d] > pivot
	//	data[d <= i < hi] = pivot
	//
	// Once b meets c, can swap the "= pivot" sections
	// into the middle of the slice.
	pivotv := vs[lo]
	a, b, c, d := lo+1, lo+1, hi, hi
	for {
		for b < c {
			if vs[b] < pivotv { // data[b] < pivot
				b++
			} else if pivotv == vs[b] { // data[b] = pivot
				//swap(cases, vals, a, b)
				//vs[a], vs[b] = vs[b], vs[a]
				//cs[a], cs[b] = cs[b], cs[a]
				swapf = vs[a]
				vs[a] = vs[b]
				vs[b] = swapf

				swapi = cs[a]
				cs[a] = cs[b]
				cs[b] = swapi

				a++
				b++
			} else {
				break
			}
		}
		for b < c {
			if pivotv < vs[c-1] { // data[c-1] > pivot
				c--
			} else if vs[c-1] == pivotv { // data[c-1] = pivot
				c--
				d--
				//swap(cases, vals, c, d)
				// vs[c], vs[d] = vs[d], vs[c]
				// cs[c], cs[d] = cs[d], cs[c]
				swapf = vs[c]
				vs[c] = vs[d]
				vs[d] = swapf

				swapi = cs[c]
				cs[c] = cs[d]
				cs[d] = swapi

			} else {
				break
			}
		}
		if b >= c {
			break
		}
		// data[b] > pivot; data[c-1] < pivot

		c--
		//swap(cases, vals, b, c)
		// vs[b], vs[c] = vs[c], vs[b]
		// cs[b], cs[c] = cs[c], cs[b]
		swapf = vs[c]
		vs[c] = vs[b]
		vs[b] = swapf

		swapi = cs[c]
		cs[c] = cs[b]
		cs[b] = swapi
		b++

	}

	n := min(b-a, a-lo)
	swapRange(cases, vals, lo, b-n, n)

	n = min(hi-d, d-c)
	swapRange(cases, vals, c, hi-n, n)

	return lo + b - a, hi - (d - c)
}

// medianOfThree moves the median of the three values data[a], data[b], data[c] into data[a].
func medianOfThree(cases *[]int, vals *[]float64, a, b, c int) {
	vs := *vals
	//cs := *cases
	m0 := b
	m1 := a
	m2 := c
	// bubble sort on 3 elements
	if vs[m1] < vs[m0] {
		swap(cases, vals, m1, m0)
	}
	if vs[m2] < vs[m1] {
		swap(cases, vals, m2, m1)
	}
	if vs[m1] < vs[m0] {
		swap(cases, vals, m1, m0)
	}
	// now data[m0] <= data[m1] <= data[m2]
}

func swapRange(cases *[]int, vals *[]float64, a, b, n int) {
	vs := *vals
	cs := *cases
	//var api, bpi = a, b
	// var swapi int
	// var swapf float64
	for i := 0; i < n; i++ {
		//swap(cases, vals, a, b+i)
		// vs[a+i], vs[b+i] = vs[b+i], vs[a+i]
		// cs[a+i], cs[b+i] = cs[b+i], cs[a+i]
		// swapf = vs[a]
		// vs[a] = vs[b]
		// vs[b] = swapf

		// swapi = cs[a]
		// cs[a] = cs[b]
		// cs[b] = swapi

		vs[a], vs[b] = vs[b], vs[a]
		cs[a], cs[b] = cs[b], cs[a]
		a++
		b++
	}
}

// Insertion sort
func insertionSort(cases *[]int, vals *[]float64, a, b int) {
	vs := *vals
	//cs := *cases
	for i := a + 1; i < b; i++ {
		for j := i; j > a && vs[j] < vs[j-1]; j-- {
			swap(cases, vals, j, j-1)
		}
	}
}

// siftDown implements the heap property on data[lo, hi).
// first is an offset into the array where the root of the heap lies.
func siftDown(cases *[]int, vals *[]float64, lo, hi, first int) {
	vs := *vals
	root := lo
	for {
		child := 2*root + 1
		if child >= hi {
			break
		}
		if child+1 < hi && vs[first+child] < vs[first+child+1] {
			child++
		}
		if vs[first+root] >= vs[first+child] {
			return
		}
		swap(cases, vals, first+root, first+child)
		root = child
	}
}

func heapSort(cases *[]int, vals *[]float64, a, b int) {
	first := a
	lo := 0
	hi := b - a

	// Build heap with greatest element at top.
	for i := (hi - 1) / 2; i >= 0; i-- {
		siftDown(cases, vals, i, hi, first)
	}

	// Pop elements, largest first, into end of data.
	for i := hi - 1; i >= 0; i-- {
		swap(cases, vals, first, first+i)
		siftDown(cases, vals, lo, i, first)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
