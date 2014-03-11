/*Package sortby is a  hybrid, non stable sort based on go's standard sort but with
all less function and many swaps inlined to sort a list of ints by an acompanying list
of floats as needed in random forest training. It is about 30-40% faster then the
standard sort.*/
package sortby

import ()

//Sortby will sort the values in cases and vals by the values in vals in increasing order.
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
			} else if pivotv >= vs[b] { // data[b] = pivot
				//swap(cases, vals, a, b)
				vs[a], vs[b] = vs[b], vs[a]
				cs[a], cs[b] = cs[b], cs[a]
				a++
				b++
			} else {
				break
			}
		}
		for b < c {
			if pivotv < vs[c-1] { // data[c-1] > pivot
				c--
			} else if vs[c-1] >= pivotv { // data[c-1] = pivot
				//swap(cases, vals, c-1, d-1)
				vs[c-1], vs[d-1] = vs[d-1], vs[c-1]
				cs[c-1], cs[d-1] = cs[d-1], cs[c-1]
				c--
				d--
			} else {
				break
			}
		}
		if b >= c {
			break
		}
		// data[b] > pivot; data[c-1] < pivot
		//swap(cases, vals, b, c-1)
		vs[b], vs[c-1] = vs[c-1], vs[b]
		cs[b], cs[c-1] = cs[c-1], cs[b]
		b++
		c--
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
	for i := 0; i < n; i++ {
		swap(cases, vals, a+i, b+i)
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

// func median3(vals *[]float64, start int, end int) float64 {
// 	vs := *vals
// 	a := vs[start]
// 	b := vs[start+(end-start)/2] //avoid int overflows
// 	c := vs[end-1]
// 	switch {
// 	case a < b:
// 		switch {
// 		case b < c:
// 			return b
// 		case a < c:
// 			return c
// 		default:
// 			return a
// 		}
// 	case b < c:
// 		if a < c {
// 			return a
// 		} else {
// 			return c
// 		}
// 	default:
// 		return b
// 	}

// }

// func introsort(cases *[]int, vals *[]float64, start int, end int, maxd int) {
// 	//cs := *cases
// 	vs := *vals
// 	var pivot float64
// 	var i, l, r int

// 	for (end - start) > 1 {
// 		if maxd <= 0 {
// 			//fmt.Println("heap sorting !! ", start, " ", end)
// 			heapsort(cases, vals, start, end)
// 			return
// 		}

// 		maxd--
// 		pivot = median3(vals, start, end)
// 		//i = l = 0
// 		i = start
// 		l = start
// 		r = end
// 		for i < r {
// 			switch {
// 			case vs[i] < pivot:
// 				swap(cases, vals, i, l)
// 				// vs[i], vs[l] = vs[l], vs[i]
// 				// cs[i], cs[l] = cs[l], cs[i]

// 				i++
// 				l++
// 			case vs[i] >= pivot:
// 				r--
// 				swap(cases, vals, i, r)
// 				// vs[i], vs[r] = vs[r], vs[i]
// 				// cs[i], cs[r] = cs[r], cs[i]
// 			default:
// 				i++
// 			}

// 		}
// 		introsort(cases, vals, start, l, maxd)
// 		start = r
// 	}

// }

// func siftdown(cases *[]int, vals *[]float64, start int, end int) {
// 	cs := *cases
// 	vs := *vals
// 	var child, maxind, root int
// 	root = start
// 	for {
// 		child = root*2 + 1
// 		maxind = root
// 		if child < end && vs[maxind] < vs[child] {
// 			maxind = child
// 		}
// 		if child+1 < end && vs[maxind] < vs[child+1] {
// 			maxind = child + 1
// 		}

// 		if maxind == root {
// 			return
// 		} else {
// 			//swap(cases, vals, root, maxind)
// 			vs[root], vs[maxind] = vs[maxind], vs[root]
// 			cs[root], cs[maxind] = cs[maxind], cs[root]
// 			root = maxind
// 		}
// 	}
// }

// func heapsort(cases *[]int, vals *[]float64, s int, e int) {
// 	cs := *cases
// 	vs := *vals
// 	var start, end int
// 	start = s + (e-s-2)/2 //avoid integer overflows
// 	end = e
// 	for {
// 		siftdown(cases, vals, start, end)
// 		if start == s {
// 			break
// 		}
// 		start--
// 	}
// 	end = e - 1
// 	for end > s {
// 		//swap(cases, vals, s, end)
// 		vs[s], vs[end] = vs[end], vs[s]
// 		cs[s], cs[end] = cs[end], cs[s]
// 		siftdown(cases, vals, s, end)
// 		end = end - 1
// 	}

// }
