/*
Package stats currentelly only implements a welch's t-test for importance score analysis
in CloudForest.
*/
package stats

import (
	"math"
)

//MeanAndVar returns the sample mean, variance and count as float64's.
func MeanAndVar(X *[]float64) (m, v, n float64) {
	for _, x := range *X {
		m += x
		v += x * x
	}
	n = float64(len(*X))
	m /= n
	v -= n * m * m
	v /= (n - 1.0)
	return
}

//Ttest performs a Welch's t test and returns p-value, t-value and the degrees of freedom.
//The p value based on the hypothesis that mean(B)>mean(A).
//Based on similar code in rf-ace (Apache 2.0, Timo Erkkilä)
func Ttest(A, B *[]float64) (p, t, v float64) {

	// Calculate means and variances for each of two samples.
	Am, Av, An := MeanAndVar(A)
	Bm, Bv, Bn := MeanAndVar(B)

	//Welch's t test
	As := Av / An
	Bs := Bv / Bn
	s := As + Bs
	t = (Am - Bm) / math.Sqrt(s)

	// Degree's Freedom for Welch's
	v = s * s / (As*As/(An-1) + Bs*Bs/(Bn-1))

	// Find the tail probability  of t

	// Transformed t-test statistic
	ttrans := v / (t*t + v)

	// This variable will store the integral of the tail of the t-distribution
	integral := 0.0

	// Comment from rf-ace:
	// When ttrans > 0.9, we need to recast the integration in order to retain
	// accuracy. In other words we make use of the following identity:
	//
	// I(x,a,b) = 1 - I(1-x,b,a)
	if ttrans > 0.9 {
		// Calculate I(x,a,b) as 1 - I(1-x,b,a)
		integral = 1 - regularizedIncompleteBeta(1-ttrans, 0.5, v/2)

	} else {
		// Calculate I(x,a,b) directly
		integral = regularizedIncompleteBeta(ttrans, v/2, 0.5)
	}

	// Comment from rf-ace:
	// We need to be careful about which way to calculate the integral so that it represents
	// the tail of the t-distribution. The sign of the tvalue hints which way to integrate
	if t > 0.0 {
		p = (integral / 2)
	} else {
		p = (1 - integral/2)
	}
	return
}

//Based on similar code in rf-ace (Apache 2.0, Timo Erkkilä)
func regularizedIncompleteBeta(x, a, b float64) float64 {
	i := 50
	continuedFraction := 1.0
	m := 0.0

	for i > 0 {
		m = float64(i)
		continuedFraction = 1.0 + dE(m, x, a, b)/(1+dO(m, x, a, b)/continuedFraction)
		i--
	}
	return (math.Pow(x, a) * math.Pow(1-x, b) / (a * beta(a, b) * (1 + dO(0, x, a, b)/continuedFraction)))
}

/*
  Even and odd factors for the infinite continued fraction representation of the
  regularized incomplete beta function.
  Based on similar code in rf-ace (Apache 2.0, Timo Erkkilä)
*/
func dO(m, x, a, b float64) float64 {
	return (-1.0 * (a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1)))
}

func dE(m, x, a, b float64) float64 {
	return (m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m)))
}

//Based on similar code in rf-ace (Apache 2.0, Timo Erkkilä)
func lgamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	//v := math.Log(math.Abs(math.Gamma(x)))
	return v
}

//Based on similar code in rf-ace (Apache 2.0, Timo Erkkilä)
func beta(a, b float64) float64 {
	return (math.Exp(lgamma(a) + lgamma(b) - lgamma(a+b)))
}
