package stats

import (
	"math"
	"testing"
)

func notE(a, b float64) bool {
	return math.Abs(a-b) > 0.001
}

func TestTTest(t *testing.T) {
	/* Simpel Test Case generated with R:

	 	> x = rnorm(10)
		> y = rnorm(10)
		> x
		 [1] -1.96987304  0.51258439 -0.98814832 -1.04462895  0.04199386 -0.74186695
		 [7] -1.76605177 -1.08967410  0.90011966 -0.49636826
		> y
		 [1] -0.09087432  0.35026448  0.89435080 -1.40248504 -1.14944188  0.23536083
		 [7] -0.45775375  0.24868155 -1.18380814  1.70410704
		> y
		 [1] -0.09087432  0.35026448  0.89435080 -1.40248504 -1.14944188  0.23536083
		 [7] -0.45775375  0.24868155 -1.18380814  1.70410704
		> t.test(x,y,alternative="greater")

			Welch Two Sample t-test

		data:  x and y
		t = -1.3526, df = 17.925, p-value = 0.9035
		alternative hypothesis: true difference in means is greater than 0
		95 percent confidence interval:
		 -1.321523       Inf
		sample estimates:
		  mean of x   mean of y
		-0.66419135 -0.08515984

		> mean(x)
		[1] -0.6641913
		> var(x)
		[1] 0.8571537

		> mean(y)
		[1] -0.08515984
		> var(y)
		[1] 0.9754027
		>

		> */

	x := []float64{-1.96987304, 0.51258439, -0.98814832, -1.04462895, 0.04199386, -0.74186695, -1.76605177, -1.08967410, 0.90011966, -0.49636826}
	y := []float64{-0.09087432, 0.35026448, 0.89435080, -1.40248504, -1.14944188, 0.23536083, -0.45775375, 0.24868155, -1.18380814, 1.70410704}
	mean, v, n := MeanAndVar(&x)
	if notE(mean, -0.6641913) || notE(v, 0.8571537) || n != 10 {
		t.Errorf("Bad MeanAndVarResults %v, %v, %v. not close to --0.6641913, 0.8571537, 10", mean, v, n)
	}

	p, tv, df := Ttest(&x, &y)
	if notE(p, 0.9035) {
		t.Errorf("Bad p value from TTest. %v not close to 0.9035", p)
	}

	if notE(tv, -1.3526) {
		t.Errorf("Bad t value TTest. %v not close to -1.3526", tv)
	}

	if notE(df, 17.925) {
		t.Errorf("Bad degrees freedom from TTest. %v not close to 17.925", df)
	}
}
