package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/ryanbressler/CloudForest"
)

func getTarget(targetf CloudForest.Target) CloudForest.Target {
	//****** Set up Target for Alternative Impurity  if needed *******//
	var target CloudForest.Target

	if density {
		fmt.Println("Estimating Density.")
		return &CloudForest.DensityTarget{&data.Data, nNonMissing}
	}

	switch targetf.(type) {
	case CloudForest.NumFeature:
		return getRegressionTarget(targetf)
	case CloudForest.CatFeature:
		return getCategoricalTarget(targetf)
	}

	return target
}

func getRegressionTarget(targetf CloudForest.Target) CloudForest.Target {

	fmt.Println("Performing regression.")

	if l1 {
		fmt.Println("Using l1/absolute deviance error.")
		targetf = &CloudForest.L1Target{targetf.(CloudForest.NumFeature)}
	}
	if ordinal {
		fmt.Println("Using Ordinal (mode) prediction.")
		targetf = CloudForest.NewOrdinalTarget(targetf.(CloudForest.NumFeature))
	}
	switch {
	case gradboost != 0.0:
		fmt.Println("Using Gradient Boosting.")
		targetf = CloudForest.NewGradBoostTarget(targetf.(CloudForest.NumFeature), gradboost)

	case adaboost:
		fmt.Println("Using Numeric Adaptive Boosting.")
		targetf = CloudForest.NewNumAdaBoostTarget(targetf.(CloudForest.NumFeature))
	}

	return targetf
}

func getCategoricalTarget(targetf CloudForest.Target) CloudForest.Target {

	fmt.Printf("Performing classification with %v categories.\n", targetf.NCats())
	switch {
	case NP:
		fmt.Printf("Performing Approximate Neyman-Pearson Classification with constrained false \"%v\".\n", NP_pos)
		fmt.Printf("False %v constraint: %v, constraint weight: %v.\n", NP_pos, NP_a, NP_k)
		targetf = CloudForest.NewNPTarget(targetf.(CloudForest.CatFeature), NP_pos, NP_a, NP_k)

	case costs != "":
		fmt.Println("Using misclassification costs: ", costs)

		costmap := make(map[string]float64)
		if err := json.Unmarshal([]byte(costs), &costmap); err != nil {
			log.Fatal(err)
		}

		regTarg := CloudForest.NewRegretTarget(targetf.(CloudForest.CatFeature))
		regTarg.SetCosts(costmap)
		targetf = regTarg
	case dentropy != "":
		fmt.Println("Using entropy with disutilities: ", dentropy)

		costmap := make(map[string]float64)
		if err := json.Unmarshal([]byte(dentropy), &costmap); err != nil {
			log.Fatal(err)
		}

		deTarg := CloudForest.NewDEntropyTarget(targetf.(CloudForest.CatFeature))
		deTarg.SetCosts(costmap)
		targetf = deTarg

	case adacosts != "":
		fmt.Println("Using cost sensative AdaBoost costs: ", adacosts)
		costmap := make(map[string]float64)

		if err := json.Unmarshal([]byte(adacosts), &costmap); err != nil {
			log.Fatal(err)
		}

		actarget := CloudForest.NewAdaCostTarget(targetf.(CloudForest.CatFeature))
		actarget.SetCosts(costmap)
		targetf = actarget

	case rfweights != "":
		fmt.Println("Using rf weights: ", rfweights)

		weightmap := make(map[string]float64)
		if err := json.Unmarshal([]byte(rfweights), &weightmap); err != nil {
			log.Fatal(err)
		}

		targetf = CloudForest.NewWRFTarget(targetf.(CloudForest.CatFeature), weightmap)

	case entropy:
		fmt.Println("Using entropy minimization.")
		targetf = &CloudForest.EntropyTarget{targetf.(CloudForest.CatFeature)}

	case adaboost:

		fmt.Println("Using Adaptive Boosting.")
		targetf = CloudForest.NewAdaBoostTarget(targetf.(CloudForest.CatFeature))

	case hellinger:
		fmt.Println("Using Hellinger Distance with postive class:", positive)
		targetf = CloudForest.NewHDistanceTarget(targetf.(CloudForest.CatFeature), positive)

	case gradboost != 0.0:
		fmt.Println("Using Gradient Boosting Classification with postive class:", positive)
		targetf = CloudForest.NewGradBoostClassTarget(targetf.(CloudForest.CatFeature), gradboost, positive)

	}

	if unlabeled != "" {
		fmt.Println("Using traduction forests with unlabeled class: ", unlabeled)
		targetf = CloudForest.NewTransTarget(targetf.(CloudForest.CatFeature), &data.Data, unlabeled, trans_alpha, trans_beta, nNonMissing)
	}

	return targetf
}
