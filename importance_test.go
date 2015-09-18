package CloudForest

import (
	"strings"
	"testing"
)

func TestImportance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping importance test on boston data set.")
	}
	boston := strings.NewReader(boston_housing)

	fm := ParseARFF(boston)

	if len(fm.Data) != 14 {
		t.Errorf("Boston feature matrix has %v features not 14", len(fm.Data))
	}

	// add artifical contrasts
	fm.ContrastAll()

	targeti := fm.Map["class"]

	candidates := make([]int, 0, 0)

	for i := 0; i < len(fm.Data); i++ {
		if i != targeti {
			candidates = append(candidates, i)
		}
	}

	numtarget := fm.Data[targeti]

	nTrees := 20
	//Brieman's importance definition
	imp := func(mean float64, count float64) float64 {
		return mean * float64(count) / float64(nTrees)
	}

	//standard
	imppnt := NewRunningMeans(len(fm.Data))
	_ = GrowRandomForest(fm, numtarget.(Feature), candidates, fm.Data[0].Length(), 6, nTrees, 1, 0, false, false, false, false, imppnt)
	//TODO read importance scores and verify RM and LSTAT come out on top

	roomimp := imp((*imppnt)[fm.Map["RM"]].Read())
	lstatimp := imp((*imppnt)[fm.Map["LSTAT"]].Read())
	beatlstat := 0
	beatroom := 0

	for _, rm := range *imppnt {
		fimp := imp(rm.Read())
		if fimp > roomimp {
			beatroom++
		}
		if fimp > lstatimp {
			beatlstat++
		}
	}
	if beatroom > 1 || beatlstat > 1 {
		t.Error("RM and LSTAT features  not most important in boston data set regression.")
	}

	//vetting
	imppnt = NewRunningMeans(len(fm.Data))
	_ = GrowRandomForest(fm, numtarget.(Feature), candidates, fm.Data[0].Length(), 6, nTrees, 1, 0, false, false, true, false, imppnt)
	//TODO read importance scores and verify RM and LSTAT come out on top

	roomimp = imp((*imppnt)[fm.Map["RM"]].Read())
	lstatimp = imp((*imppnt)[fm.Map["LSTAT"]].Read())
	beatlstat = 0
	beatroom = 0

	for _, rm := range *imppnt {
		fimp := imp(rm.Read())
		if fimp > roomimp {
			beatroom++
		}
		if fimp > lstatimp {
			beatlstat++
		}
	}
	if beatroom > 1 || beatlstat > 1 {
		t.Error("RM and LSTAT features  not most important in vetted boston data set regression.")
	}

	//evaloob
	//vetting
	imppnt = NewRunningMeans(len(fm.Data))
	_ = GrowRandomForest(fm, numtarget.(Feature), candidates, fm.Data[0].Length(), 6, nTrees, 1, 0, false, false, false, true, imppnt)
	//TODO read importance scores and verify RM and LSTAT come out on top

	roomimp = imp((*imppnt)[fm.Map["RM"]].Read())
	lstatimp = imp((*imppnt)[fm.Map["LSTAT"]].Read())
	beatlstat = 0
	beatroom = 0

	for _, rm := range *imppnt {
		fimp := imp(rm.Read())
		if fimp > roomimp {
			beatroom++
		}
		if fimp > lstatimp {
			beatlstat++
		}
	}
	if beatroom > 1 || beatlstat > 1 {
		t.Error("RM and LSTAT features  not most important in boston data set regression with eval oob.")
	}

}
