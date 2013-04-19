package CloudForest

import ()

type VoteTallyer interface {
	Vote(casei int, vote string)
	TallyError(feature *Feature) float64
}
