package CloudForest

import ()

//VoteTallyer  is used to tabulate votes by trees and is implemented by feature type specific
//structs like NumBallotBox and CatBallotBox.
//Vote should register a cote that casei should be predicted as pred.
//TallyError returns the error vs the supplied feature.
type VoteTallyer interface {
	Vote(casei int, pred string, weight float64)
	TallyError(feature Feature) float64
	Tally(casei int) string
}
