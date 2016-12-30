package main

import (
	"encoding/json"
	"log"
	"os"

	"github.com/ryanbressler/CloudForest"
)

var scikikittrees = make([]CloudForest.ScikitTree, 0, nTrees)

func addScikit(tree *CloudForest.Tree) {
	if scikitforest != "" {
		skt := CloudForest.NewScikitTree(nFeatures)
		CloudForest.BuildScikitTree(0, tree.Root, skt)
		scikikittrees = append(scikikittrees, *skt)
	}
}

func writeScikit() {
	if scikitforest != "" {
		skfile, err := os.Create(scikitforest)
		if err != nil {
			log.Fatal(err)
		}
		defer skfile.Close()
		skencoder := json.NewEncoder(skfile)
		err = skencoder.Encode(scikikittrees)
		if err != nil {
			log.Fatal(err)
		}
	}

}
