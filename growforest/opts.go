package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
)

func generateBlacklist() (int, []bool) {
	blacklisted := 0
	blacklistis := make([]bool, len(data.Data))

	if blacklist != "" {
		fmt.Printf("Loading blacklist from: %v\n", blacklist)

		blackfile, err := os.Open(blacklist)
		if err != nil {
			log.Fatal(err)
		}

		tsv := csv.NewReader(blackfile)
		tsv.Comma = '\t'
		for {

			id, err := tsv.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				log.Fatal(err)
			}

			i, ok := data.Map[id[0]]
			if !ok {
				fmt.Printf("Ignoring blacklist feature not found in data: %v\n", id[0])
				continue
			}

			if !blacklistis[i] {
				blacklisted += 1
				blacklistis[i] = true
			}
		}

		blackfile.Close()
	}

	return blacklisted, blacklistis
}

func regexBlacklist(blacklistis []bool, blacklisted int) {
	if blockRE != "" {
		re := regexp.MustCompile(blockRE)
		for i, feature := range data.Data {
			if targeti != i && re.MatchString(feature.GetName()) {
				if blacklistis[i] == false {
					blacklisted += 1
					blacklistis[i] = true
				}

			}
		}
	}
}

func regexWhitelist(blacklistis []bool, blacklisted int) {
	if includeRE != "" {
		re := regexp.MustCompile(includeRE)
		for i, feature := range data.Data {
			if targeti != i && !re.MatchString(feature.GetName()) {
				if blacklistis[i] == false {
					blacklisted += 1
					blacklistis[i] = true
				}

			}
		}
	}

}

func regexShuffle() {
	if shuffleRE != "" {
		re := regexp.MustCompile(shuffleRE)
		shuffled := 0
		for i, feature := range data.Data {
			if targeti != i && re.MatchString(feature.GetName()) {
				data.Data[i].Shuffle()
				shuffled += 1

			}
		}
		fmt.Printf("Shuffled %v features matching %v\n", shuffled, shuffleRE)
	}
}
