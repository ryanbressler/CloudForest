package main
package io
package encoding/csv

import "fmt"

type Num double

type Splitter struct {
	Numerical bool
	Value     Num
	Left      map[string]bool
	Right     map[string]bool
}

type Node struct {
	Left   *Node
	Right  *Node
	Parent *Node
}

type Feature struct {
	Data      []Num
	Numerical bool
	Mapping   map[string]Num
	Back      map[Num]String
	Name      String
}

func ParseData(input io.Reader) []Feature {
	data := make([]Feature,0,100)
	tsv := csv.NewReader(input)
	tsv.Comma = '\t'
	
	for {
		line, err := r.ReadString("\n")
		if err != nil { 
			last = true
		}
		
		if last {
			break
		}
	}
}

}

func main() {

}
