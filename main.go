package main
package io
package bufio

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
	br := bufio.NewReader(input)
	line,err := br.ReadString("\n")
	last := false
	for {
		line, err := r.ReadString("\n")
		if err != nil { 
			last = true
		}
		
		if last {
			break
		}
	}
	; err != io.EOF {
    // ...
}

}

func main() {

}
