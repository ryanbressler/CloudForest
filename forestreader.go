package CloudForest

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"
)

/*
ForestReader wraps an io.Reader to reads a forest. It includes ReadForest for reading an
entire forest or ReadTree for reading a forest tree by tree.
The forest should be in .sf format see the package doc's in doc.go for full format details.
It ignores fields that are not use by CloudForest.
*/
type ForestReader struct {
	br *bufio.Reader
}

//NewForestReader wraps the supplied io.Reader as a ForestReader.
func NewForestReader(r io.Reader) *ForestReader {
	return &ForestReader{bufio.NewReader(r)}
}

/*
ForestReader.ReadForest reads the next forest from the underlying reader.
If io.EOF or another error is encountered it returns that.
*/
func (fr *ForestReader) ReadForest() (forest *Forest, err error) {
	peek := []byte(" ")
	peek, err = fr.br.Peek(1)
	if err != nil {
		return
	}
	if peek[0] != 'F' {
		err = errors.New("Forest Header Not Found.")
		return
	}
	for {
		peek, err = fr.br.Peek(1)
		if peek[0] == 'F' && forest != nil {
			return
		}
		t, f, e := fr.ReadTree()
		if forest != nil && f != nil {
			return
		}
		if forest == nil && f != nil {
			forest = f
		}
		if t != nil {
			forest.Trees = append(forest.Trees, t)
		}
		if e == io.EOF {
			return forest, nil
		}
		if e != nil {
			return
		}

	}
}

/*ForestReader.ReadTree reads the next tree from the underlying reader. If the next tree
is in a new forest it returns a forest object as well. If an io.EOF or other error is
encountered it returns that as well as any partially parsed structs.*/
func (fr *ForestReader) ReadTree() (tree *Tree, forest *Forest, err error) {
	intree := false
	line := ""
	peek := []byte(" ")
	for {
		peek, err = fr.br.Peek(1)
		//If their is no next line or it starts a new Tree or Forest return
		if err != nil || (intree && (peek[0] == 'T' || peek[0] == 'F')) {
			return
		}

		line, err = fr.br.ReadString('\n')
		if err != nil {
			return
		}
		parsed := fr.ParseRfAcePredictorLine(line)
		switch {
		case strings.HasPrefix(line, "FOREST"):
			forest = new(Forest)
			forest.Target = parsed["TARGET"]

		case strings.HasPrefix(line, "TREE"):
			intree = true
			tree = new(Tree)

		case strings.HasPrefix(line, "NODE"):
			if intree == false {
				err = errors.New("Poorly formed .sf file. Node found outside of tree.")
				return
			}
			var splitter *Splitter

			pred := ""
			if filepred, ok := parsed["PRED"]; ok {
				pred = filepred
			}

			if stype, ok := parsed["SPLITTERTYPE"]; ok {
				splitter = new(Splitter)
				splitter.Feature = parsed["SPLITTER"]
				switch stype {
				case "CATEGORICAL":
					splitter.Numerical = false

					splitter.Left = make(map[string]bool)
					for _, f := range strings.Split(parsed["LVALUES"], ":") {
						splitter.Left[f] = true
					}

				case "NUMERICAL":
					splitter.Numerical = true
					lvalue, err := strconv.ParseFloat(parsed["LVALUES"], 64)
					if err != nil {
						log.Print("Error parsing lvalues value ", err)
					}
					splitter.Value = float64(lvalue)
				}
			}

			tree.AddNode(parsed["NODE"], pred, splitter)

		}
	}

}

/*
ParseRfAcePredictorLine parses a single line of an rf-ace sf "stoicastic forest"
and returns a map[string]string of the key value pairs.
*/
func (fr *ForestReader) ParseRfAcePredictorLine(line string) map[string]string {
	clauses := make([]string, 0)
	insidequotes := make([]string, 0)
	terms := strings.Split(strings.TrimSpace(line), ",")
	for _, term := range terms {
		term = strings.TrimSpace(term)
		quotes := strings.Count(term, "\"")
		//if quotes have been opend join terms
		if quotes == 1 || len(insidequotes) > 0 {
			insidequotes = append(insidequotes, term)
		} else {
			//If the term doesn't have an = in it join it to the last term
			if strings.Count(term, "=") == 0 {
				clauses[len(clauses)-1] += "," + term
			} else {
				clauses = append(clauses, term)
			}
		}
		//quotes were closed
		if quotes == 1 && len(insidequotes) > 1 {
			clauses = append(clauses, strings.Join(insidequotes, ","))
			insidequotes = make([]string, 0)
		}

	}
	parsed := make(map[string]string, 0)
	for _, clause := range clauses {
		vs := strings.Split(clause, "=")
		for i, v := range vs {
			vs[i] = strings.Trim(strings.TrimSpace(v), "\"")
		}
		if len(vs) != 2 {
			fmt.Println("Parser Choked on : \"", line, "\"")
		}
		parsed[vs[0]] = vs[1]
	}

	return parsed
}

/*TODO: refactor into forestreader struct that supports tree by tree analysis*/

/*ParseRfAcePredictor reads a forest from an io.Reader.
The forest should be in rf-ace's "stoicastic forest" sf format
It ignores fields that are not use by cloud forest.
Start of an example file:

	FOREST=RF,TARGET="N:CLIN:TermCategory:NB::::",NTREES=12800,CATEGORIES="",SHRINKAGE=0
	TREE=0
	NODE=*,PRED=3.48283,SPLITTER="B:SURV:Family_Thyroid:F::::maternal",SPLITTERTYPE=CATEGORICAL,LVALUES="false",RVALUES="true"
	NODE=*L,PRED=3.75
	NODE=*R,PRED=1

Node should be a path the form *LRL where * indicates the root L and R indicate Left and Right.
func ParseRfAcePredictor(input io.Reader) *Forest {
	r := bufio.NewReader(input)
	var forest *Forest
	var tree *Tree
	for {
		line, err := r.ReadString('\n')
		if err == io.EOF {
			break
		}
		parsed := ParseRfAcePredictorLine(line)
		switch {
		case strings.HasPrefix(line, "FOREST"):
			forest = new(Forest)
			forest.Target = parsed["TARGET"]

		case strings.HasPrefix(line, "TREE"):
			tree = new(Tree)
			forest.Trees = append(forest.Trees, tree)

		case strings.HasPrefix(line, "NODE"):
			var splitter *Splitter

			pred := ""
			if filepred, ok := parsed["PRED"]; ok {
				pred = filepred
			}

			if stype, ok := parsed["SPLITTERTYPE"]; ok {
				splitter = new(Splitter)
				splitter.Feature = parsed["SPLITTER"]
				switch stype {
				case "CATEGORICAL":
					splitter.Numerical = false

					splitter.Left = make(map[string]bool)
					for _, f := range strings.Split(parsed["LVALUES"], ":") {
						splitter.Left[f] = true
					}

				case "NUMERICAL":
					splitter.Numerical = true
					lvalue, err := strconv.ParseFloat(parsed["LVALUES"], 64)
					if err != nil {
						log.Print("Error parsing lvalues value ", err)
					}
					splitter.Value = float64(lvalue)
				}
			}

			tree.AddNode(parsed["NODE"], pred, splitter)

		}
	}

	return forest

}*/
