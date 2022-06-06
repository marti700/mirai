package treemodels

import (
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/marti700/veritas/linearalgebra"
)

type Tree struct {
	Left      *Tree
	Right     *Tree
	feature   int
	Condition float64
	Data      linearalgebra.Matrix
	Predict   float64
}

func (t *Tree) Insert(direction string, tree *Tree) {
	if direction == "LEFT" {
		t.Left = tree
	} else {
		t.Right = tree
	}
}

func (t *Tree) Plot() {
	dotStr := `digraph Tree {
		node [shape=box, style="filled, rounded", fontname=helvetica] ;
		edge [fontname=helvetica] ;
		`
	tStr := generateDotFile(t, dotStr, 0)

	f, err := os.Create("tree.dot")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	f.WriteString(tStr)

}

func generateDotFile(parent *Tree, str string, node int) string {

	var queue []Tree

	queue = append(queue, *parent)
	parentNodeID := 0
	nodeID := 0
	// Root node
	cond := strconv.FormatFloat(parent.Condition, 'f', -1, 64)
	nod := strconv.Itoa(parentNodeID)
	str = str + nod + fmt.Sprintf(`[label="X[%d] <= %s "] ;
	`, parent.feature, cond)
	nodeID++

	for len(queue) != 0 {
		cNode := queue[0]
		queue = queue[1:]
		parentID := strconv.Itoa(parentNodeID)

		if cNode.Left != nil {
			queue = append(queue, *cNode.Left)
			nod1 := strconv.Itoa(nodeID)

			if !isTerminalNode(cNode.Left) {
				str = str + nod1 + fmt.Sprintf(`[label="X[%d] <= %f "] ;
				%s ->  %s ;
				`, cNode.Left.feature, cNode.Left.Condition, parentID, nod1)

				nodeID++
			} else {
				str = str + nod1 + fmt.Sprintf(`[label="class: %f "] ;
				%s ->  %s ;
				`, cNode.Left.Predict, parentID, nod1)

				nodeID++
			}
		}

		if cNode.Right != nil {
			queue = append(queue, *cNode.Right)
			nod2 := strconv.Itoa(nodeID)

			if !isTerminalNode(cNode.Right) {
				str = str + nod2 + fmt.Sprintf(`[label="X[%d] <= %f "] ;
				%s ->  %s ;
				`, cNode.Right.feature, cNode.Right.Condition, parentID, nod2)
				nodeID++
			} else {
				str = str + nod2 + fmt.Sprintf(`[label="class: %f "] ;
				%s ->  %s ;
				`, cNode.Right.Predict, parentID, nod2)

				nodeID++
			}
		}

		parentNodeID++
	}

	return str + "}"
}

func isTerminalNode(node *Tree) bool {
	if node.Left == nil && node.Right == nil {
		return true
	}
	return false
}
