package treemodels

import (
	"fmt"
	"os"
	"strconv"

	"github.com/marti700/veritas/linearalgebra"
)

type Tree struct {
	Left      *Tree
	Right     *Tree
	feature		int
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

func (t *Tree) toDotFile() {
	dotStr := `digraph Tree {
		node [shape=box, style="filled, rounded", fontname=helvetica] ;
		edge [fontname=helvetica] ;
		`
	tStr := generateDotFile(t, dotStr, 0)

	f, _ := os.Create("tree.dot")

	f.WriteString(tStr)

}

func generateDotFile(parent *Tree, str string, node int) string {

	var queue []Tree

	queue = append(queue, *parent)
	parentNodeID := 0
	nodeID := 0
	// Root node
	cond := strconv.FormatFloat(parent.Condition, 'E', -1, 64)
	nod := strconv.Itoa(parentNodeID)
	str = str + nod + fmt.Sprintf(`[label="X[%d] <= %s "] ;
	`,parent.feature, cond)
	nodeID++

	for len(queue) != 0 {
		cNode := queue[0]
		queue = queue[1:]
		parentID := strconv.Itoa(parentNodeID)

		if cNode.Left != nil {
			cond1 := strconv.FormatFloat(parent.Left.Condition, 'E', -1, 64)
			nod1 := strconv.Itoa(nodeID)

			str = str + nod1 + fmt.Sprintf(`[label="X[%d] <= %s "] ;
				%s ->  %s ;
				`, cNode.feature, cond1, parentID, nod1)

			nodeID++
			if cNode.Left.Left != nil {
				queue = append(queue, *cNode.Left)
			}
			if cNode.Left.Right != nil {
				queue = append(queue, *cNode.Right)
			}
		}

		if cNode.Right != nil {
			cond2 := strconv.FormatFloat(parent.Right.Condition, 'E', -1, 64)
			nod2 := strconv.Itoa(nodeID)
			str = str + nod2 + fmt.Sprintf(`[label="X[%d] <= %s "] ;
				%s ->  %s ;
				`, cNode.feature, cond2, parentID, nod2)
			nodeID++

			if cNode.Right.Left != nil {
				queue = append(queue, *cNode.Left)
			}
			if cNode.Right.Right != nil {
				queue = append(queue, *cNode.Right)
			}
		}

		parentNodeID++

	}

	return str+"}"
}
