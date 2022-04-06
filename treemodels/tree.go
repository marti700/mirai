package treemodels

import "github.com/marti700/veritas/linearalgebra"

type Tree struct {
	Left *Tree
	Right *Tree
	Condition float64
	Data linearalgebra.Matrix
	Predict float64
}


func (t *Tree) Insert(direction string, tree *Tree) {
	if direction == "LEFT" {
		t.Left = tree;
	} else {
		t.Right = tree
	}
}




