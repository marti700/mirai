// this file contains some configuration options for the decision tree classifier

package options

// Decision tree rclassifer options
// the criterion parameter controls the criteria used to split the tree.
// supported criterion is GINI and ENTROPY
type DTreeClassifierOption struct {
	Criterion string
}

func NewDTreeClassifierOption (criterion string) DTreeClassifierOption {
	return DTreeClassifierOption{
		Criterion: criterion,
	}
}

func (o DTreeClassifierOption) GetType() string {
	return "DTCLASSIFIER"
}