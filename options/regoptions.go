package options

// represents regularization options that can be passed to the linear regression model
// supported regularization types are ridge (l2 regularization) and lasso (l1 regularizations)
// the lambda parameter controls how strong is the penalty applied to the model
type RegOptions struct {
	Type string
	Lambda float64
}

func NewRegOptions(kind string, lambda float64) RegOptions {
	return RegOptions{
		Type: kind,
		Lambda: lambda,
	}
}

func (o RegOptions) GetType() string {
	return "reg"
}



