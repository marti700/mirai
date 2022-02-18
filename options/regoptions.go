package options

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



