package options

type RidgeOptions struct {
	Lambda float64
	Estimator Option
}

func NewRidgeOption(lambda float64, estimator Option) RidgeOptions {
	return RidgeOptions{
		Lambda: lambda,
		Estimator: estimator,
	}
}

func (o RidgeOptions) GetType() string {
	return "ridge"
}



