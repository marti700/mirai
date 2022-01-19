// this files declares a struct that defines some possible configurations
// for the linear regression model
package options

//Linear regression options
// the estimator field specifies the estimation method to be used to compute the model hyper parameters
// the supported methods are 'GDOption' for the gradient descent and 'OLSOption' for Ordinary Least Square
type LROptions struct {
	Estimator Option
}

func (o LROptions) GetType() string {
	return "LR"
}