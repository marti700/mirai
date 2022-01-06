// this files defines a struct that defines some possible configurations 
// for the linear regression model
package linearmodels

//Linear regression options
// the estimator field specifies the stimation method to be used to stimate the model hyper parameters
// the supported method are 'gd' for the gradient descent and 'OLS' for Ordinary Least Square
// the learning rate field should only be specified when estimation = 'gd'
type LROptions struct {
	Estimator string
	LearningRate float64
}