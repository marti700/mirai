// this files defines a struct that defines some possible configurations 
// for the linear regression model

package options

// represents the options for the Gradiant Descend estimator
// Iterations: is the max number of iterations the algorithm will perform
// LearningRate: the gradiand descent learning rate
// MinStepSize: if the difference betwen the last estimation and this estimation is less or equal to this number
// the algorithm will stop iterating even if it have not reach the max number Iterations value
type GDOptions struct {
	Iteations int
	LearningRate float64
	MinStepSize float64
}

func NewGDOptions(iterations int, learningRate, minStepSize float64) GDOptions {
	return GDOptions{
		Iteations: iterations,
		LearningRate: learningRate,
		MinStepSize: minStepSize,
	}
}

func (o GDOptions) GetType() string {
	return "gd"
}
