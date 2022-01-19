// this files defines a struct that defines some possible configurations
// for the linear regression model

package options

// represents the options for the Ordinari least square estimator
type OLSOptions struct {
	estimator string
}

func NewOLSOptions() OLSOptions {
	return OLSOptions{
		estimator: "ols",
	}
}

func (o OLSOptions) GetType() string {
	return "ols"
}
