package options

// Option defines possible configurations used to customize the behavior of mirai objects 
// such (but not limited to) models and estimators
type Option interface {
	GetType () string
}