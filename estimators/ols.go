package estimators

import "github.com/marti700/veritas/linearalgebra"

// Estimates the Linear regression coeffients using the least square method
func OLS(data, target linearalgebra.Matrix) linearalgebra.Matrix {
	// (X.T*X)^(-1)*X^(-1)*y <--- closed form solution
	return data.T().Mult(data).Inv().Mult(data.T()).Mult(target)
}
