package estimators

import "github.com/marti700/veritas/linearalgebra"

// Estimates the Linear regression coeffients using the least square method
// (X.T*X)^(-1)*X*y <--- closed form solution
func OLS(data, target linearalgebra.Matrix) linearalgebra.Matrix {
	dataT := data.T()
	m,_ := dataT.Mult(data)
	m1,_ := m.Inv().Mult(dataT)
	m2,_ :=m1.Mult(target)
	return m2
}