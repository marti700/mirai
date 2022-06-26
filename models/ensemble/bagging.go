package ensamble

import (
	"sync"

	model "github.com/marti700/mirai/models"
	"github.com/marti700/mirai/utils"
	"github.com/marti700/veritas/linearalgebra"
)

type BaggingClassifier struct {
	Model         model.Model
	N_models      int
	trainedModels []model.Model
}

// trains this Bagging classifier
func (b *BaggingClassifier) Train(data, target linearalgebra.Matrix) {
	b.trainedModels = make([]model.Model, b.N_models)
	// resp := make(chan model.Model, b.N_models)

	var wg sync.WaitGroup
	wg.Add(b.N_models)

	for i := 0; i < b.N_models; i++ {
		tData := linearalgebra.Insert(target, data, data.Col)
		in, _ := utils.Bootstrap(tData)
		//get features
		feat := linearalgebra.Slice(in, 0, tData.Col-1, "y")
		// get target
		tar := in.GetCol(tData.Col - 1)
		b.trainedModels[i] = b.Model.Clone()
		go func(m model.Model) {
			defer wg.Done()
			m.Train(feat, tar)
		}(b.trainedModels[i])
	}

	wg.Wait()
}

// Predicts by averaging each model prediction
func (b *BaggingClassifier) Predict(data linearalgebra.Matrix) linearalgebra.Matrix {
	p_sum := b.trainedModels[0].Predict(data)

	for i := 1; i < len(b.trainedModels); i++ {
		p_sum.Sum(b.trainedModels[i].Predict(data))
	}

	return p_sum.Map(func(x float64) float64 {
		return x/float64(b.N_models)
	})
}
