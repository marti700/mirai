package ensamble

import (
	"sync"

	model "github.com/marti700/mirai/models"
	"github.com/marti700/mirai/utils"
	"github.com/marti700/veritas/linearalgebra"
)

// BaggingRegressor and BaggingClassifier struct types
// The model field is the model that will be use to run the bagging
// the N_models field specifies the number of models that ara going to be trained in this bag
// trainedModels is an internal field used to save the trained model of the bag
type BaggingRegressor struct {
	Model         model.Model
	N_models      int
	trainedModels []model.Model
}

type BaggingClassifier struct {
	Model         model.Model
	N_models      int
	trainedModels []model.Model
}

// trains this Bagging Regressor
func (b *BaggingRegressor) Train(data, target linearalgebra.Matrix) {
	b.trainedModels = make([]model.Model, b.N_models)
	trainModels(b.trainedModels, b.Model, data, target)
}

// trains this Bagging classifier
func (b *BaggingClassifier) Train(data, target linearalgebra.Matrix) {
	b.trainedModels = make([]model.Model, b.N_models)
	trainModels(b.trainedModels, b.Model, data, target)
}

// Bagging regressor predict method.
// Predicts by averaging each model prediction
func (b *BaggingRegressor) Predict(data linearalgebra.Matrix) linearalgebra.Matrix {
	p_sum := b.trainedModels[0].Predict(data)

	for i := 1; i < len(b.trainedModels); i++ {
		p_sum = p_sum.Sum(b.trainedModels[i].Predict(data))
	}

	return p_sum.Map(func(x float64) float64 {
		return x / float64(b.N_models)
	})
}

// Bagging classifier predict method.
// classifies by taking the most voted class of all models in the ensemble
func (b *BaggingClassifier) Predict(data linearalgebra.Matrix) linearalgebra.Matrix {
	p_agg := b.trainedModels[0].Predict(data)
	for i := 1; i < b.N_models; i++ {
		p_agg = linearalgebra.Insert(b.trainedModels[i].Predict(data), p_agg, 0)
	}

	preditions := make([]float64, p_agg.Row)
	mostVoted := func(r linearalgebra.Matrix) float64 {
		frequencies := make(map[float64]int)

		// generates a frecuency table
		for i := 0; i < r.Col; i++ {
			_, present := frequencies[r.Get(0, i)]
			if !present {
				frequencies[r.Get(0, i)] = 1
			} else {
				frequencies[r.Get(0, i)]++
			}
		}

		// select the class with the most votes
		var winner float64
		var votes int
		for key, value := range frequencies {
			if value > votes {
				winner = key
				votes = value
			}
		}
		return winner
	}

	// creates the vector with the final predictions
	for i := 0; i < p_agg.Row; i++ {
		preditions[i] = mostVoted(p_agg.GetRow(i))
	}

	return p_agg
}

//NOT IMPLEMENTED YET
func (b *BaggingRegressor) Clone() model.Model {
	panic("Not implemented Yet")
}

//NOT IMPLEMENTED YET
func (b *BaggingClassifier) Clone() model.Model {
	panic("Not implemented Yet")
}

func trainModels(models []model.Model, mod model.Model, data, target linearalgebra.Matrix) {
	var wg sync.WaitGroup
	wg.Add(len(models))

	for i := 0; i < len(models); i++ {
		tData := linearalgebra.Insert(target, data, data.Col)
		in, _ := utils.Bootstrap(tData)
		//get features
		feat := linearalgebra.Slice(in, 0, tData.Col-1, "y")
		// get target
		tar := in.GetCol(tData.Col - 1)
		models[i] = mod.Clone()
		go func(m model.Model) {
			defer wg.Done()
			m.Train(feat, tar)
		}(models[i])
	}

	wg.Wait()
}
