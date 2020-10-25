const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];

    //if we dont provide learning rate in options the default rate will be 0.1
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 },
      options
    );

    //initial guesses, M and B previously
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  gradientDescent(features, labels) {
    //mathmul matrix multiplication
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose() // RESHAPING TENSOR so we can match the shape of differences
      .matMul(differences)
      .div(features.shape[0]);

    return this.weights.sub(slopes.mul(this.options.learningRate)); //this.m = this.m - mSlope * this.options.learningRate;
  }

  train() {
    //batch gradient descent        rows/batchsize == how many times we iterate
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    ); // in case we have soem leftover odd number

    for (let index = 0; index < this.options.iterations; index++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const batchSize = this.options.batchSize;
  //tf tidy, garbage collector saves memory
        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [batchSize, -1]
          );
          const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
  
          return this.gradientDescent(featureSlice, labelSlice);
        });
      }
      this.recordCost();
      this.updateLearningRate();
    }
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures); //probabilities of being the '1' label, 
    testLabels = tf.tensor(testLabels).argMax(1);
    const incorrect = predictions.notEqual(testLabels).sum().get();
    //number of predictions shape[0] a.k.a row
    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1); //horizontal axis
  }

  processFeatures(features) {
    //generates an extra column so we can use the matrix multiplication
    //ones([shape]) shape = features row, one column,    1 for concatenation axis
    features = tf.tensor(features);
    //we have to reapply mean and variance for our test set if it is second time
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
      //we use our helper function for the first case
    } else {
      features = this.standardize(features);
    }
    //column of ones needs to come after standardization so we do not change the ones values
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    const filler = variance.cast('bool').logicalNot().cast("float32");
    //in case the variance is 0 we will use this playaround to check if it is 0, turn it into 1 and replace all 0 features with ones
    this.mean = mean;
    this.variance = variance.add(filler);

    return features.sub(mean).div(this.variance.pow(0.5));
  }
  //vectorized solution
  recordCost() {
    //tf tidy, garbage collector saves memory
    const cost = tf.tidy(() => {
      const guesses = this.features.
      matMul(this.weights)
      .sigmoid();
  
      const termOne = this.labels
      .transpose()
      .matMul(guesses.add(1e-7).log());
  
      const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(guesses.mul(-1).add(1).add(1e-7).log()) // 1x10 ^ -7 = 0.00000001 MAKE SURE NEVER TAKE A LOG THAT IS 0  so we dont end up with infinite value
      //this arbitrary change is here just to prevent that infinite bullshit and it has only small effect on the results
      return termOne.add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .get(0, 0);
    });
    //vectorized cost entropy formula instead of mean squaared error

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }
    //if the value of mse goes up we are overshooting and getting incorrect values we need to decrease our learning rate
    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate = this.options.learningRate / 2;
    } else {
      this.options.learningRate *= 1.05; //increase the learning by 5% if the MSE error goes down and we are getting closer to the optimal value
    }
  }
}

module.exports = LogisticRegression;
