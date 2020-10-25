require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logisic-regression");
const plot = require("node-remote-plot");
const _ = require("lodash");
const mnist = require("mnist-data");


function loadData(){
const mnistData = mnist.training(0, 60000);

const features = mnistData.images.values.map((image) => _.flatMap(image));

//The fill() method changes all elements in an array to a static value
//, from a start index (default 0) to an end index (default array.length). It returns the modified array.
const encodedLabels = mnistData.labels.values.map((label) => {
  //replacing 0 with 1 value if the value of a number from labels matches the index of column
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

return {features, labels: encodedLabels};
}

//relasing reference, javascript garbage collector
const {features, labels} = loadData();

const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 80,
  batchSize: 500,
});

regression.train();

const testMnistData = mnist.testing(0, 10000);

const testFeatures = testMnistData.images.values.map((image) =>
  _.flatMap(image)
);
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  //replacing 0 with 1 value if the value of a number from labels matches the index of column
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});


const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log('Accuracy', accuracy);

plot({
  x: regression.costHistory.reverse(), // newest values are in the front of the array
  xLabel: "Iteration", //for each interation
  yLabel: "Cost", //how wrong we were
});


