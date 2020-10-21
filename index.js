require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logisic-regression");
const plot = require("node-remote-plot");
const _ = require('lodash');
const mnist = require('mnist-data');

const mnistData = mnist.training(0, 10);

const features = mnistData.images.values.map(image => _.flatMap(image));

//The fill() method changes all elements in an array to a static value
//, from a start index (default 0) to an end index (default array.length). It returns the modified array.
const encodedLabels = mnistData.labels.values.map(label => {  //replacing 0 with 1 value if the value of a number from labels matches the index of column
  const row = new Array(10).fill(0);
  row[label] = 1
  return row;
});

console.log(encodedLabels);







