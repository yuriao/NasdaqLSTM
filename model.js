async function trainModel(X, Y, window_size, n_epochs, learning_rate, n_layers, callback){

  const batch_size = 32;

  // input dense layer
  const input_layer_shape = window_size;
  const input_layer_neurons = 64;

  // LSTM
  const rnn_input_layer_features = 16;
  const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;
  const rnn_input_shape = [rnn_input_layer_features, rnn_input_layer_timesteps]; // the shape have to match input layer's shape
  const rnn_output_neurons = 16; // number of neurons per LSTM's cell

  // output dense layer
  const output_layer_shape = rnn_output_neurons; // dense layer input size is same as LSTM cell
  const output_layer_neurons = 1; // return 1 value

  // ## old method
  // const xs = tf.tensor2d(X, [X.length, X[0].length])//.div(tf.scalar(10));
  // const ys = tf.tensor2d(Y, [Y.length, 1]).reshape([Y.length, 1])//.div(tf.scalar(10));

  // ## new: load data into tensor and normalize data
  const inputTensor = tf.tensor2d(X, [X.length, X[0].length])
  const labelTensor = tf.tensor2d(Y, [Y.length, 1]).reshape([Y.length, 1])

  console.log(inputTensor)

  const [xs, inputMax, inputMin] = normalizeTensorFit(inputTensor)
  const [ys, labelMax, labelMin] = normalizeTensorFit(labelTensor)

  // ## define model

  const model = tf.sequential();

  model.add(tf.layers.dense({units: input_layer_neurons, inputShape: [input_layer_shape]}));
  model.add(tf.layers.reshape({targetShape: rnn_input_shape}));

  let lstm_cells = [];
  for (let index = 0; index < n_layers; index++) {
       lstm_cells.push(tf.layers.lstmCell({units: rnn_output_neurons}));
  }

  model.add(tf.layers.rnn({
    cell: lstm_cells,
    inputShape: rnn_input_shape,
    returnSequences: false
  }));

  model.add(tf.layers.dense({units: output_layer_neurons, inputShape: [output_layer_shape]}));

  model.compile({
    optimizer: tf.train.adam(learning_rate),
    loss: 'meanSquaredError'
  });

  // ## fit model
  
  const hist = await model.fit(xs, ys,{batchSize: batch_size, epochs: n_epochs, callbacks: {
    onEpochEnd: async (epoch, log) => {
      console.log("epoch: "+epoch+" loss: "+log.loss);
      let logHtml = document.getElementById("train_log").innerHTML;
      logHtml = "<div>Epoch: " + (epoch + 1) + " / "+ n_epochs  +", loss: " + log.loss +"</div>" + logHtml;
      //epoch_loss.push(log.loss);
      document.getElementById("train_log").innerHTML = logHtml;
    }
  }});

  // return { model: model, stats: hist };
  //const saveResult = await model.save('localstorage://QQQmodel');
  //localStorage.setItem('QQQmodel', model);
  //localStorage.setItem('normalizeFactor', JSON.stringify({inputMax:inputMax, inputMin:inputMin, labelMax:labelMax, labelMin:labelMin, window_size:window_size}));
  //const loadResult = await tf.loadLayersModel('localstorage://QQQmodel');
  //console.log(loadResult)
  //console.log(saveResult)
  console.log('training fin')
  console.log(model)
  
  return { model: model, stats: hist, normalize: {inputMax:inputMax, inputMin:inputMin, labelMax:labelMax, labelMin:labelMin, window_size:window_size} };
}

function makePredictions(X, model)
{
    // const predictedResults = model.predict(tf.tensor2d(X, [X.length, X[0].length]).div(tf.scalar(10))).mul(10); // old method
    console.log(X)
    X = tf.tensor2d(X, [X.length, X[0].length]);
    //const normalizedInput = normalizeTensor(X, dict_normalize["inputMax"], dict_normalize["inputMin"]);
    const [normalizedInput, maxx, minn] = normalizeTensorFit(X);
    const model_out = model.predict(normalizedInput);
    const predictedResults = unNormalizeTensor(model_out, maxx, minn);

    return Array.from(predictedResults.dataSync());
}

function normalizeTensorFit(tensor) {
  const maxval = tensor.max();
  const minval = tensor.min();
  const normalizedTensor = normalizeTensor(tensor, maxval, minval);
  return [normalizedTensor, maxval, minval];
}

function normalizeTensor(tensor, maxval, minval) {
  const normalizedTensor = tensor.sub(minval).div(maxval.sub(minval));
  return normalizedTensor;
}

function unNormalizeTensor(tensor, maxval, minval) {
  const unNormTensor = tensor.mul(maxval.sub(minval)).add(minval);
  return unNormTensor;
}

function fitCallBack(epoch, log) {
  //let logHtml = document.getElementById("div_traininglog").innerHTML;
  //logHtml = "<div>Epoch: " + (epoch + 1) + " (of "+ n_epochs +")" +
    //", loss: " + log.loss +
    // ", difference: " + (epoch_loss[epoch_loss.length-1] - log.loss) +
    //"</div>" + logHtml;

  //epoch_loss.push(log.loss);

  //document.getElementById("div_traininglog").innerHTML = logHtml;
  //document.getElementById("div_training_progressbar").style.width = Math.ceil(((epoch + 1) * (100 / n_epochs))).toString() + "%";
  //document.getElementById("div_training_progressbar").innerHTML = Math.ceil(((epoch + 1) * (100 / n_epochs))).toString() + "%";

  //let graph_plot = document.getElementById('div_linegraph_trainloss');
  //Plotly.newPlot( graph_plot, [{x: Array.from({length: epoch_loss.length}, (v, k) => k+1), y: epoch_loss, name: "Loss" }], { margin: { t: 0 } } );
};