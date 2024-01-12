async function trainModel(X, Y, window_size, n_epochs, learning_rate, n_layers,outSize, callback){

  const batch_size = 32;

  // input dense layer params
  const input_layer_shape = window_size;
  const input_layer_neurons = 64;

  // LSTM params
  const rnn_input_layer_features = 16;
  const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;
  const rnn_input_shape = [rnn_input_layer_features, rnn_input_layer_timesteps]; // the shape have to match input layer's shape
  const rnn_output_neurons = 16; // number of neurons per LSTM's cell

  // output dense layer params
  const output_layer_shape = rnn_output_neurons; // dense layer input size is same as LSTM cell
  const output_layer_neurons = outSize; // return 5 value


  // load data into tensor and normalize data
  const inputTensor = tf.tensor2d(X, [X.length, X[0].length])
  const labelTensor = tf.tensor2d(Y, [Y.length, Y[0].length])

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
  
  const hist = await model.fit(xs, ys,{batchSize: batch_size, epochs: n_epochs+1, callbacks: {
    onEpochEnd: async (epoch, log) => {
      console.log("epoch: "+epoch+" loss: "+log.loss);
      let logHtml = document.getElementById("train_log").innerHTML;
      logHtml = "<div>Epoch: " + (epoch + 1) + " / "+ n_epochs  +", loss: " + log.loss +"</div>" + logHtml;
      document.getElementById("train_log").innerHTML = logHtml;
      if(epoch + 1==n_epochs){
        let logHtml = document.getElementById("train_log").innerHTML;
        logHtml = "<div>training finished, predicting...</div>" + logHtml;
        document.getElementById("train_log").innerHTML = logHtml;
      }
    }
  }});

  console.log('training fin, predicting...')
  document.getElementById("train_log").innerHTML = '';
  
  return { model: model, stats: hist, normalize: {inputMax:inputMax, inputMin:inputMin, labelMax:labelMax, labelMin:labelMin, window_size:window_size} };
}

function makePredictions(X, model)
{
    // const predictedResults = model.predict(tf.tensor2d(X, [X.length, X[0].length]).div(tf.scalar(10))).mul(10); // old method
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

function createInputData(input, window_size, output_size){
  // use xx day data to predict the next day's closing price
  // input: matrix including all fields, with first field as closing price
  let h_all=input.map(function(value,index) { return value[1];});
  let l_all=input.map(function(value,index) { return value[2];});
  let n_all=input.map(function(value,index) { return value[3];});
  let o_all=input.map(function(value,index) { return value[4];});
  let v_all=input.map(function(value,index) { return value[5];});
  let vw_all=input.map(function(value,index) { return value[6];});
  
  let Yall=input.map(function(value,index) { return value[0];});

  h_in=[];
  l_in=[];
  n_in=[];
  o_in=[];
  v_in=[];
  vw_in=[];

  for(let i=0;i<h_all.length-window_size-output_size+1;i++){
    h_in.push(h_all.slice(i,i+window_size));
    l_in.push(l_all.slice(i,i+window_size));
    n_in.push(n_all.slice(i,i+window_size));
    o_in.push(o_all.slice(i,i+window_size));
    v_in.push(v_all.slice(i,i+window_size));
    vw_in.push(vw_all.slice(i,i+window_size));
  }

  let Y_in=[];
  for(let i=window_size;i<h_all.length-output_size+1;i++){
    Y_in.push(Yall.slice(i,i+output_size));
  }
  
  //let sma_in=ComputeSMA(vw_all, window_size);
  return {h:h_in,l:l_in,n:n_in,o:o_in,v:v_in,vw:vw_in, Y:Y_in, X: input};
}