let timeAll={0:[],1:[],2:[],3:[]}; // 1m data, 3m data, 1y data
let dataAll={0:[],1:[],2:[],3:[]}; // 1m data, 3m data, 1y data
let dataplot=[{x: [],y: [] },{x: [],y: [] },{x: [],y: [] },{x: [],y: [] }]; // 1m data, 3m data, 1y data
let predictplot=[{x: [],y: [] },{x: [],y: [] },{x: [],y: [] },{x: [],y: [] }]; // 1m data, 3m data, 1y data
let lastPredictScatter=[{x: [],y: [] },{x: [],y: [] },{x: [],y: [] },{x: [],y: [] }]; // 1m data, 3m data, 1y data
let window_size = 20;

function firstplot(){
  let urls = stock_urls('QQQ');
  console.log('url generated');
    fetch(urls[2])
    .then(response=>response.json())
    .then((response)=>{
        data=response;
        console.log(data)
        for (let i = 0; i < data.results.length; i++) {
          timeAll[2].push(data.results[i].t);
          dataAll[2].push([data.results[i].c,data.results[i].h,data.results[i].l,data.results[i].n,data.results[i].o,data.results[i].v,data.results[i].vw]);
          
          dataplot[2].x.push(data.results[i].t);
          dataplot[2].y.push(data.results[i].c);
        } 
        Plotly.newPlot("plot", [dataplot[2]]);  // need to put data into a array

        // predict
        //let model=tf.loadLayersModel('localstorage://QQQmodel') // tensorflow.js have a issue saving and loading RNN models and seems it is not fixed yet. https://github.com/tensorflow/tfjs/issues/1476
    })
}

function grabData(){
  let urls = stock_urls('QQQ');
  for (let k = 3; k < 4; k++){ // url for 1 year
    fetch(urls[k])
    .then(response=>response.json())
    .then((response)=>{
        data=response;
        console.log(data)
        for (let i = 0; i < data.results.length; i++) {
          timeAll[k].push(data.results[i].t);
          dataAll[k].push([data.results[i].c,data.results[i].h,data.results[i].l,data.results[i].n,data.results[i].o,data.results[i].v,data.results[i].vw]);
          
          dataplot[k].x.push(data.results[i].t);
          dataplot[k].y.push(data.results[i].c);
        }
        
        if(k==3){
          let logHtml = document.getElementById("train_log").innerHTML;
          logHtml = "data grabbed";
          document.getElementById("train_log").innerHTML = logHtml;
        }
      })
  }
}

function train(){
    //let rawDatForTrain=toRaw(this.dataAll[2]); // attn: the async feature let the following functions directly take empty array and go on (https://stackoverflow.com/questions/42260524/array-length-is-zero-but-the-array-has-elements-in-it)
    let rawDatForTrain=dataAll[3];
    console.log(rawDatForTrain);
    let trainDat=createInputData(rawDatForTrain,20);
    console.log(trainDat.vw)
    let modelPromise=new Promise(function(resolve, reject){ // resolve and reject are builtin functions for Promise, feed them with value and errors
      try {
        
        let modelDat=trainModel(trainDat.vw, trainDat.Y, window_size, 10, 0.001, 1);
        resolve(modelDat);
      } catch (error) {
        reject(error);
        // Expected output: ReferenceError: nonExistentFunction is not defined
        // (Note: the exact output may be browser-dependent)
      }
    } );
    
    modelPromise.then(
      (value)=>{
        predictt(value.model,window_size);
      },
      (error)=>{
        console.log(error);
      }
    )

}

function createInputData(input, window_size){
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

  for(let i=0;i<h_all.length-window_size;i++){
    h_in.push(h_all.slice(i,i+window_size));
    l_in.push(l_all.slice(i,i+window_size));
    n_in.push(n_all.slice(i,i+window_size));
    o_in.push(o_all.slice(i,i+window_size));
    v_in.push(v_all.slice(i,i+window_size));
    vw_in.push(vw_all.slice(i,i+window_size));
  }

  let Y_in=Yall.slice(window_size,Yall.length);
  let sma_in=ComputeSMA(vw_all, window_size);
  return {h:h_in,l:l_in,n:n_in,o:o_in,v:v_in,vw:vw_in, sma:sma_in, Y:Y_in, X: input};
}

function predictt(model,window_size){
  
  console.log(model);
  let rawDatForTrain3=dataAll[3];
  let trainDat3=createInputData(rawDatForTrain3,window_size);
  let predictDat3=makePredictions(trainDat3.vw, model);

  //predictplot[3].x=dataplot[3].x.slice(dataplot[3].x.length-dataplot[2].x.length,dataplot[3].x.length);
  predictplot[3].x=dataplot[2].x;
  predictplot[3].y=predictDat3.slice(dataplot[3].x.length-dataplot[2].x.length-window_size,dataplot[3].x.length);

  for(let i=0;i<dataplot[2].x.length;i++){
    dataplot[2].x[i]=formatDate(dataplot[2].x[i]);
  }
  for(let i=0;i<predictplot[3].x.length;i++){
    predictplot[3].x[i]=formatDate(predictplot[3].x[i]);
  }

  console.log(predictplot[3].x)
  console.log(predictplot[3].y)
  Plotly.newPlot("plot", [dataplot[2],predictplot[3]]);  // overlay original price and predicted price
  
  // save today's prediction
  //let today = new Date();
  //let dd = String(today.getDate()).padStart(2, '0');
  //let mm = String(today.getMonth() + 1).padStart(2, '0'); //January is 0!
  //let yyyy = today.getFullYear();
  //today = mm + '/' + dd + '/' + yyyy;

  //let predictionHistory=localStorage.getItem('predictionHistory')
  //if(predictionHistory==null){
  //  predictionHistory=[];
  //  predictionHistory.push([today,predictplot[2].y[predictplot[2].y.length-1]])
  //  localStorage.setItem('predictionHistory',predictionHistory)
  //}else{
  //  predictionHistory.push([today,predictplot[2].y[predictplot[2].y.length-1]])
  //  localStorage.setItem('predictionHistory',predictionHistory)
  //}
}

function ComputeSMA(data, window_size)
{
  let r_avgs = [], avg_prev = 0;
  for (let i = 0; i <= data.length - window_size; i++){
    let curr_avg = 0.00, t = i + window_size;
    for (let k = i; k < t && k <= data.length; k++){
      curr_avg += data / window_size;
    }
    r_avgs.push({ set: data.slice(i, i + window_size), avg: curr_avg });
    avg_prev = curr_avg;
  }
  return r_avgs;
}

function formatDate(time) {
  var d = new Date(time),
      month = '' + (d.getMonth() + 1),
      day = '' + d.getDate(),
      year = d.getFullYear();

  if (month.length < 2) month = '0' + month;
  if (day.length < 2) day = '0' + day;

  return [year, month, day].join('-');
}

function stock_urls(stock){

  let timeRanges=timeRangeCalculation()

  const url_1m = "https://api.polygon.io/v2/aggs/ticker/"+stock+"/range/1/day/"+timeRanges[0]+"?adjusted=true&sort=asc&apiKey=trh_nPl8ol7yoY_tp2hF2y5uCG4v3rrW";
  const url_3m = "https://api.polygon.io/v2/aggs/ticker/"+stock+"/range/1/day/"+timeRanges[1]+"?adjusted=true&sort=asc&apiKey=trh_nPl8ol7yoY_tp2hF2y5uCG4v3rrW";
  const url_1y = "https://api.polygon.io/v2/aggs/ticker/"+stock+"/range/1/day/"+timeRanges[2]+"?adjusted=true&sort=asc&apiKey=trh_nPl8ol7yoY_tp2hF2y5uCG4v3rrW";
  const url_3y = "https://api.polygon.io/v2/aggs/ticker/"+stock+"/range/1/day/"+timeRanges[3]+"?adjusted=true&sort=asc&apiKey=trh_nPl8ol7yoY_tp2hF2y5uCG4v3rrW";

  console.log(url_1y);

  let res=[url_1m,url_3m,url_1y,url_3y];
  return res;
}

function timeRangeCalculation(){
  // UTC today 
  let dateObj = new Date();
  let month = dateObj.getUTCMonth() + 1; //months from 1-12
  let day = dateObj.getUTCDate();
  let year = dateObj.getUTCFullYear();

  let adjMonth=month;
  if(adjMonth<10){
      adjMonth="0"+adjMonth;
  }
  let adjDay=day;
  if(adjDay<10){
      adjDay="0"+adjDay;
  }

  //1m
  let year1m=year;
  let month1m=month-1;
  if(month1m<=0){ // 1m earlier is previous year
      month1m=month1m+12;
      year1m=year1m-1;
  }
  let adjDay1m=daysInMonth(adjDay,month1m, year1m);
  if(month1m<10){
      month1m="0"+month1m;
  }
  let timeRange_1m=year1m+"-"+month1m+"-"+adjDay1m+"/"+year+"-"+adjMonth+"-"+adjDay;

  //3m
  let year3m=year;
  let month3m=month-3;
  if(month3m<=0){ // 1m earlier is previous year
      month3m=month3m+12;
      year3m=year3m-1;
  }
  let adjDay3m=daysInMonth(adjDay,month3m, year3m);
  if(month3m<10){
      month3m="0"+month3m;
  }
  let timeRange_3m=year3m+"-"+month3m+"-"+adjDay3m+"/"+year+"-"+adjMonth+"-"+adjDay;

  //1y
  let year1y=year-1;
  let month1y=month;
  if(month1y<10){
      month1y="0"+month1y;
  }
  let adjDay1y=daysInMonth(adjDay,month1y, year1y);
  let timeRange_1y=year1y+"-"+month1y+"-"+adjDay1y+"/"+year+"-"+adjMonth+"-"+adjDay;

  //3y
  let year3y=year-3;
  let month3y=month;
  if(month3y<10){
      month3y="0"+month3y;
  }
  let adjDay3y=daysInMonth(adjDay,month3y, year3y);
  let timeRange_3y=year3y+"-"+month3y+"-"+adjDay3y+"/"+year+"-"+adjMonth+"-"+adjDay;

  let timeRanges=[timeRange_1m,timeRange_3m,timeRange_1y,timeRange_3y]
  return timeRanges;
}

function daysInMonth(d, m, y) { // https://stackoverflow.com/questions/1433030/validate-number-of-days-in-a-given-month, m: 1-12
  switch (m-1) {
      case 1 : // Feb
          let actualFebEndDay= (y % 4 == 0 && y % 100) || y % 400 == 0 ? 29 : 28;
          if(d>actualFebEndDay){
              return actualFebEndDay;
          }else{
              return d;
          }
      case 3 : case 5 : case 8 : case 10 : // 30 days months
          if (d>30){
              return 30;
          }else{
              return d;
          }  
      default : // 31 days months
          return d
  }
}
