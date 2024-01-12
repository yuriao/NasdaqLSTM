let timeAll={0:[],1:[],2:[],3:[]}; // 1m data, 3m data, 1y data
let dataAll={0:[],1:[],2:[],3:[]}; // 1m data, 3m data, 1y data
let dataplot=[{x: [],y: [], name: 'original 1 month' },{x: [],y: [], name: 'original 3 month' },{x: [],y: [], name: 'original 1 year' },{x: [],y: [] }]; // 1m data, 3m data, 1y data
let predictplot=[{x: [],y: [] },{x: [],y: [] },{x: [],y: [] },{x: [],y: [] }]; // 1m data, 3m data, 1y data
let lastPredictScatter=[{x: [],y: [] },{x: [],y: [] },{x: [],y: [] },{x: [],y: [] }]; // 1m data, 3m data, 1y data
let oriDateTime=[]
let window_size = 20;
let LSTM_outSize=1;
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

        oriDateTime=JSON.parse(JSON.stringify(dataplot[2].x));
        for(let i=0;i<dataplot[2].x.length;i++){
          dataplot[2].x[i]=formatDate(dataplot[2].x[i]);
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

        for(let i=0;i<predictplot[k].x.length;i++){
          predictplot[k].x[i]=formatDate(predictplot[k].x[i]);
        }
        
      })
  }
}

function train(){
    //let rawDatForTrain=toRaw(this.dataAll[2]); // attn: the async feature let the following functions directly take empty array and go on (https://stackoverflow.com/questions/42260524/array-length-is-zero-but-the-array-has-elements-in-it)
    

    // Get the value of the input element and convert it to an integer\
    var epochInput = document.getElementById('epochInput');
    let epochNumber = parseInt(epochInput.value, 10);

    let rawDatForTrain=dataAll[3];
    let trainDat=createInputData(rawDatForTrain,window_size,LSTM_outSize);
    let modelPromise=new Promise(function(resolve, reject){ // resolve and reject are builtin functions for Promise, feed them with value and errors
      try {
        
        let modelDat=trainModel(trainDat.vw, trainDat.Y, window_size, epochNumber, 0.001, 1,LSTM_outSize);
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


function predictt(model,window_size){
  
  let rawDatForTrain3=dataAll[3];
  let trainDat3=createInputData(rawDatForTrain3,window_size,LSTM_outSize);
  let vw=trainDat3.vw
  let predictDat3=makePredictions(vw, model);
  
  // next 30 days
  newDateTime=oriDateTime
  predictDat4=predictDat3;
  vw.shift();
  vw.push(predictDat3.slice(-20))
  for(let k = 0; k < 81; k++){
    p=makePredictions(vw, model);
    predictDat4.push(p[p.length-1])
    vw.shift();
    vw.push(p.slice(-20))
    newDateTime.push(newDateTime[newDateTime.length-1]+86400000)
  }

  
  for(let i=0;i<newDateTime.length;i++){
    newDateTime[i]=formatDate(newDateTime[i]-30*86400000);
  }
  //predictplot[3].x=dataplot[3].x.slice(dataplot[3].x.length-dataplot[2].x.length,dataplot[3].x.length);
  predictplot[3].x=newDateTime;
  predictplot[3].y=predictDat4.slice(predictDat4.length-predictplot[3].x.length-window_size,predictDat4.length);
  predictplot[3].name='predicted'

  Plotly.newPlot("plot", [dataplot[2],predictplot[3]]);  // overlay original price and predicted price
  trace0.name = 'Updated First Trace'
  trace1.name = 'Updated Second Trace'

  fig.data = [trace0, trace1]
  
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
