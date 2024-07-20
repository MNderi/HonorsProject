const express= require('express');
const app=express();
const path = require('path');
const cors = require('cors');
app.use(cors());

app.use(express.json());
port=3000

const {coarsePredict }=require('./modules/coarsepredictor')
const {finePredict }=require('./modules/finepredictor')
const{binaryPredict}=require('./modules/bertDetection')
const {analyzeMessage}=require('./modules/criticalquestions')
const { explainMessage } = require('./modules/explainLogic.js');
const {explainFineMessage} =require('./modules/explainFineLogic.js');



// Endpoint to make predictions
app.post('/coarsepredict', async (req, res) => {
    console.log('Request body:', req.body);
    try {
        if (!req.body || !req.body.input) {
            throw new Error('Invalid request body. Missing input data.');
        }

        const input = req.body.input;
        const output = await coarsePredict(input);
        res.json({ prediction: output });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Failed to make prediction' });
    }
});


app.post('/finepredict', async (req, res) => {
    const { input } = req.body;

    try {
        // Perform prediction using the model
        const output = await finePredict(input);

        // Send the prediction as JSON response
        res.json({ prediction: output });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Failed to make prediction' });
    }
});

app.post('/detectpredict', async (req, res) => {
  const { input } = req.body;

  try {
      // Perform prediction using the model
      const output = await binaryPredict(input);

      // Send the prediction as JSON response
      res.json({ prediction: output });
  } catch (error) {
      console.error('Prediction error:', error);
      res.status(500).json({ error: 'Failed to make prediction' });
  }
});

app.post('/criticalquestions', async (req, res) => {
  const { input } = req.body;

  if (!input) {
    return res.status(400).json({ error: "Input is required" });
  }

  try {
    const response = await analyzeMessage(input);
    res.json(response);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.post('/explaincoarse', async (req, res) => {
  const { input } = req.body;

  if (!input) {
    return res.status(400).json({ error: "Input is required" });
  }

  try {
    const { predictedClass, classScores } = await coarsePredict(input);
    const explanation = await explainMessage(input, predictedClass, classScores);
    res.json({ predictedClass, classScores, explanation });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});
app.post('/explainfine', async (req, res) => {
  const { input } = req.body;

  if (!input) {
    return res.status(400).json({ error: "Input is required" });
  }

  try {
    const { predictedClass, classScores } = await finePredict(input);
    const explanation = await explainFineMessage(input, predictedClass, classScores);
    res.json({ predictedClass, classScores, explanation });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get('/', (req,res)=> res.send("Hello, Welcome!") )
app.listen(port, ()=> console.log(`VerityVault app listening on port ${port}!`))