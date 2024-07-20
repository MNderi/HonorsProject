const { InferenceSession, Tensor } = require('onnxruntime-node');
const path = require('path');
const axios = require('axios');


// Load the model
const modelPath = path.join(__dirname, 'modelgen.onnx'); 

// Function to load the model
let session;
async function loadModel() {
  if (!session) {
    const options = { providers: ['WebAssembly'] }; // Specify provider (optional)
    session = await InferenceSession.create(modelPath, options);
    console.log('Model loaded successfully');
  }
  return session;
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);  // for numerical stability
    const exps = logits.map(logit => Math.exp(logit - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);
    return exps.map(exp => exp / sumExps);
}


// Function to preprocess the input and run inference
async function binaryPredict(inputText, isBinary = false) {
  try {
    // Ensure inputText is a string
    if (typeof inputText !== 'string') {
      inputText = String(inputText);
    }

    // Remove full stops from input text
    inputText = inputText.replace(/\./g, '');

    // Make HTTP POST request to tokenizing endpoint
    const tokenizingEndpoint = 'http://localhost:5000/tokenize'; // Adjust URL as needed
    const response = await axios.post(tokenizingEndpoint, { text: inputText });

    // Ensure response is valid
    if (!response || !response.data || !response.data.input_ids || !response.data.attention_mask || !response.data.token_type_ids) {
      throw new Error('Invalid response from tokenizing endpoint');
    }

    // Extract tokenized inputs from response
    const { input_ids, attention_mask, token_type_ids } = response.data;

    // Log tokenized inputs for debugging
    console.log('Tokenized Inputs:', { input_ids, attention_mask, token_type_ids });

    // Load the model (or reuse existing session)
    let session = await loadModel();

    // Create tensors directly from arrays
    const inputIdsTensor = new Tensor('int64', input_ids, [1, input_ids.length]);
    const attentionMaskTensor = new Tensor('int64', attention_mask, [1, attention_mask.length]);
    const tokenTypeIdsTensor = new Tensor('int64', token_type_ids, [1, token_type_ids.length]);

    // Perform inference
    const outputs = await session.run({ input_ids: inputIdsTensor, attention_mask: attentionMaskTensor, token_type_ids: tokenTypeIdsTensor });

    // Ensure there is at least one output
    if (!outputs || !outputs.output_1) { // Check for the correct output name
      throw new Error('Model did not return any outputs');
    }
    console.log(outputs);
    const outputTensor = outputs.output_1.data; // Adjust the output tensor name if necessary

    // For binary classification, apply sigmoid to get probabilities or logits
    if (isBinary) {
      const logits = outputTensor[0]; // Assuming batch size 1
      const probability = 1 / (1 + Math.exp(-logits)); // Sigmoid function for binary classification
      
      // Determine the predicted class based on probability threshold (e.g., 0.5)
      const predictedClass = probability >= 0.5 ? 'positive' : 'negative';
      console.log('Predicted Class:', predictedClass);
      return { predictedClass, confidence: probability };
    } else {
      // For multi-class classification, apply softmax
      const logits = outputTensor;
      const expLogits = logits.map(Math.exp);
      const sumExpLogits = expLogits.reduce((a, b) => a + b, 0);
      const probabilities = expLogits.map(expLogit => expLogit / sumExpLogits);
      const pooledOutput = outputTensor;

        // Apply softmax to pooled output to get classification probabilities
        const probabil = softmax(pooledOutput);

        // Get the probabilities as a JavaScript array
      console.log(probabil)
      // Determine the predicted class based on highest probability
      const maxProbability = Math.max(...probabilities);
      const predictedClassIndex = probabilities.indexOf(maxProbability);
      console.log('Predicted Class Index:', predictedClassIndex);
      return { predictedClassIndex, confidence: maxProbability };
    }
  } catch (error) {
    console.error('Error during tokenization or inference:', error);
    throw error; // Rethrow the error to propagate it upwards
  }
}

module.exports = { binaryPredict };
