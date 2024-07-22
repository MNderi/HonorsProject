const { InferenceSession, Tensor } = require('onnxruntime-node');
const path = require('path');
const axios = require('axios'); // Import axios for making HTTP requests

// Load the model once at startup
const modelPath = path.join(__dirname, 'model.onnx'); // Constructing the absolute path
let session;

async function initializeModel() {
  try {
    const options = { providers: ['WebAssembly'] }; // Specify provider (optional)
    session = await InferenceSession.create(modelPath, options);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

// Call initializeModel once during startup
initializeModel();

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sumExps = exps.reduce((sum, exp) => sum + exp, 0);
  return exps.map(exp => exp / sumExps);
}

// Function to preprocess the input and run inference
async function coarsePredict(inputText) {
  try {
    // Ensure inputText is a string
    if (typeof inputText !== 'string') {
      inputText = String(inputText);
    }

    // Remove full stops from input text
    inputText = inputText.replace(/\./g, '');

    // Make HTTP POST request to tokenizing endpoint
    const tokenizingEndpoint = 'http://54.87.214.222:5000/tokenize'; // Adjust URL as needed
    const response = await axios.post(tokenizingEndpoint, { text: inputText });

    // Ensure response is valid
    if (!response || !response.data || !response.data.input_ids || !response.data.attention_mask || !response.data.token_type_ids) {
      throw new Error('Invalid response from tokenizing endpoint');
    }

    // Extract tokenized inputs from response
    const { input_ids, attention_mask, token_type_ids } = response.data;

    // Log tokenized inputs for debugging
    console.log('Tokenized Inputs:', { input_ids, attention_mask, token_type_ids });

    // Create tensors directly from arrays
    const inputIdsTensor = new Tensor('int64', input_ids, [1, input_ids.length]);
    const attentionMaskTensor = new Tensor('int64', attention_mask, [1, attention_mask.length]);
    const tokenTypeIdsTensor = new Tensor('int64', token_type_ids, [1, token_type_ids.length]);

    // Perform inference
    const outputs = await session.run({ input_ids: inputIdsTensor, attention_mask: attentionMaskTensor, token_type_ids: tokenTypeIdsTensor });

    // Ensure there is at least one output
    if (!outputs || outputs.length === 0) {
      throw new Error('Model did not return any outputs');
    }

    // Access the first output tensor (assuming a single output tensor in this example)
    const outputTensor = Array.from(outputs.output.cpuData);
    console.log(outputs);
    const probabilities = softmax(outputTensor);

    console.log('Probabilities:', probabilities);

    const classList = ['fallacy of relevance', 'component fallacy', 'equivocation'];

    // Find the index of the maximum value (predicted class)
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));

    // Determine the predicted class
    const predictedClass = classList[maxIndex];

    // Prepare class scores (probabilities for each class)
    const classScores = classList.reduce((acc, className, index) => {
      acc[className] = probabilities[index];
      return acc;
    }, {});

    return { predictedClass, classScores };
  } catch (error) {
    console.error('Error during tokenization or inference:', error);
    throw error; // Rethrow the error to propagate it upwards
  }
}

module.exports = { coarsePredict };
