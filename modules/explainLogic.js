// modules/explainLogic.js
const OpenAI = require("openai");
const dotenv = require("dotenv");

dotenv.config();

const openai = new OpenAI(process.env.OPENAI_API_KEY);

async function explainMessage(inputText, predictedClass, classScores) {
  const classList = ['fallacy of relevance', 'component fallacy', 'equivocation'];
  const classDescriptions = {
    'fallacy of relevance': 'occurs for arguments with premises that are logically irrelevant to the conclusion. Fallacy of Relevance subsumes the fine-grained classes Ad Hominem, Ad Populum, Appeal to Emotion, Fallacy of Extension, Intentional Fallacy. All of these fallacy classes present different means for using peripheral premises as support for claims. ',
    'component fallacy': 'Component fallacies involve errors in the logical structure of the argument itself. the premises seemingly provide ground for the conclusion but upon analysis prove to be insufficient and weak for supporting the claim made. ',
    'equivocation': 'Equivocation occurs when a word is used with different meanings in different parts of an argument, leading to a false conclusion.occurs when words or phrases are used in an equivocal way, thus causing ambiguity in the logic that connects the premise and the conclusion'
  };

  const explanationPrompt = `
  Analyze the following input: "${inputText}"
  It has been classified as "${predictedClass}".
  Here are the scores for each class:
  ${classList.map(className => `${className}: ${classScores[className].toFixed(2)}`).join("\n")}
  
  Explain why this input might fit into the "${predictedClass}" category based on the given scores and provide a brief description of each class.
  
  Descriptions:
  ${classList.map(className => `${className}: ${classDescriptions[className]}`).join("\n")}
  `;

  const completion = await openai.chat.completions.create({
    messages: [{ role: "system", content: "You are a helpful assistant." }, { role: "user", content: explanationPrompt }],
    model: "gpt-3.5-turbo",
  });

  return completion.choices[0].message.content;
}

module.exports = { explainMessage };
