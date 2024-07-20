// modules/explainFineLogic.js
const OpenAI = require("openai");
const dotenv = require("dotenv");

dotenv.config();

const openai = new OpenAI(process.env.OPENAI_API_KEY);

async function explainFineMessage(inputText, predictedClass, classScores) {
  const classList = [
    'faulty generalization', 'false causality', 'circular reasoning', 
    'ad populum', 'ad hominem', 'fallacy of logic', 'appeal to emotion', 
    'false dilemma', 'equivocation', 'fallacy of extension', 
    'fallacy of relevance', 'fallacy of credibility', 'intentional'
  ];

  const classDescriptions = {
    'faulty generalization': 'A fallacy where a conclusion is drawn from a sample that is too small or not representative.',
    'false causality': 'A fallacy where a cause is incorrectly identified for an effect.',
    'circular reasoning': 'A fallacy where the conclusion is included in the premise.',
    'ad populum': 'A fallacy that concludes a proposition is true because many or most people believe it.',
    'ad hominem': 'A fallacy that attacks the person making the argument rather than the argument itself.',
    'fallacy of logic': 'A general category for errors in reasoning.',
    'appeal to emotion': 'A fallacy where an argument is made by manipulating emotions rather than using valid reasoning.',
    'false dilemma': 'A fallacy that presents only two options when there are actually more.',
    'equivocation': 'A fallacy where a word is used with different meanings in different parts of an argument.',
    'fallacy of extension': 'A fallacy that extends an argument to an unrelated or extreme point.',
    'fallacy of relevance': 'A fallacy where the premises are not logically relevant to the conclusion.',
    'fallacy of credibility': 'A fallacy that occurs when the credibility of the source is attacked rather than the argument.',
    'intentional': 'A fallacy where intentions are presumed without evidence.'
  };

  const positiveClasses = Object.entries(classScores)
    .filter(([_, score]) => score > 0)
    .map(([className, score]) => `${className}: ${score.toFixed(2)}`);

  const explanationPrompt = `
  Analyze the following input: "${inputText}"
  It has been classified as "${predictedClass}" with the highest positive score.
  Here are the scores for each class with a positive score:
  ${positiveClasses.join("\n")}
  
  Explain why this input might fit into the "${predictedClass}" category and provide a brief description of each class with a positive score, relating each explanation to the input text.

  Descriptions:
  ${positiveClasses.map(className => `${className.split(":")[0]}: ${classDescriptions[className.split(":")[0]]}`).join("\n")}
  `;

  const completion = await openai.chat.completions.create({
    messages: [{ role: "system", content: "You are a helpful assistant." }, { role: "user", content: explanationPrompt }],
    model: "gpt-3.5-turbo",
  });

  return completion.choices[0].message.content;
}

module.exports = { explainFineMessage };
