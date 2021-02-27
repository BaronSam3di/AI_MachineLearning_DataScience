 
# Machine Learning and Artificial Intelligence
KEY
(R:) Resource
(Q:) Question to be answered

- AI is concerned with: Reasoning, knowledge representation, planning, learning, natural language processing, perception, robotics, social intelligence, general intelligence.
- ML is concerned with: Learning functions, Learning patterns to do things like Classification, prediction... 
 
Regression: data corresponding to multiple attributes and you want to fit to a curve.

# Considerations on your Journey to AI
### Clarity of purpose
- Be clear on why do you want to use AI for this business problem. – Automation or optimization with decision support.
- Be clear on what AI can or cannot do for the problem at hand.
- Have clarity on how it fits into and connects with existing business processes and existing tools.
- Define goals and value measurement criteria. 
- Define exit strategy if things don’t work out.

### Setting expectations with end users
- How does the end user experience change?
- Set expectations for end users.
- Customer satisfaction impact.

### Time to Value
- Be clear on time to integrate: how long it takes to implement the solution, to integrate with existing tools and business processes.
- Be clear on time to value: What pre-requisites (data, labeling etc.) does it need to deliver value.
- What does it buy me? Which metrics does it improve?
- Is there a POC period during which the model needs to learn?
- Do SMEs need to provide rules/feedback to bootstrap or
improve the system before it can be fully ready?

### Data
- How much data is needed by the AI model/solution to train and to get to value?
- Where does this data reside? How long does it take to aggregate or assemble it?
- Ownership, governance and compliance of data. Who?
- Lineage of data for audit
- Is labeling required? If so, who will do it? How much does it cost? How long does it take?

### Skills
- Are data scientists needed to manage the lifecycle of the AI models?
- What training is required to existing people in current roles to get them to manage AI models in production?
- Are new roles needed in the organization/company?

### Tools & Infrastructure
- How much infrastructure is needed to train and run the AI models?
- How to optimize costs?
- Decide on On-prem vs Cloud
availability for the AI model.
- Onboarding, training and lifecycle tools.

# Measureing Success
## Typical metrics to measure the success/Accuracy
- Precision = TP/(TP+FP) - Of the samples tested, how precise were you.  
- Recall = TP/(TP+FN) - Of the category overall you are looking for, how many did you get right. 
- Accuracy = (TP+TN)/(TP+TN+FN+FP) 
- F1 = 2*(precision*recall)/(precision+recall) 
- F-measure
- Mean Absolute Error 
- Word Error Rate 
- Sentence Error Rate 
- Fit (under fit/good fit/overfit)
- Confusion matrix 
- General Language Understanding Evaluation (GLUE) Score

R: https://mccormickml.com/2019/11/05/GLUE/
 
 Narrowing the problem will accelerate the learning and accuracy. Context switching again!
R: http://jalammar.github.io/illustrated-word2vec/
 
Cool - R: https://landscape.lfai.foundation/zoom=120

## Asking When you can accept the response from your model
R: https://arxiv.org/abs/1808.07261

We know how to measure accuracy. But how do we measure ... 
- Robustness, 
- accountability, 
- consistency, 
- explainability, 
- fairness, 
- continuous learning, 
- transparency, 
- security & compliance

R: [Characterizing machine learning process: A maturity framework](https://arxiv.org/abs/1811.04871)
R: [Model Maturity Assessment](https://www.ibm.com/cloud/architecture/assessments/ai-maturity-assessment)

## MODEL CREATION for enterprises
### Pneumonic - **RACE your FACTS**

- **Robustness**: Monitor carefully to mitigate adversarial manipulation of training data, payload data.
- **Accuracy**: Narrow the scope of the problem
- **Consistency**: Statistical improvements Vs. consistent improvements? Know your options. Machine learning + rules
- **Explainability**
- **Fairness**: 
    - Lack of clarity in purpose leads to undesirable biases (perceivable prejudice in the prediction outcomes). Also check your training data set. 
- **Accountability**
- **Continuous learning**: 
    - Enable continuous improvement via model customization
    - Diligent with Error Analysis and fix the ones that matter in each iteration, if you can't fix them all with your resources! 
- **Transparency**: How transparent do you need to be? Black-box testing after-the-fact testing of Machine learning + rules.
- **Security and Privacy**

## Strategies for mitigateing unwanted biases
1. Set clear goals, Start with Test Cases. 
2. Declare your biases
3. Can you pre-define your bias attributes? If so, ensure proper representation in the models during training/building.  >>  Age, gender, race, geography
4. What about unknown biases? Have strategies for discovering them dynamically. 
 - Names of specific individuals, political parties, organizations Generate alerts
 - Allow for bias measurement against user-defined attributes 
5. De-biasing techniques (https://aif360.mybluemix.net/)
 - Depends on the type of the error.
 - Data augmentation, data synthesis, quality measurement metrics, simulation. 

 
## Strategies for dealing with Transparency & Explainability
Problem: IF you have to explain why you have choosen an outcome eg: regulations. 

1. Understand your options and their tradeoffs. Deep-learning models Vs. rules ( eg Linear regression is more transparant if required)
- Build deep-learning models and try to explain their predictions after-the-fact? or
- Build transparent models that might take time and effort but are explainable from the get-go?

2. If you must build statistical deep-learning models, then use black-box testing techniques to understand and explain its behavior after-the-fact.
- Pros: relatively easy to build with applied machine learning knowledge.
- Cons: requires large amounts of labeled data, non-transparent, doesn’t guarantee consistent learning (may unlearn things in each iteration)

3. If you use rule-based approach
- Pros: explainable, transparent, consistent improvements across iterations 
- Cons: careful crafting required, expert validation of rules

## Learning and Unlearning
- Statistical machine learning models may forget what they learnt once.
- This leads to inconsistencies in predictions.
 “...there comes a time when for every addition of knowledge you forget something that you knew before...” – Sherlock Holmes in ‘A Study in Scarlet’ – Sir Arthur Conan Doyle.

## Strategies for dealing with Accountability
As you are feeding in more data with each iteration, if you are not careful, you could be feeding any kind of data with any number of labels in different volumes from what you had originally intended therefore you are drifting your model to a different place and direction than you had originally intended. Especially consider public domain models if any. 
Make sure you are plotting the distributions of the training data sets from where you started to where you are.

1. Many industries are regulated! All predictions and transactions need to be logged and maintained.
2. Maintaining training data lineage and model lineage along with payload and predictions is critical.
3. Analysis of transaction logs for drift and misfit analysis could be very useful to make any necessary adjustments.

## Strategies for making AI models Robust
1. Monitor carefully to mitigate adversarial manipulation of training data, payload data. Manual and automation checks may be required.
- Adversarial training: Generating lots of adversarial examples and label them as `not`s to make the model more robust.
2. Start with testing the models for adversarial samples
- There are toolkits for testing models (https://github.com/IBM/adversarial-robustness-toolbox)
3. There may not be such a thing as fool proof security!
- Be vigilant, do random checks by humans, incorporate robust testing of training data.
