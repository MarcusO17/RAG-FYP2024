# Evaluation Standards 

## 1. Base RAG (No Reranker)
Evaluate the initial RAG model setup without a reranker.

### b. Generation Quality
- **Accuracy and Completeness**: Evaluate the accuracy and completeness of the generated answers.
- **Hallucinations**: Identify and quantify instances where the model generates information not supported by the retrieved documents.

## 2. RAG with Reranker
Evaluate the RAG model enhanced with a reranker that reorders the retrieved documents.

### b. Generation Quality
- **Accuracy and Completeness**: Evaluate the accuracy and completeness of the generated answers.
- **Hallucinations Reduction**: Compare the frequency of hallucinations before and after applying the reranker.

## 3. Finetuned Reranker with Finetuned Embeddings
Evaluate the RAG model with both the reranker and embeddings finetuned for the specific task or domain.

### b. Generation Quality
- **Accuracy and Completeness**: Evaluate the accuracy and completeness of the generated answers.

## Evaluation Metrics

### a. Precision and Recall
- **Precision**: Calculate the proportion of relevant documents among the top N retrieved documents.
- **Recall**: Calculate the proportion of relevant documents retrieved out of all relevant documents available.

### b. F1 Score
- **F1 Score**: Calculate the harmonic mean of precision and recall for a balanced evaluation metric.


## Evaluation Procedure

1. **Baseline Evaluation (Base RAG)**:
   - Conduct initial evaluations to establish a performance baseline for retrieval quality and generation quality.
   - Metrics: Precision, Recall, F1 Score, Human Evaluation Scores.

2. **Reranker Integration (RAG with Reranker)**:
   - Integrate the reranker and evaluate its impact on retrieval and generation quality.
   - Compare the performance metrics with the baseline.

3. **Finetuning Evaluation (Finetuned Reranker with Finetuned Embeddings)**:
   - Evaluate the model with finetuned reranker and embeddings.
   - Compare the performance metrics with the previous configurations.

## Continuous Monitoring and Feedback

- **User Feedback**: Collect feedback from users regarding the accuracy and relevance of the information provided by each configuration.
- **Error Analysis**: Regularly analyze errors and performance metrics to identify areas for improvement.
- **Iterative Improvements**: Use the evaluation results to iteratively refine and improve the model configurations.


NOTES
** Try https://huggingface.co/vectara/hallucination_evaluation_model for evalaution

1. FIND A OPEN-DOMAIN DATASET with CONTEXT
sub tasks evaluate
on Closed Domain QA
on Open Domain QA

https://aclanthology.org/2020.nlpcovid19-acl.18/
CMU 
https://www.youtube.com/watch?v=Ki8aZKugTJ0