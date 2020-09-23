# textRec (work in progress)
textRec utlizes Latent Dirichlet Allocation and Jensen-Shannon-Divergence computations on the discrete probability distributions over LDA topics per document, in order to recommend unique and novel (never-before-seen in training) text documents to specific users. 

In order to get recoommendations, simply input a dataframe of users, a dataframe of documents, and a dataframe of user/document interactions, set model hyperparameters and the amount of LDA topics to model (or alternatively, rely on textRec to automate modt hyperparameters using ldatuning::FindTopicsNumber() to find an optimal k number of topics). If not all customers have interaction history with documents, you can use the integrated ColdStart engine in order to find k-nearest neighbors and force a cold-start a recommendation for those users. See below function use example, for an explanation of hyperparameters and inputs. 

Run examplerun.R for an example of functionality and use, with test data from the /dat folder. 

# Main functions

### textRec() function: Dataframes and hyperparameters

```textRec()``` is the main function, which takes three dataframe inputs, and outputs one single dataframe containing all users with recommendations. One row in the output is one recommendation for one user. See below comments for a brief explanation of what each parameter requires. 

```R 
textRec(users = custo, 
        documents = texts, 
        user_id = "UserID", 
        text_id = "TextID",
        text_column_name = "DocumentText",
        interactions = inter, 
        ngram_vector = c(1, 2),
        lda_method = "Gibbs",
        topics = 4,
        automate_topics = FALSE,
        min_topics = 10,
        max_topics = 120,
        iterate_by = 3,
        lda_alpha = 0.2, 
        r_seed = 123, 
        jsd_max = 0.1,
        stopwords = c("example", "of", "stopword", "vector"),
        enable_coldstart = FALSE)
        
```

#### Output example:

In the output dataframe returned from the ```textRec()``` function, the 'item_history' column constitutes the user interaction history. The 'recommendation' column indicates the recommended document based on Jensen-Shannon Divergence from text documents with which the user has interacted, to another document. The 'votes' column specifies number of times this document was recommended to nearest neighbors, if the recommendation was provided by the cold-start engine. The 'type' column specifies whether the recommendation is derived from the LDA model or from cold-start.

| doc_history    | customer   |	recommendation  |	JSD   | votes  | type   |
|----------------:|-----------:|-----------------:|----------:|------------:|-------:|
| 20,23,24,27     |     1001   |	           29   |	0.01      | NA	      | LDA_JSD  |
| 22,27,33        |     1002   |	           20   |	0.08      | NA	      | LDA_JSD   |
| 11,20,33        |     1003   |	           27   |	NA        | 32	      | ColdStart   |


