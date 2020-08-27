# textRec
textRec is an R package utilizing Latent Dirichlet Allocation and Jensen-Shannon-Divergence computations in order to recommend unique and novel (never-before-seen in training) text documents to specific users. 


# Main functions

### textRec() function: Dataframes and hyperparameters

```textRec()``` is the main function, which takes three dataframe inputs, and outputs one single dataframe containing all users with recommendations. One row in the output is one recommendation for one user. See below comments for a brief explanation of what each parameter requires. 

```R 
textRec(users = users_df,               # df of users
        documents = text_df,            # df of documents
        user_id = users_df$ID,          # ID of users
        text_id = text_df$ID,           # ID of documents
        interactions = interactions_df, # df containing user/doc interactions
        hyperparameters = c(            # list of hyperparameters
                lda_method = "Gibbs",   # set which method the LDA model should use
                topics = 50,            # set the K number of topics with which to run the LDA model
                automate_topics = FALSE # set whether the number of topics should be automated
                alpha = 0.2,            # set alpha hyperparameter for the LDA model
                seed = 123,             # set random seed 
                jsd_max = 0.1)          # set maximum Jensen-Shannon Divergence to qualify as recomnedation
        enable_coldstart = TRUE)        # toggles whether knn cold start engine should be enabled
        
```

#### Output example:

In the output dataframe returned from the ```textRec()``` function, the 'item_history' column constitutes the user interaction history. The 'recommendation' column indicates the recommended document based on Jensen-Shannon Divergence from text documents with which the user has interacted, to another document. The 'votes' column specifies number of times this document was recommended to nearest neighbors, if the recommendation was provided by the cold-start engine. The 'type' column specifies whether the recommendation is from the LDA model or from cold-start.

| doc_history    | customer   |	recommendation  |	JSD   | votes  | type   |
|----------------:|-----------:|-----------------:|----------:|------------:|-------:|
| 20,23,24,27     |     1001   |	           29   |	0.01      | NA	      | LDA_JSD  |
| 22,27,33        |     1002   |	           20   |	0.08      | NA	      | LDA_JSD   |
| 11,20,33        |     1003   |	           27   |	NA        | 32	      | ColdStart   |

### posterior() function: Dataframes and hyperparameters
```R
posterior(event_lda, # trained LDA model
          dtm_new)   # document term matrix
```

