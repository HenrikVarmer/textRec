# textRec
textRec is an R package utilizing Latent Dirichlet Allocation and Jensen-Shannon-Divergence computations in order to recommend unique and novel (never-before-seen in training) text documents to specific users. 


# WORK IN PROGRESS

```R 

hyperparameters <- c(LDAmethod = "Gibbs", Topics = 50, alpha = 0.2, seed = 123, JSDmax = 0.08)

textRec(users_df, # df of users
        text_df,  # df of documents
        user_ID,  # ID of users
        text_ID,  # ID of documents
        interactions_df, # df containing user/doc interactions
        hyperparameters = parameters) # list of hyperparameters
        
```
