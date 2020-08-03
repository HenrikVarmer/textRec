# textRec
textRec is an R package utilizing Latent Dirichlet Allocation and Jensen-Shannon-Divergence computations in order to recommend unique and novel (never-before-seen in training) text documents to specific users. 


# WORK IN PROGRESS

```R 

hyperparameters <- c(LDAmethod = "Gibbs", Topics = 50, alpha = 0.2, seed = 123, JSDmax = 0.08)

textRec(users = users_df, # df of users
        documents = text_df,  # df of documents
        user_ID = users_df$ID,  # ID of users
        text_ID = text_df$ID,  # ID of documents
        interactions = interactions_df, # df containing user/doc interactions
        hyperparameters = c( # list of hyperparameters
                LDAmethod = "Gibbs", # choose which method the LDA model should use
                Topics = 50, # choose the K number of topics with which to run the LDA model
                Automate_topics = FALSE # choose if # of topics should be automated. FALSE = no automation, K is used. TRUE = automated through and LDA optimizer. Topics parameter is ignored as a result.
                alpha = 0.2, # set alpha hyperparameter for the LDA model
                seed = 123, # set random seed 
                JSDmax = 0.08)) # set JSDmax in order to qualify as a recommendation. anything below is ignored.
        
```
