# textRec
textRec is an R package utilizing Latent Dirichlet Allocation and Jensen-Shannon-Divergence computations in order to recommend unique and novel (never-before-seen in training) text documents to specific users. 


# WORK IN PROGRESS

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
                jsd_max = 0.1))         # set maximum Jensen-Shannon Divergence to qualify as recomnedation
        
```
