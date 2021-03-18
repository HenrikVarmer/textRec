# textRec (work in progress)
textRec utlizes Latent Dirichlet Allocation and Jensen-Shannon-Divergence on the discrete probability distributions over LDA topics per document, in order to recommend unique and novel documents to specific users.

In order to get recoommendations, simply input a dataframe of users, a dataframe of documents, and a dataframe of user/document interactions, set model hyperparameters and the amount of LDA topics to model (or alternatively, rely on textRec to automate modt hyperparameters using ldatuning::FindTopicsNumber() to find an optimal k number of topics). 

If not all customers have interaction history with documents, you can use the integrated ColdStart engine in order to find k-nearest neighbors and force a cold-start a recommendation for those users. See below function use example, for an explanation of hyperparameters and inputs. 

Run examplerun.R for an example of functionality and use, with test data from the /dat folder. 

### Installing textRec
Install the package directly from github with devtools. Run the first line if you do not currently have devtools installed.

```R
# install.packages('devtools') 
devtools::install_github('HenrikVarmer/textRec')
```

# Main functions

### textRec() function: Dataframes and hyperparameters

```textRec()``` is the main function, which takes three dataframe inputs, and outputs one single dataframe containing all users with recommendations. One row in the output is one recommendation for one user. See below comments for a brief explanation of what each parameter requires. 

```R 
textRec(users = custo,                          # dataframe of customers/users
        documents = texts,                      # dataframe of documents
        user_id = "UserID",                     # user ID column name
        text_id = "TextID",                     # document ID column name
        text_column_name = "DocumentText",      # text column name
        interactions = inter,                   # dataframe of interactions
        ngram_vector = c(1, 2),                 # vector of min and max ngrams
        lda_method = "Gibbs",                   # LDA method
        topics = 4,                             # k topics
        automate_topics = FALSE,                # automate topics TRUE/FALSE
        min_topics = 10,                        # min topics to iterate from (optional)
        max_topics = 120,                       # max topics to iterate to (optional)
        iterate_by = 3,                         # topic search iterate by (optional)
        lda_alpha = 0.2,                        # LDA alpha hyperparameter (optional)
        r_seed = 123,                           # R random seed for repex (optional)
        jsd_max = 0.1,                          # JSD max value to recommend (optional)
        stopwords = c("example", "vector"),     # stopword vector (optional)
        enable_coldstart = FALSE)               # coldstart (TRUE/FALSE)
        
```

#### Output example:

In the output dataframe returned from the ```textRec()``` function, the 'item_history' column constitutes the user interaction history. The 'recommendation' column indicates the recommended document based on Jensen-Shannon Divergence from text documents with which the user has interacted, to another document. The 'votes' column specifies number of times this document was recommended to nearest neighbors, if the recommendation was provided by the cold-start engine. The 'type' column specifies whether the recommendation is derived from the LDA model or from the cold-start engine.

| doc_history    | customer   |	recommendation  |	JSD   | votes  | type   |
|----------------:|-----------:|-----------------:|----------:|------------:|-------:|
| 20,23,24,27     |     1001   |	           29   |	0.01      | NA	      | LDA_JSD  |
| 22,27,33        |     1002   |	           20   |	0.08      | NA	      | LDA_JSD   |
| 11,20,33        |     1003   |	           27   |	NA        | 32	      | ColdStart   |


