library(DBI)
library(dplyr)
library(dbplyr)
library(tidyr)
library(yaml)
library(odbc)
library(stringr)
library(stringi)
library(textmineR)
library(topicmodels)
library(ldatuning)
library(lubridate)
library(tidytext)
source("./R/constructDTM.R")



textRec <- function(users = custo, 
                   documents = texts, 
                   user_id = "UserID", 
                   text_id = "TextID",
                   text_column_name = "DocumentText",
                   interactions = inter, 
                   ngram_vector = c(1, 2),
                   lda_method = "Gibbs",
                   topics = 50,
                   automate_topics = FALSE,
                   min_topics = 10,
                   max_topics = 120,
                   iterate_by = 3,
                   lda_alpha = 0.2, 
                   r_seed = 123, 
                   jsd_max = 0.1,
                   stopwords = c("No, Yes"),
                   trained_LDA = NULL,  
                   enable_coldstart = TRUE) {

  
  
  # transformm data
  message("transforming data...")
  users <- users %>% 
    inner_join(interactions, by = c(user_id))
  message("done!")
  
  message("constructing DTM...")
  dtm <- constructDTM(documents, text_id, text_column_name, stopwords, ngram_vector)
  message("done!")
  
  k <- if(automate_topics) {
    message("attempting to automate topics...")
    #iterate n Latent Direchlet Allocation models and find local minima k
    result <- FindTopicsNumber(
      dtm,
      topics   = seq(from = min_topics, to = max_topics, by = iterate_by),
      metrics  = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
      method   = lda_method,
      control  = list(seed = r_seed),
      mc.cores = parallel::detectCores(),
      verbose  = TRUE
    )
    message(paste0("done! found number of optimal topics to be ", result, " ..."))
  } else {
    message(paste0("k = ", topics))
    topics
    }
  
  
  if(is.null(trained_LDA)) {
    message("training LDA model...")
    # create the LDA model
    lda_m <- LDA(dtm, 
                 k = k, 
                 control = list(
                   alpha = lda_alpha, 
                   seed  = r_seed), 
                 method  = lda_method)
    message("done!")
  } else if(grepl("LDA", class(trained_LDA)[1])) {
    message("trained LDA model supplied, skipping training...")
    lda_m <- posterior(trained_LDA, dtm)
    message("done!")
  }
  
  #calculate which words belong to which topic
  message("assigning LDA topic probablity distributions to documents...")
  document_topics <- tidytext::tidy(lda_m, matrix = "gamma") %>%
    mutate(gamma = round(gamma,5))
  message("done!")
  
  # Get contact divergence
  message("computing Jensen-Shannon divergence...")
  jsd_divergence <- ComputeInteractionJSD(document_topics)
  message("done!")
  
  # Cold Start
#  if (enable_coldstart) {
#    message("cold-starting users with no-interactions using knn...")
#    ColdStart <- ColdStart(users, lda_m)
 # } else {
#    mesage("skipping cold-start!")
#  }
#  message("done!")
  
#  if (is.null(ColdStart)) {
#    Recommendations <- Recommendations } 
##  else {
#    Recommendations <- rbind(EventHistDivergence, ColdStart) }
  
  
  message("computing recommendations...")
  recs <- rec_JSD(interactions, jsd_divergence, text_id, user_id, jsd_max)
  message("done!")
  
  return(recs)
  
}


















