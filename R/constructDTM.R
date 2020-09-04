library(SnowballC)
# creates IDA Document-Term-Matrix
constructDTM <- function(df, id, doc_col, stopwords, ngram_vector) {
  StopWords <- stopwords

  
  return(
    CreateDtm(doc_vec = df[[doc_col]], # character vector of documents
              doc_names = df[[id]], # document names
              ngram_window = ngram_vector, # minimum and maximum n-gram length
              stopword_vec = c(StopWords), 
              stem_lemma_function = function(x) SnowballC::wordStem(x, "porter"), #apply porter's stemming and lemmitization
              lower = TRUE, # lowercase 
              remove_punctuation = TRUE, #punctuation 
              remove_numbers = TRUE, #want numbers removed?
              verbose = FALSE, #turn off status bar 
              cpus = parallel::detectCores()) #default is all available cpus on the system
  )
  
}


