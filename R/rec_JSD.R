
# generate the final recommendations based on max_jsd and a basic lookup in hist divergence table
rec_JSD <- function(interactions, jsd_divergence, text_id, user_id, jsd_max) {
  
  interactions[,text_id] <- as.integer(interactions[,text_id])
  jsd_divergence[,"document"] <- as.integer(jsd_divergence[,"document"])
  
  var1 <- text_id
  var2 <- "document"
  
  
  interactions <- interactions %>%
    inner_join(jsd_divergence, by = setNames(var2, var1))
  
  final_hist <- interactions %>% 
    rename(recommendation = document_dist) %>% 
    rename(JSD = value) %>% 
    rename(doc_history = TextID) %>% 
    filter(JSD < jsd_max)
  
  return(final_hist)
  
}
