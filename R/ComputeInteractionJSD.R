


ComputeInteractionJSD <- function(df) {
  
  # apply cdmwid function
  prob_distrib <- create_data_matrix_with_ID(df_in = df, colcol = "topic", 
                             value = "gamma", colrow = "document")
  
  col_prob_distrib <- prob_distrib

  #prob_distrib[,"document"] <- as.numeric(prob_distrib[,"document"])
  #col_prob_distrib[,"document"] <- as.numeric(col_prob_distrib[,"document"])
  
  #format as dataframes
  prob_distrib <- as.data.frame(prob_distrib)
  col_prob_distrib <- as.data.frame(col_prob_distrib)
  
  #set rownames to ID columns
  row.names(prob_distrib) <- prob_distrib[,"document"]
  row.names(col_prob_distrib) <- col_prob_distrib[,"document"]
  
  #delete ID columns now
  prob_distrib[,"document"] <- NULL
  col_prob_distrib[,"document"] <- NULL
  
  #create numeric matrix
  prob_distrib <- as.matrix(prob_distrib)
  col_prob_distrib <- as.matrix(col_prob_distrib)
  
  #construct jensen-shannon divergence matrix
  dist <- JSD_matrix(prob_distrib, col_prob_distrib) %>%
    as.data.frame()
 
  #assign rownames so we can melt
  dist[,"document"] <- rownames(dist)
  
  
  m_all_dist <- melt(dist, id = "document")
  
  m_all_dist <- m_all_dist %>% 
    rename(document_dist = variable)
  
  return(m_all_dist) 
  
  }