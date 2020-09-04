


ComputeInteractionJSD <- function(df, text_id) {
  
  # apply cdmwid function
  prob_distrib <- create_data_matrix_with_ID(df_in = df, colcol = "topic", 
                             value = "gamma", colrow = text_id)
  
  col_prob_distrib <- prob_distrib
  

  prob_distrib[,text_id] <- as.numeric(prob_distrib[,text_id])
  col_prob_distrib[,text_id] <- as.numeric(col_prob_distrib[,text_id])
  
  #format as dataframes
  prob_distrib <- as.data.frame(prob_distrib)
  col_prob_distrib <- as.data.frame(col_prob_distrib)
  
  #set rownames to ID columns
  row.names(prob_distrib) <- prob_distrib[,text_id]
  row.names(col_prob_distrib) <- col_prob_distrib[,text_id]
  
  #delete ID columns now
  prob_distrib[,text_id] <- NULL
  col_prob_distrib[,text_id] <- NULL
  
  #create numeric matrix
  prob_distrib <- as.matrix(prob_distrib)
  col_prob_distrib <- as.matrix(col_prob_distrib)
  
  #construct jensen-shannon divergence matrix
  dist <- JSD_matrix(prob_distrib, col_prob_distrib) %>%
    as.data.frame()
 
  #assign rownames so we can melt
  dist[,text_id] <- rownames(dist)
  
  
  m_all_dist <- melt(dist, id = text_id)
  
  return(m_all_dist) 
  
  }