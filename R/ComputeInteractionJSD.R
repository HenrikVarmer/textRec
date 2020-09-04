


ComputeInteractionJSD <- function(df, text_id) {
  
  # apply cdmwid function
  prob_distrib <- create_data_matrix_with_ID(df_in = df, colcol = "topic", 
                             value = "gamma", colrow = text_id)
  
  col_prob_distrib <- prob_distrib
  

  prob_distrib[,text_id] <- as.numeric(prob_distrib[,text_id])
col_prob_distrib[,text_id] <- as.numeric(col_prob_distrib[,text_id])
  
  #construct jensen-shannon divergence matrix
  dist <- JSD_matrix(prob_distrib, col_prob_distrib)
 
  m_all_dist <- melt(dist, id = text_id)
  
  return(m_all_dist) 
  
  }