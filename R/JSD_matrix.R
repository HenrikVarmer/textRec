JSD_matrix <- function(prob_distrib, col_prob_distrib) {
  
  #create numeric matrix
  prob_distrib <- as.matrix(prob_distrib)
  col_prob_distrib <- as.matrix(col_prob_distrib)
  
  data <- apply(prob_distrib, 1, JSD_by_row, col_prob_distrib)
  
  return(data)
  }