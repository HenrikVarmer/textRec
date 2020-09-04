JSD_by_row <- function(v,data){
  return(apply(data,1,JensenShannonDivergence,v))
}