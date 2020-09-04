JSDbyRow <- function(v,data){
  return(apply(data,1,JensenShannonDivergence,v))
}