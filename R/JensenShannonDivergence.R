JensenShannonDivergence <- function(p, q) {
  m <- 0.5 * (p + q)
  return(0.5 * (sum(p * log(p / m)) + sum(q * log(q / m))))
}