MinMaxNormalize <- function(values) {
  #normalize to [0,1]
  columns <- base::subset(values, select = 2:ncol(values))
  values  <- base::data.frame(
    values["topics"],
    base::apply(columns, 2, function(column) {
      scales::rescale(column, to = c(0, 1), from = range(column))
    })
  )
  #melt
  values <- reshape2::melt(values, id.vars = "topics", na.rm = TRUE)
  #separate max-arg & min-arg metrics
  values$group <- values$variable %in% c("Griffiths2004", "Deveaud2014")
  values$group <- base::factor(
    values$group,
    levels = c(FALSE, TRUE),
    labels = c("minimize", "maximize")
  )
  return(values)
}