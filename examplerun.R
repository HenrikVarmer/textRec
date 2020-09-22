


# Example RUn

source("./R/textRec.R")
source("./R/constructDTM.R")
source("./R/ComputeInteractionJSD.R")
source("./R/create_data_matrix_with_ID.R")
source("./R/JSD_matrix.R")
source("./R/JSD_by_row.R")
source("./R/JensenShannonDivergence.R")
source("./R/rec_JSD.R")
library(reshape2)

# load data

custo <- read.csv("./dat/users.csv", stringsAsFactors = FALSE, sep = ";", encoding = "UTF-8") %>% rename(UserID = X.U.FEFF.UserID)
texts <- read.csv("./dat/articles.csv", stringsAsFactors = FALSE, sep = ";", encoding = "UTF-8") %>% rename(TextID = X.U.FEFF.TextID)
inter <- read.csv("./dat/userinteractions.csv", stringsAsFactors = FALSE, sep = ";", encoding = "UTF-8") %>% rename(UserID = X.U.FEFF.UserID)

head(custo)
#head(texts)
head(inter)

recommendations <- textRec(users = custo, 
                           documents = texts, 
                           user_id = "UserID", 
                           text_id = "TextID",
                           text_column_name = "DocumentText",
                           interactions = inter, 
                           ngram_vector = c(1, 2),
                           lda_method = "Gibbs",
                           topics = 4,
                           automate_topics = FALSE,
                           min_topics = 10,
                           max_topics = 120,
                           iterate_by = 3,
                           lda_alpha = 0.2, 
                           r_seed = 123, 
                           jsd_max = 0.1,
                           stopwords = c("example", "of", "stopword", "vector"),
                           enable_coldstart = FALSE)


head(recommendations)


