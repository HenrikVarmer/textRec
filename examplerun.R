


# Example RUn

source("./R/textRec.R")
source("./R/constructDTM.R")
source("./R/ComputeInteractionJSD.R")
library(reshape2)

# load data

custo <- read.csv("./dat/users.csv", stringsAsFactors = FALSE, sep = ";") %>% rename(UserID = ï..UserID)
texts <- read.csv("./dat/articles.csv", stringsAsFactors = FALSE, sep = ";") %>% rename(TextID = ï..TextID)
inter <- read.csv("./dat/userinteractions.csv", stringsAsFactors = FALSE, sep = ";") %>% rename(UserID = ï..UserID)

head(custo)
head(texts)
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
                           stopwords = c("No, hi"),
                           enable_coldstart = FALSE)


