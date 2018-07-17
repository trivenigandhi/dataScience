## This script takes in the cleaned earnings by year data to test the effect of
## of gender on 2017 income across the top ten largest departments in the city of Boston.

library(data.table) #I use data.table for basic manipulations, however please note that
# installing this package on a Mac requires additional steps. 

library(lme4) # this is the package that provides support for a random effects model

localPath <- ## insert your local directory here
earnings <- fread(paste0(localPath,"earningsByYear.csv")) # read in the data

# This line of code creates a count of all records where earnings2017 is not null,
# sorts the counts in descending fashion, and extracts the names of the
# top ten departments in the data 
topDepartments  <- earnings[!is.na(earnings2017),.N,department][order(-N)][1:10,department]

earnings[,c('Male') := 0] # create a binary field for Male, initialized at 0
earnings[gender == "M", Male :=1] # update the field to 1 where sex is "M"

# Here I cut the full dataset to only the departments in the top ten and only keep
# a few select variables. The information is still at the individual level however.
modelData <- earnings[department %in% topDepartments,.(department,Male,earnings2016,earnings2017)]

# This code builds a random effects model that allows the effect of being male to
# vary across each department, while holding the intercept and effect of 2016 earnings constant
reModel<- lmer(earnings2017 ~ earnings2016 + (0 + Male|department),data = modelData)

# This code extracts the coefficents for each department into a data.table for writing
modelCoefs <- data.table(coefficients(reModel)$department,keep.rownames = TRUE)

setnames(modelCoefs,c("department","male","intercept","earnings2016"))

# fwrite outputs the model coefficents data into a csv file
fwrite(modelCoefs,file = paste0(localPath,"modelCoefficents.csv"))
