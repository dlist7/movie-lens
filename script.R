##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#####
##### Everything below this is my additions. #####
#####

if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")

library(data.table)
library(lubridate)
library(gridExtra)
library(grid)

###
### Model Parameters
###
#seed <- 1
#lmv <- 2.5
#lu <- 5
#lg <- 0
#lgu <- 8
#bt_window <- 1920
#bin_span <- 0.1

# Set the default seed value
if (!exists("seed")) {
  seed <- 1
}
parameters <- data.frame(parameter = "seed", value = seed)

# Don't calculate the final result unless if any parameters have been adjusted
# from the values calculated with a seed of 1. (Parameters are manually adjusted
# to validate the Cross Validation assumptions in Section "Cross Validation Sanity
# Check".) Note this also ensures the final result is only calculated from a clean
# environment.
if (seed == 1 && !exists("lmv") && !exists("lu") && !exists("lg") && !exists("lgu")
    && !exists("bt_window") && !exists("bin_span")) {
  calculate_final <- TRUE
} else {
  calculate_final <- FALSE
}

###
### Setup Cross Validation
###

# Build new train/test sets so we don't use the final_holdout_test set
# until the very end.  Use 1.0/9.0 so edx_test is the roughly the
# same size as final_holdout_test.
set.seed(seed, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 1.0/9.0,
                                      list = FALSE)
edx_train <- edx[-edx_test_index,]
edx_temp <- edx[edx_test_index,]

# As above, keep only the rows with users and movies that exist
# in the training set.
edx_test <- edx_temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")
edx_removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, edx_removed)

rm(edx_test_index, edx_temp, edx_removed)

# Define a function for calculating for calculating rmse.
rmse <- function(y_hat, y) {
  sqrt(mean((y_hat - y)^2))
}

# Calculate the approximate RMSE from randomly guessing.
random_guess <- sample(c(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
                      nrow(edx_test), replace = TRUE)
y_hat <- random_guess
rmse_random <- rmse(y_hat, edx_test$rating)
rmse_random
results <- data.frame(description = "Random Guess",
                      rmse = round(rmse_random, digits = 5))

# Calculate the average rating from the training set, mu, and calculate the
# RMSE associated with always using mu.
mu <- mean(edx_train$rating)
y_hat <- mu
rmse_mu <- rmse(y_hat, edx_test$rating)
rmse_mu
results[nrow(results) + 1,] = c("Average Rating (mu)",
                                round(rmse_mu, digits = 5))

# Calculate RMSE considering just movie effects.
bi <- edx_train %>%
  group_by(movieId) %>% summarize(bi = mean(rating - mu))
y_hat <- mu + edx_test %>% left_join(bi, by = 'movieId') %>% .$bi
rmse_bi <- rmse(y_hat, edx_test$rating)
results[nrow(results) + 1,] = c("Movie Effects (bi)",
                                round(rmse_bi, digits = 5))

# Calculate a value for lambda only considering movie effects.
if (!exists("lmv")) {

  lmvs <- seq(0, 10, 0.25)
  bi_sums <- edx_train %>% group_by(movieId) %>%
    summarize(sum = sum(rating - mu), n = n())
  lmv_rmses <- sapply(lmvs, function(l) {
    bi_r <- bi_sums %>% summarize(movieId = movieId, bi_r = sum/(n + l))
    y_hat <- mu + edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r
    rmse(y_hat, edx_test$rating)
  })
  
  # Choose the lambda with the smallest rmse.
  lmv <- lmvs[which.min(lmv_rmses)]
  
  # Store the lambda/rmse values in data frame so they can be more easily
  # plotted in the report.
  lmv_df <- data.frame(lambda = lmvs, rmse = lmv_rmses)

}
lmv
parameters[nrow(parameters) + 1,] = c("lmv", lmv)

# Calculate rmse considering just movie effects plus regularization.
bi_r <- edx_train %>%
  group_by(movieId) %>%
  summarize(bi_r = sum(rating - mu)/(n() + lmv))
y_hat <- mu +
  edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r
rmse_bi_r <- rmse(y_hat, edx_test$rating)
rmse_bi_r
results[nrow(results) + 1,] = c("Movie Effects w/ Regularization (bi_r)",
                                round(rmse_bi_r, digits = 5))

# Calculate rmse additionally considering user effects.
bu <- edx_train %>%
  left_join(bi_r, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(bu = mean(rating - mu - bi_r))
y_hat <- mu +
  edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
  edx_test %>% left_join(bu, by = 'userId') %>% .$bu
rmse_bu <- rmse(y_hat, edx_test$rating)
rmse_bu
results[nrow(results) + 1,] = c("User Effects (bu)",
                                round(rmse_bu, digits = 5))

# Calculate a user effects specific lambda.
if (!exists("lu")) {
  
  lus <- seq(0, 10, 0.25)
  bu_sums <- edx_train %>%
    left_join(bi_r, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(sum = sum(rating - mu - bi_r), n = n())
  lu_rmses <- sapply(lus, function(l) {
    bu_r <- bu_sums %>%
      summarize(userId = userId, bu_r = sum/(n + l))
    y_hat <- mu +
      edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
      edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r
    rmse(y_hat, edx_test$rating)
  })
  
  # Choose the lambda with the smallest rmse.
  lu <- lus[which.min(lu_rmses)]
  
  # Store the lambda/rmse values in data frame so they can be more easily
  # plotted in the report.
  lu_df <- data.frame(lambda = lus, rmse = lu_rmses)
  
}
lu
parameters[nrow(parameters) + 1,] = c("lu", lu)

# Calculate rmse additionally considering user effects plus regularization.
bu_r <- edx_train %>%
  left_join(bi_r, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(bu_r = sum(rating - mu - bi_r)/(n() + lu))
y_hat <- mu +
  edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
  edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r
rmse_bu_r <- rmse(y_hat, edx_test$rating)
rmse_bu_r
results[nrow(results) + 1,] = c("User Effects w/ Regularization (bu_r)",
                                round(rmse_bu_r, digits = 5))

# Calculate rmse additionally considering genre effects.  Note the genres
# column includes zero to many genres separated by the pipe character.
# The following considers each combination of genres as a separate "genre"
# so for instance the effects of "Comedy", "Romance", and "Comedy|Romance"
# are considered separately.
bg <- edx_train %>%
  left_join(bi_r, by = 'movieId') %>%
  left_join(bu_r, by = 'userId') %>%
  group_by(genres) %>%
  summarize(bg = mean(rating - mu - bi_r - bu_r))
y_hat <- mu +
  edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
  edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r +
  edx_test %>% left_join(bg, by = 'genres') %>% .$bg
rmse_bg <- rmse(y_hat, edx_test$rating)
rmse_bg
results[nrow(results) + 1,] = c("Genre Effects (bg)",
                                round(rmse_bg, digits = 5))

# Calculate a genre effects specific lambda.
if (!exists("lg")) {
  
  lgs <- seq(0, 10, 0.25)
  bg_sums <- edx_train %>%
    left_join(bi_r, by = 'movieId') %>%
    left_join(bu_r, by = 'userId') %>%
    group_by(genres) %>%
    summarize(sum = sum(rating - mu - bi_r - bu_r), n = n())
  lg_rmses <- sapply(lgs, function(l) {
    bg_r <- bg_sums %>%
      summarize(genres = genres, bg_r = sum/(n + l))
    y_hat <- mu +
      edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
      edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r +
      edx_test %>% left_join(bg_r, by = 'genres') %>% .$bg_r
    rmse(y_hat, edx_test$rating)
  })
  lg <- lgs[which.min(lg_rmses)]
  
  # Store the lambda/rmse values in data frame so they can be more easily
  # plotted in the report.
  lg_df <- data.frame(lambda = lgs, rmse = lg_rmses)
  
}
lg
parameters[nrow(parameters) + 1,] = c("lg", lg)

# No benefit observed using regularization with genre effects so need
# to calculate a new rmse.

# Calculate rmse additionally considering user specific genre effects.
bgu <- edx_train %>%
  left_join(bi_r, by = 'movieId') %>%
  left_join(bu_r, by = 'userId') %>%
  left_join(bg, by = 'genres') %>%
  group_by(genres, userId) %>%
  summarize(bgu = mean(rating - mu - bi_r - bu_r - bg))
y_hat <- mu +
  edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
  edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r +
  edx_test %>% left_join(bg, by = 'genres') %>% .$bg +
  edx_test %>% left_join(bgu, by = c('genres', 'userId')) %>%
  # Replace N/A with 0 when user has no ratings for a given genre.
  mutate(bgu = ifelse(is.na(bgu), 0, bgu)) %>% .$bgu
rmse_bgu <- rmse(y_hat, edx_test$rating)
rmse_bgu
results[nrow(results) + 1,] = c("User Specific Genre Effects (bgu)",
                                round(rmse_bgu, digits = 5))

# Calculate a user specific genre effects specific lambda.
if (!exists("lgu")) {
  
  lgus <- seq(0, 20, 0.5)
  bgu_sums <- edx_train %>%
    left_join(bi_r, by = 'movieId') %>%
    left_join(bu_r, by = 'userId') %>%
    left_join(bg, by = 'genres') %>%
    group_by(genres, userId) %>%
    summarize(sum = sum(rating - mu - bi_r - bu_r - bg), n = n())
  lgu_rmses <- sapply(lgus, function(l) {
    bgu_r <- bgu_sums %>%
      summarize(genres = genres, userId = userId, bgu_r = sum/(n + l))
    y_hat <- mu +
      edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
      edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r +
      edx_test %>% left_join(bg, by = 'genres') %>% .$bg +
      edx_test %>% left_join(bgu_r, by = c('genres', 'userId')) %>%
      # Replace N/A with 0 when user has no ratings for a given genre.
      mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r)) %>% .$bgu_r
    rmse(y_hat, edx_test$rating)
  })
  lgu <- lgus[which.min(lgu_rmses)]
  
  # Store the lambda/rmse values in data frame so they can be more easily
  # plotted in the report.
  lgu_df <- data.frame(lambda = lgus, rmse = lgu_rmses)
  
}
lgu
parameters[nrow(parameters) + 1,] = c("lgu", lgu)

# Calculate rmse additionally considering user specific genre effects plus
# regularization.
bgu_r <- edx_train %>%
  left_join(bi_r, by = 'movieId') %>%
  left_join(bu_r, by = 'userId') %>%
  left_join(bg, by = 'genres') %>%
  group_by(genres, userId) %>%
  summarize(bgu_r = sum(rating - mu - bi_r - bu_r - bg)/(n() + lgu))
y_hat <- mu +
  edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
  edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r +
  edx_test %>% left_join(bg, by = 'genres') %>% .$bg +
  edx_test %>% left_join(bgu_r, by = c('genres', 'userId')) %>%
  mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r)) %>% .$bgu_r
rmse_bgu_r <- rmse(y_hat, edx_test$rating)
rmse_bgu_r
results[nrow(results) + 1,] = c("User Specific Genre Effects w/ Regularization (bgu_r)",
                                round(rmse_bgu_r, digits = 5))

# Put this outside the if to ensure the report has access to it.
# This is just 15 iteratively multiplied by the square root of 2.
bt_windows <- c(15, 21, 30, 42, 60, 85, 120, 170, 240, 339, 480, 679, 960, 1358,
                1920, 2715, 3840, 5431, 7680, 10861, 15360, 21722, 30720, 43445,
                61140, 86400)

if (!exists("bt_window")) {
  
  bt_data <- edx_train %>%
    left_join(bi_r, by = 'movieId') %>%
    left_join(bu_r, by = 'userId') %>%
    left_join(bg, by = 'genres') %>%
    left_join(bgu_r, by = c('genres', 'userId')) %>%
    mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r))
  bt_window_rmses <- sapply(bt_windows, function(w) {
    bt <- bt_data %>%
      mutate(tsw = timestamp %/% w) %>%
      group_by(tsw) %>%
      summarize(bt = mean(rating - mu - bi_r - bu_r - bg - bgu_r))
    y_hat <- mu +
      edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
      edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r +
      edx_test %>% left_join(bg, by = 'genres') %>% .$bg +
      edx_test %>% left_join(bgu_r, by = c('genres', 'userId')) %>%
      mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r)) %>% .$bgu_r +
      edx_test %>% mutate(tsw = timestamp %/% w) %>%
      left_join(bt, by = 'tsw') %>%
      mutate(bt = ifelse(is.na(bt), 0, bt)) %>% .$bt
    rmse(y_hat, edx_test$rating)
  })
  bt_window <- bt_windows[which.min(bt_window_rmses)]
  
  # Store the lambda/rmse values in data frame so they can be more easily
  # plotted in the report.
  bt_window_df <- data.frame(window = bt_windows, rmse = bt_window_rmses)
  
}
bt_window
parameters[nrow(parameters) + 1,] = c("bt_window", bt_window)

# Calculate rmse additionally considering user specific genre effects plus
# regularization.
bt <- edx_train %>%
  left_join(bi_r, by = 'movieId') %>%
  left_join(bu_r, by = 'userId') %>%
  left_join(bg, by = 'genres') %>%
  left_join(bgu_r, by = c('genres', 'userId')) %>%
  mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r)) %>%
  mutate(tsw = timestamp %/% bt_window) %>%
  group_by(tsw) %>%
  summarize(bt = mean(rating - mu - bi_r - bu_r - bg - bgu_r))
y_hat <- mu +
  edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
  edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r +
  edx_test %>% left_join(bg, by = 'genres') %>% .$bg +
  edx_test %>% left_join(bgu_r, by = c('genres', 'userId')) %>%
  mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r)) %>% .$bgu_r +
  edx_test %>% mutate(tsw = timestamp %/% bt_window) %>%
  left_join(bt, by = 'tsw') %>%
  mutate(bt = ifelse(is.na(bt), 0, bt)) %>% .$bt
rmse_bt <- rmse(y_hat, edx_test$rating)
rmse_bt
results[nrow(results) + 1,] = c("Timestamp Window Effects (bt)",
                                round(rmse_bt, digits = 5))

# Movies with fewer ratings tend to have lower average ratings.
# In theory this should be fully taken into account by
# looking at movie effect above, however, regularization
# under-corrects for movies with fewer ratings.  The
# following selects the best value of the span parameter
# for loess smoothing of the resulting bias as a function of the
# number of movie ratings.
if (!exists("bin_span")) {
  
  spans <- seq(0.1, 1.0, 0.1)
  bin_rmses <- sapply(spans, function(s) {
    bin_data <- edx_train %>%
      left_join(bi_r, by = 'movieId') %>%
      left_join(bu_r, by = 'userId') %>%
      left_join(bg, by = 'genres') %>%
      left_join(bgu_r, by = c('genres', 'userId')) %>%
      mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r)) %>%
      mutate(tsw = timestamp %/% bt_window) %>%
      left_join(bt, by = 'tsw') %>%
      mutate(bt = ifelse(is.na(bt), 0, bt)) %>%
      group_by(movieId) %>%
      mutate(n = n()) %>% ungroup() %>%
      group_by(n) %>%
      summarize(avg_bias = mean(rating - mu - bi_r - bu_r - bg - bgu_r - bt))
    bin_fit <- loess(avg_bias~n, data=bin_data, span=s)
    bin <- edx_train %>%
      group_by(movieId) %>%
      summarize(n = n()) %>%
      mutate(bin = predict(bin_fit, .)) %>%
      select(movieId, bin)
    y_hat <- mu +
      edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
      edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r +
      edx_test %>% left_join(bg, by = 'genres') %>% .$bg +
      edx_test %>% left_join(bgu_r, by = c('genres', 'userId')) %>%
      mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r)) %>% .$bgu_r +
      edx_test %>% mutate(tsw = timestamp %/% bt_window) %>%
      left_join(bt, by = 'tsw') %>%
      mutate(bt = ifelse(is.na(bt), 0, bt)) %>% .$bt +
      edx_test %>% left_join(bin, by = 'movieId') %>% .$bin
    rmse(y_hat, edx_test$rating)
  })
  bin_span <- spans[which.min(bin_rmses)]
  
  # Store the span/rmse values in data frame so they can be more easily
  # plotted in the report.
  bin_span_df <- data.frame(span = spans, rmse = bin_rmses)
  
}
bin_span
parameters[nrow(parameters) + 1,] = c("bin_span", bin_span)

# Calculate rmse additionally considering ratings per movie effects.
bin_data <- edx_train %>%
  left_join(bi_r, by = 'movieId') %>%
  left_join(bu_r, by = 'userId') %>%
  left_join(bg, by = 'genres') %>%
  left_join(bgu_r, by = c('genres', 'userId')) %>%
  mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r)) %>%
  mutate(tsw = timestamp %/% bt_window) %>%
  left_join(bt, by = 'tsw') %>%
  mutate(bt = ifelse(is.na(bt), 0, bt)) %>%
  group_by(movieId) %>%
  mutate(n = n()) %>% ungroup() %>%
  group_by(n) %>%
  summarize(avg_bias = mean(rating - mu - bi_r - bu_r - bg - bgu_r - bt))
bin_fit <- loess(avg_bias~n, data=bin_data, span=bin_span)
bin <- edx_train %>%
  group_by(movieId) %>%
  summarize(n = n()) %>%
  mutate(bin = predict(bin_fit, .)) %>%
  select(movieId, bin)
plot(bin_fit$x, bin_fit$fitted)
y_hat <- mu +
  edx_test %>% left_join(bi_r, by = 'movieId') %>% .$bi_r +
  edx_test %>% left_join(bu_r, by = 'userId') %>% .$bu_r +
  edx_test %>% left_join(bg, by = 'genres') %>% .$bg +
  edx_test %>% left_join(bgu_r, by = c('genres', 'userId')) %>%
  mutate(bgu_r = ifelse(is.na(bgu_r), 0, bgu_r)) %>% .$bgu_r +
  edx_test %>% mutate(tsw = timestamp %/% bt_window) %>%
  left_join(bt, by = 'tsw') %>%
  mutate(bt = ifelse(is.na(bt), 0, bt)) %>% .$bt +
  edx_test %>% left_join(bin, by = 'movieId') %>% .$bin
rmse_bin <- rmse(y_hat, edx_test$rating)
rmse_bin
results[nrow(results) + 1,] = c("Movie Number of Ratings Effects (bin)",
                                round(rmse_bin, digits = 5))

rmse_final <- 0
if (calculate_final == TRUE) {
  
  # Calculate the final results with edx and final_holdout_test.
  mu_final <- mean(edx$rating)
  
  bi_r_final <- edx %>%
    group_by(movieId) %>%
    summarize(bi_r_final = sum(rating - mu_final)/(n() + lmv))
  
  bu_r_final <- edx %>%
    left_join(bi_r_final, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(bu_r_final = sum(rating - mu_final - bi_r_final)/(n() + lu))
  
  bg_final <- edx %>%
    left_join(bi_r_final, by = 'movieId') %>%
    left_join(bu_r_final, by = 'userId') %>%
    group_by(genres) %>%
    summarize(bg_final = mean(rating - mu_final - bi_r_final - bu_r_final))
  
  bgu_r_final <- edx %>%
    left_join(bi_r_final, by = 'movieId') %>%
    left_join(bu_r_final, by = 'userId') %>%
    left_join(bg_final, by = 'genres') %>%
    group_by(genres, userId) %>%
    summarize(bgu_r_final = sum(rating - mu_final - bi_r_final - bu_r_final -
                                  bg_final)/(n() + lgu))
  
  bt_final <- edx %>%
    left_join(bi_r_final, by = 'movieId') %>%
    left_join(bu_r_final, by = 'userId') %>%
    left_join(bg_final, by = 'genres') %>%
    left_join(bgu_r_final, by = c('genres', 'userId')) %>%
    mutate(bgu_r_final = ifelse(is.na(bgu_r_final), 0, bgu_r_final)) %>%
    mutate(tsw = timestamp %/% bt_window) %>%
    group_by(tsw) %>%
    summarize(bt_final = mean(rating - mu_final - bi_r_final - bu_r_final -
                                bg_final - bgu_r_final))
  
  bin_data_final <- edx %>%
    left_join(bi_r_final, by = 'movieId') %>%
    left_join(bu_r_final, by = 'userId') %>%
    left_join(bg_final, by = 'genres') %>%
    left_join(bgu_r_final, by = c('genres', 'userId')) %>%
    mutate(bgu_r_final = ifelse(is.na(bgu_r_final), 0, bgu_r_final)) %>%
    mutate(tsw = timestamp %/% bt_window) %>%
    left_join(bt_final, by = 'tsw') %>%
    mutate(bt_final = ifelse(is.na(bt_final), 0, bt_final)) %>%
    group_by(movieId) %>%
    mutate(n = n()) %>% ungroup() %>%
    group_by(n) %>%
    summarize(avg_bias = mean(rating - mu_final - bi_r_final - bu_r_final -
                                bg_final - bgu_r_final - bt_final))
  
  bin_fit_final <- loess(avg_bias~n, data=bin_data_final, span=bin_span)
  bin_final <- edx %>%
    group_by(movieId) %>%
    summarize(n = n()) %>%
    mutate(bin_final = predict(bin_fit_final, .)) %>%
    select(movieId, bin_final)
  
  y_hat <- mu_final +
    final_holdout_test %>% left_join(bi_r_final, by = 'movieId') %>% .$bi_r_final +
    final_holdout_test %>% left_join(bu_r_final, by = 'userId') %>% .$bu_r_final +
    final_holdout_test %>% left_join(bg_final, by = 'genres') %>% .$bg_final +
    final_holdout_test %>% left_join(bgu_r_final, by = c('genres', 'userId')) %>%
    mutate(bgu_r_final = ifelse(is.na(bgu_r_final), 0, bgu_r_final)) %>% .$bgu_r_final +
    final_holdout_test %>% mutate(tsw = timestamp %/% bt_window) %>%
    left_join(bt_final, by = 'tsw') %>%
    mutate(bt_final = ifelse(is.na(bt_final), 0, bt_final)) %>% .$bt_final +
    final_holdout_test %>% left_join(bin_final, by = 'movieId') %>% .$bin_final
  
  rmse_final <- rmse(y_hat, final_holdout_test$rating)
}

print.data.frame(parameters, right = FALSE)
print.data.frame(results, right = FALSE, digits = 5)

rmse_final

save.image("ml-session")
