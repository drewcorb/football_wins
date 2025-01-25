# Understanding football wins

library(dplyr)
library(readr)
library(tidyr) # pivot_wider
library(stringr) # str_replace_all
library(ggplot2)
library(corrplot)
library(rsample) # resampling functions
library(recipes)
library(parsnip) # boost_tree
library(workflows) # add_xxx functions
library(yardstick) # metric_set, other metric functions
library(dials) # tuning functions
library(tune)
library(finetune) # control_sim_anneal
library(vip) # vip (variable importance plot)

library(lubridate)
library(tidymodels)
library(xgboost)
library(bestNormalize) # step_orderNorm
# library(embed) # step_umap
library(RSQLite)
tidymodels_prefer()
conflicted::conflicts_prefer(dplyr::lag
                             , dplyr::filter)

# ==== Objective of this analysis ====
# I don't know much about soccer but I do enjoy watching it. Other than goals and assists, I don't know how how to look at a player's stat line and know if he/she is good. Thus I'm looking at match team defense and passing stats from the past few years and see which stats are most important in predicting the outcome of a match.

# Let's build a classification model using boosted trees, in this case xgboost.

# ==== Load data from csv ====
# get historic seasons data

defense_data <-
  read_csv(file = "GitHub/football_wins/defense_data.csv") |>
  # should've renamed these cols before creating the csv but such is life
  rename(Home_Goal_Scorers = Home_Goals
         , Away_Goal_Scorers = Away_Goals) |>
  # let's clean up a few more column names. get rid of spaces and some symbols
  rename_with(.fn = ~ str_replace_all(., " ", "_")) |>
  rename_with(.fn = ~ str_replace_all(., "\\+", "_plus_"))

passing_data <-
  read_csv(file = "GitHub/football_wins/passing_data.csv") |>
  # once again
  rename(Home_Goal_Scorers = Home_Goals
         , Away_Goal_Scorers = Away_Goals)

# ==== Initial data processing ====
# First notice that each table contains 2 rows per match: one each for home and away. This results in lots of redundant data. Pivot each wider so there is only one row per match
wide_defense_data <-
  defense_data |>
  select(-Team) |>
  pivot_wider(names_from = Home_Away
              , values_from = Min:Err
              , names_glue = "{Home_Away}_{.value}")

wide_passing_data <-
  passing_data |>
  select(-Team) |>
  pivot_wider(names_from = Home_Away
              , values_from = Min:PrgP
              , names_glue = "{Home_Away}_{.value}")

# Now let's join all these tables together so we have a single table with all the stats for each match in a single row.
all_match_data <-
  wide_defense_data |>
  left_join(wide_passing_data
            , by = join_by(League
                           , Match_Date, Matchweek
                           , Home_Team, Away_Team)) |>
  # lots of duplicate columns, keep those with suffix .x and drop those with .y
  rename_with(.fn = ~ str_replace_all(., "\\.x$", "")) |>
  select(-contains(".y")) |>
  # finally add the column that we'll be predicting on (match result)
  mutate(Match_Result = case_when(Home_Score > Away_Score ~ "home_win"
                                  , Home_Score == Away_Score ~ "draw"
                                  , Home_Score < Away_Score ~ "away_win"))

# ==== Data exploration ====
# First let's take a look at the distribution of outcomes.
all_match_data |>
  count(Match_Result) |>
  mutate(pct = n/sum(n))
# We have a relatively balanced data set, with home wins being the most common and draws being the least common.

# Defensive data
# As far as predictor variables, let's start looking at defensive variables.
# First examine the correlation between the variables. At this point we won't yet need to work with the wide data, so let's select the defensive variables from defense data.
defensive_exploration <-
  defense_data |>
  # add a column denoting goals allowed
  mutate(Goals_Allowed = case_when(Home_Away == "Home" ~ Away_Score
                                   , Home_Away == "Away" ~ Home_Score)) |>
  # select only the variables that are associated with defense
  select(contains("Tackles") | contains("Challenges") | contains("Blocks") |
              contains("Int") | contains("Clr") | contains("Err"))

defensive_exploration |>
  # For now, 
  select(-Tkl_Tackles
         , -Blocks_Blocks
         , -Tkl_Challenges, -Lost_Challenges
         , -Tkl_plus_Int
         ) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# The correlation plot shows that these variables are relatively uncorrelated with each other. However we see that tackles won is somewhat correlated with the number of tackles in each third of the pitch. Let's convert tackles won into a won tackles percent and take another look.
defensive_exploration <-
  defensive_exploration |>
  mutate(TklW_Percent = TklW_Tackles/Tkl_Tackles) |>
  relocate(TklW_Percent) # no other argument moves the column to the front

defensive_exploration |>
  # For now, 
  select(-Tkl_Tackles, -TklW_Tackles
         , -Blocks_Blocks
         , -Tkl_Challenges, -Lost_Challenges
         , -Tkl_plus_Int
  ) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# That looks like it helped. So let's create a new object, model data, that has these transformations.
defense_model_data <-
  defense_data |>
  # add a column denoting goals allowed
  mutate(Goals_Allowed = case_when(Home_Away == "Home" ~ Away_Score
                                   , Home_Away == "Away" ~ Home_Score)
         , TklW_Percent = TklW_Tackles/Tkl_Tackles) |>
  select(League, Match_Date # let's keep some identifying data here
         , TklW_Percent, Def_3rd_Tackles, Mid_3rd_Tackles, Att_3rd_Tackles
         , Att_Challenges, Tkl_percent_Challenges
         , Sh_Blocks, Pass_Blocks
         , Int, Clr, Err
         , Goals_Allowed)

# Let's also look at the distribution of the response variable, Goals_Allowed.
ggplot(data = defense_model_data) +
  geom_histogram(aes(x = Goals_Allowed)
                 , binwidth = 0.5
                 , color = "#000000", fill = "#0099F8") +
  scale_x_continuous(breaks = seq(from = -1, to = 10, by = 1)) +
  labs(title = "Distribution of goals allowed"
       , x = "Goals allowed")
# A quick glance at the histogram of goals allowed shows that the data are positively skewed. This could inhibit our ability to predict that variable, so let's see if we can reduce the skewness.
moments::skewness(defense_model_data$Goals_Allowed) # 0.98
# Let's try applying a log transformation to see if we can reduce the skewness.
ggplot(data = defense_model_data |>
                    mutate(ln_Goals_Allowed = log(Goals_Allowed+1))) +
  geom_histogram(aes(x = ln_Goals_Allowed)
                 # , binwidth = 0.5
                 , color = "#000000", fill = "#0099F8") +
  # scale_x_continuous(breaks = seq(from = -1, to = 10, by = 1)) +
  labs(title = "Distribution of ln(goals allowed)"
       , x = "ln(goals allowed)")
# It's not clearly better by just looking at this, so let's check the skewness.
moments::skewness(log(defense_model_data$Goals_Allowed + 1))
# The skewness is definitely closer to 0, -0.07 but I'm still not sure if this is the right call. Let's add this log-transformed variable to our dataframe and decide later whether or not to use it.
defense_model_data <-
  defense_model_data |>
  mutate(ln_Goals_Allowed = log(Goals_Allowed + 1))



# ==== Build a model ====
# There's plenty more we can explore with the data but let's first build the structure of a simple model.


# Let's use n-fold cross-validation so that we can test out some hyperparameter values on out-of-sample data. Let's also stratify by the outcome variable.

defense_folds <- vfold_cv(defense_model_data
                          , v = 5
                          , strata = Goals_Allowed)

# create the recipe
defense_recipe <-
  recipe(Goals_Allowed ~ .
         , data = defense_model_data) |>
  # make sure ln_goals_allowed is not a predictor
  update_role(ln_Goals_Allowed, new_role = "alt_outcome") |>
  # make sure the League and Match_Date columns are not treated as predictors
  update_role(League, new_role = "league") |>
  update_role(Match_Date, new_role = "match_date")

num_predictors_defense <-
  defense_recipe$var_info |>
  filter(role == "predictor") |>
  nrow()

# specify the model
defense_model <-
  boost_tree(mtry = tune()
             , trees = 400 # let's semi-arbitrarily say 400 is just right
             , min_n = tune() 
             , tree_depth = tune()
             , learn_rate = tune() # eta
             , loss_reduction = tune() # gamma
             , sample_size = tune()) |>
  set_engine("xgboost") |>
  set_mode("regression")

defense_workflow <-
  workflow() |>
  add_recipe(defense_recipe) |>
  add_model(defense_model)

defense_eval_metrics <- metric_set(mae)

# Set up some ranges for tuning parameters
defense_params <-
  defense_workflow |>
  extract_parameter_set_dials() |>
  update(mtry = mtry(c(3, round(0.8*num_predictors_defense)))) |>
  update(tree_depth = tree_depth(c(2, 8))) |>
  update(learn_rate = learn_rate(c(0, 0.2)))

defense_start_grid <-
  defense_params |>
  grid_max_entropy(size = 64)

defense_initial <-
  defense_workflow |>
  tune_grid(resamples = defense_folds
            , grid = defense_start_grid
            , metrics = defense_eval_metrics)

ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 20L)

xgb_sa <-
  defense_workflow |>
  tune_sim_anneal(
    resamples = defense_folds,
    metrics = defense_eval_metrics,
    initial = defense_initial,
    param_info = defense_params,
    iter = 200,
    control = ctrl_sa
  )

show_best(xgb_sa
          , metric = "mae")
autoplot(xgb_sa, type = "performance")
autoplot(xgb_sa, type = "parameters")

# Let's use the second best by performance but with simpler structure (larger loss_reduction value). out of sample mean(mae) = 0.000808
defense_tuned_params <-
  tibble(mtry = 9
         , min_n = 3
         , tree_depth = 7
         , learn_rate = 10^-1.01
         , loss_reduction = 0.0005
         , sample_size = 0.835)

defense_final_workflow <-
  defense_workflow |>
  finalize_workflow(defense_tuned_params)

defense_final_fit <-
  defense_final_workflow |>
  fit(defense_model_data)

# ==== Explore the model ====

# We can examine variable importance to get an idea of which variables make the biggest impact on preventing goals.
defense_final_fit |>
  pull_workflow_fit() |>
  vip(geom = "point")
# It's clear from this plot that tackles-won percent, tackle-percent-challenges, and clearances are the most important stats to predicting goals allowed.


