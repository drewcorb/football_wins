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
library(DALEXtra) # model explanation

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

possession_data <-
  read_csv(file = "GitHub/football_wins/possession_data.csv") |>
  # let's clean up a few more column names. get rid of spaces and some symbols
  rename_with(.fn = ~ str_replace_all(., " ", "_")) |>
  rename_with(.fn = ~ str_replace_all(., "\\+", "_plus_"))

# ==== Data exploration ====
# I don't really know whether possession impacts goals scored or goals allowed more significantly. Thus, let's start by examining its effect on goal difference.
possession_data <-
  possession_data |>
  mutate(Goal_Difference = case_when(Home_Away == "Home" ~
                                       Home_Score - Away_Score
                                     , Home_Away == "Away" ~
                                       Away_Score - Home_Score))
# First let's take a look at the distribution of outcomes. This will be a symmetric distribution but that is fine for just exploring now.
possession_data |>
  count(Goal_Difference)

# Let's take a more visual look at the distribution of the response variable, Goals_Allowed.
ggplot(data = possession_data) +
  geom_histogram(aes(x = Goal_Difference)
                 , binwidth = 0.5
                 , color = "#000000", fill = "#0099F8") +
  scale_x_continuous(breaks = seq(from = -1, to = 10, by = 1)) +
  labs(title = "Distribution of goal difference"
       , x = "Goal difference")

# First examine the correlation between the variables. At this point we won't yet need to work with the wide data, so let's select the defensive variables from defense data.
possession_exploration <-
  possession_data |>
  # select only the variables that are associated with possession
  select(Touches_Touches:PrgR_Receiving)

possession_exploration |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# The correlation plot shows that there is strong correlation between many of these variables. Let's examine them in sequence and see if we can either neglect some variables or create new ones that are less correlated.
possession_exploration |>
  select(
    # Touches_Touches and Live_Touches are practically identical to each other. We can safely remove one of them. Let's remove Live_Touches, since that statistic neglects various types of touches during non-live balls.
    -Live_Touches
    # There still remain many Touches variables. I wonder if touches in the penalty zones are the most consequential, so I am inclined to keep those. Let's also keep the total number of touches, but get rid of the columns divided into thirds
    , -contains("3rd")
    # Succ_Take_Ons = Att_Take_Ons * Succ_percent_Take_Ons, so it contains no unique information
    , -Succ_Take_Ons
    # total passes received is strongly correlated with many variables, and we have another variable that focuses on (perhaps the more relevant) progressive passes received.
    , -Rec_Receiving
    ) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")


# There are still a lot of correlated variables here, but with my lack of subject matter knowledge, I'm concerned about doing any more selecting. Let's progress to building a model, and then perhaps we can learn more from the model's properties.

possession_predictor_variables <-
  possession_exploration |>
  select(
    -Live_Touches
    , -contains("3rd")
    , -Succ_Take_Ons
    , -Rec_Receiving
  ) |>
  colnames()

possession_model_data <-
  possession_data |>
  select(all_of(possession_predictor_variables), Goal_Difference)

# ==== Build a model ====

# There's plenty more we can explore with the data but let's first build the structure of a simple model.

# Let's use n-fold cross-validation so that we can test out some hyperparameter values on out-of-sample data. Let's also stratify by the outcome variable.

possession_folds <- vfold_cv(possession_model_data
                          , v = 5
                          , strata = Goal_Difference)

# create the recipe
possession_recipe <-
  recipe(Goal_Difference ~ .
         , data = possession_model_data)
  # make sure the League and Match_Date columns are not treated as predictors
  # update_role(League, new_role = "league") |>
  # update_role(Match_Date, new_role = "match_date")

num_predictors_possession <-
  possession_recipe$var_info |>
  filter(role == "predictor") |>
  nrow()

# specify the model
possession_model <-
  boost_tree(mtry = tune()
             , trees = 400 # let's semi-arbitrarily say 400 is just right
             , min_n = tune() 
             , tree_depth = tune()
             , learn_rate = tune() # eta
             , loss_reduction = tune() # gamma
             , sample_size = tune()) |>
  set_engine("xgboost") |>
  set_mode("regression")

possession_workflow <-
  workflow() |>
  add_recipe(possession_recipe) |>
  add_model(possession_model)

possession_eval_metrics <- metric_set(mae)

# Set up some ranges for tuning parameters
possession_params <-
  possession_workflow |>
  extract_parameter_set_dials() |>
  update(mtry = mtry(c(3, round(0.8*num_predictors_possession)))) |>
  update(tree_depth = tree_depth(c(2, 6))) |>
  update(learn_rate = learn_rate(c(-5, -1))) # 10^-5 to 10^-1

possession_start_grid <-
  possession_params |>
  grid_max_entropy(size = 64)

possession_initial <-
  possession_workflow |>
  tune_grid(resamples = possession_folds
            , grid = possession_start_grid
            , metrics = possession_eval_metrics)

ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 20L)

xgb_sa <-
  possession_workflow |>
  tune_sim_anneal(
    resamples = possession_folds,
    metrics = possession_eval_metrics,
    initial = possession_initial,
    param_info = possession_params,
    iter = 200,
    control = ctrl_sa
  )

show_best(xgb_sa
          , metric = "mae")
autoplot(xgb_sa, type = "performance")
autoplot(xgb_sa, type = "parameters")

# There are several hyperparameter sets with equivalent out of sample mean(mae) =1.32. Let's use one that is minimizes complexity of each tree, with tree_depth = 4 (as opposed to others that are 5 or 6).
possession_tuned_params <-
  tibble(mtry = 12
         , min_n = 28
         , tree_depth = 4
         , learn_rate = 0.0174
         , loss_reduction = 0.165
         , sample_size = 0.159)

possession_final_workflow <-
  possession_workflow |>
  finalize_workflow(possession_tuned_params)

possession_final_fit <-
  possession_final_workflow |>
  fit(possession_model_data)


