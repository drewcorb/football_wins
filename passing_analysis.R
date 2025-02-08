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

passing_data <-
  read_csv(file = "GitHub/football_wins/passing_data.csv") |>
  # let's clean up a few more column names. get rid of spaces and some symbols
  rename_with(.fn = ~ str_replace_all(., " ", "_")) |>
  rename_with(.fn = ~ str_replace_all(., "\\+", "_plus_"))

# ==== Data exploration ====
# Let's assume passing is more descriptive of offense, so let's use the passing stats to predict goals scored for each team.
passing_data <-
  passing_data |>
  mutate(Goals_Scored = case_when(Home_Away == "Home" ~
                                       Home_Score
                                     , Home_Away == "Away" ~
                                       Away_Score))
# First let's take a look at the distribution of outcomes. This will be a symmetric distribution but that is fine for just exploring now.
passing_data |>
  count(Goals_Scored)

# Let's take a more visual look at the distribution of the response variable, Goals_Allowed.
ggplot(data = passing_data) +
  geom_histogram(aes(x = Goals_Scored)
                 , binwidth = 0.5
                 , color = "#000000", fill = "#0099F8") +
  scale_x_continuous(breaks = seq(from = -1, to = 10, by = 1)) +
  labs(title = "Distribution of goals scored"
       , x = "Goals")

# First examine the correlation between the variables. At this point we won't yet need to work with the wide data, so let's select the defensive variables from defense data.
passing_exploration <-
  passing_data |>
  # select only the variables that are associated with passing
  select(Cmp_Total:PrgP)

passing_exploration |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# The correlation plot shows that there is strong correlation between many of these variables. Let's examine them in sequence and see if we can either neglect some variables or create new ones that are less correlated.
passing_exploration |>
  select(
    # Cmp_Total and Att_Total are highly correlated, and we already have Cmp_percent_Total. So we need only 2 of these 3 variables to contain all the information here. Drop one of them.
    -Att_Total
    # We actually see similar redundancy with completions by distance
    , -Att_Short, -Att_Medium, -Att_Long
    # We also should remove Ast, since this is basically a proxy for how many goals were scored. I don't think we'll learn much about which passing stats are important if this is included in the model. We could even use Ast as the target variable if we wanted, but let's keep using Goals_Scored for now.
    , -Ast
  ) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# There is actually less correlation in a lot of these variables than I would've expected. Let's progress to building a model, and then perhaps we can learn more from the model's properties.
# ==== Build a model ====

passing_predictors <-
  passing_exploration |>
  select(
    -Att_Total
    , -Att_Short, -Att_Medium, -Att_Long
    , -Ast, -xAG, -xA
  ) |>
  colnames()

num_predictors_passing <- length(passing_predictors)

passing_model_data <-
  passing_data |>
  select(all_of(passing_predictors), Goals_Scored)
# There's plenty more we can explore with the data but let's first build the structure of a simple model.

# Let's use n-fold cross-validation so that we can test out some hyperparameter values on out-of-sample data. Let's also stratify by the outcome variable.

passing_folds <- vfold_cv(passing_model_data
                             , v = 5
                             , strata = Goals_Scored)

# create the recipe
passing_recipe <-
  recipe(Goals_Scored ~ .
         , data = passing_model_data)
# make sure the League and Match_Date columns are not treated as predictors
# update_role(League, new_role = "league") |>
# update_role(Match_Date, new_role = "match_date")

# specify the model
passing_model <-
  boost_tree(mtry = tune()
             , trees = 400 # let's semi-arbitrarily say 400 is just right
             , min_n = tune() 
             , tree_depth = tune()
             , learn_rate = tune() # eta
             , loss_reduction = tune() # gamma
             , sample_size = tune()) |>
  set_engine("xgboost") |>
  set_mode("regression")

passing_workflow <-
  workflow() |>
  add_recipe(passing_recipe) |>
  add_model(passing_model)

passing_eval_metrics <- metric_set(mae)

# Set up some ranges for tuning parameters
passing_params <-
  passing_workflow |>
  extract_parameter_set_dials() |>
  update(mtry = mtry(c(3, round(0.8*num_predictors_passing)))) |>
  update(tree_depth = tree_depth(c(2, 6))) |>
  update(learn_rate = learn_rate(c(-5, -1))) # 10^-5 to 10^-1

passing_start_grid <-
  passing_params |>
  grid_space_filling(size = 64)

passing_initial <-
  passing_workflow |>
  tune_grid(resamples = passing_folds
            , grid = passing_start_grid
            , metrics = passing_eval_metrics)

ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 20L)

passing_sa <-
  passing_workflow |>
  tune_sim_anneal(
    resamples = passing_folds,
    metrics = passing_eval_metrics,
    initial = passing_initial,
    param_info = passing_params,
    iter = 200,
    control = ctrl_sa
  )

show_best(passing_sa
          , metric = "mae")
autoplot(passing_sa, type = "performance")
autoplot(passing_sa, type = "parameters")

# There are several hyperparameter sets with equivalent out of sample mean(mae) = 0.89. Let's use one that is minimizes depth of each tree (depth of 5 instead of 6) and also has a higher minimum loss reduction.
passing_tuned_params <-
  tibble(mtry = 12
         , min_n = 13
         , tree_depth = 5
         , learn_rate = 0.00531
         , loss_reduction = 6.61e-10
         , sample_size = 0.110)

passing_final_workflow <-
  passing_workflow |>
  finalize_workflow(passing_tuned_params)

passing_final_fit <-
  passing_final_workflow |>
  fit(passing_model_data)

# Before moving on, let's view the performance of the model.
train_performance <- # performance over the training set
  predict(passing_final_fit, passing_model_data) |>
  bind_cols(passing_model_data |> select(Goals_Scored)) |>
  mae(Goals_Scored, .pred)
train_performance

# also record the average mae across all the resamples, which will measure the model's out-of-sample performance
resample_metrics <-
  fit_resamples(object = passing_final_workflow
                , resamples = passing_folds
                , metrics = metric_set(mae))
split_performance <- collect_metrics(resample_metrics)
split_performance

passing_model_performance <-
  tibble(model_iteration = 1
         , train_mae = pull(train_performance, .estimate)
         , split_mae = pull(split_performance, mean))

# ==== Explore the model ====

# We can examine variable importance to get an idea of which variables make the biggest impact on goal difference. Let's start by plotting the gain attributed to each variable.
passing_final_fit |>
  extract_fit_parsnip() |>
  vip(geom = "point"
      # , method = "permute"
      # , train = defense_model_data
      # , target = "Goals_Allowed"
      # , metric = "RMSE"
      # , pred_wrapper = predict
  )


passing_model_data |>
  select(Touches_Touches, Att_Pen_Touches, Carries_Carries, CPA_Carries) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# Clearly there is strong correlation between these variables, but most surprising to me (perhaps it shouldn't be) is the strongest correlation between total touches and total carries. That does make sense, but I suppose I just expected the strongest correlations to be within the same type of statistic (between total touches and penalty touches, and between total carries and penalty carries). I think there will be some opportunity to consolidate some variables and let others emerge as strong predictors too. But let's continue this analysis, and then come back to some different types of tweaks.

# Let's continue our investigation of passing parameters.

passing_explainer <-
  explain_tidymodels(
    model = passing_final_fit
    , data = passing_model_data |> select(-Goals_Scored)
    , y = passing_model_data$Goals_Scored
    , label = "passing xgboost model"
    , verbose = FALSE
  )

# We can compute a type of feature importance in which we shuffle the values of a feature amongst the observations and then predict the target variable, then compare to what degree the model performance is affected.
permute_vip <- model_parts(passing_explainer
                           , loss_function = loss_root_mean_square)

# write a custom function that will help with 
ggplot_importance <- function(...) {
  obj <- list(...)
  metric_name <- attr(obj[[1]], "loss_name")
  metric_lab <- paste(metric_name
                      , "after permutations\n(higher indicates more important)")
  
  full_vip <-
    bind_rows(obj) |>
    filter(variable != "_baseline_")
  
  perm_vals <-
    full_vip |>
    filter(variable == "_full_model_") |>
    group_by(label) |>
    summarise(dropout_loss = mean(dropout_loss))
  
  p <-
    full_vip |>
    filter(variable != "_full_model_") |>
    mutate(variable = forcats::fct_reorder(variable, dropout_loss)) |>
    ggplot(aes(dropout_loss, variable))
  
  if (length(obj) > 1) {
    p <- p +
      facet_wrap(vars(label)) +
      geom_vline(data = perm_vals
                 , aes(xintercept = dropout_loss
                       , color = label)
                 , linewidth = 1.4, lty = 2, alpha = 0.7)
  } else {
    p <- p +
      geom_vline(data = perm_vals
                 , aes(xintercept = dropout_loss)
                 , linewidth = 1.4, lty = 2, alpha = 0.7) +
      geom_boxplot(fill = "#91CBD765", alpha = 0.4)
  }
  
  p +
    theme(legend.position = "none") +
    labs(x = metric_lab, y = NULL
         , fill = NULL, color = NULL)
}

ggplot_importance(permute_vip)
# Here we have key passes (KP) as the variable that dominates all others in importance. This refers to a pass that led directly to a shot. This doesn't seem too far off from expected assists or expected assist-goals, and I worry that by including it as a predictor variable prevents us from learning about the smaller things that lead to these opportunities. After seeing this, I think I'd like to create a model that uses those other statistics to predict key passes as an outcome. So let's go ahead and do that.

# ==== Build a second model (key passes) ====
# First let's take a look at the distribution of key passes.
passing_data |>
  ggplot() +
  geom_histogram(aes(x = KP))
# It's a little skewed to the right, but overall it's a pretty symmetric looking distribution. That should be helpful for building a good model.

KP_predictors <-
  passing_exploration |>
  select(
    -Att_Total
    , -Att_Short, -Att_Medium, -Att_Long
    , -Ast, -xAG, -xA
    , -KP
  ) |>
  colnames()

num_predictors_KP <- length(KP_predictors)

KP_model_data <-
  passing_data |>
  select(all_of(passing_predictors), KP)
# There's plenty more we can explore with the data but let's first build the structure of a simple model.

# Let's use n-fold cross-validation so that we can test out some hyperparameter values on out-of-sample data. Let's also stratify by the outcome variable.

KP_folds <- vfold_cv(KP_model_data
                          , v = 5
                          , strata = KP)

# create the recipe
KP_recipe <-
  recipe(KP ~ .
         , data = KP_model_data)
# make sure the League and Match_Date columns are not treated as predictors
# update_role(League, new_role = "league") |>
# update_role(Match_Date, new_role = "match_date")

# specify the model
KP_model <-
  boost_tree(mtry = tune()
             , trees = tune() # let's semi-arbitrarily say 400 is just right
             , min_n = tune() 
             , tree_depth = tune()
             , learn_rate = tune() # eta
             , loss_reduction = tune() # gamma
             , sample_size = tune()) |>
  set_engine("xgboost") |>
  set_mode("regression")

KP_workflow <-
  workflow() |>
  add_recipe(KP_recipe) |>
  add_model(KP_model)

KP_eval_metrics <- metric_set(mae)

# Set up some ranges for tuning parameters
KP_params <-
  KP_workflow |>
  extract_parameter_set_dials() |>
  update(mtry = mtry(c(3, round(0.8*num_predictors_KP)))) |>
  update(trees = trees(range = c(200, 800))) |>
  update(tree_depth = tree_depth(c(2, 6))) |>
  update(learn_rate = learn_rate(c(-5, -1))) # 10^-5 to 10^-1

KP_start_grid <-
  KP_params |>
  grid_space_filling(size = 64)

KP_initial <-
  KP_workflow |>
  tune_grid(resamples = KP_folds
            , grid = KP_start_grid
            , metrics = KP_eval_metrics)

ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 20L)

KP_sa <-
  KP_workflow |>
  tune_sim_anneal(
    resamples = KP_folds,
    metrics = KP_eval_metrics,
    initial = KP_initial,
    param_info = KP_params,
    iter = 200,
    control = ctrl_sa
  )

show_best(KP_sa
          , metric = "mae"
          , n = 10)
autoplot(KP_sa, type = "performance")
autoplot(KP_sa, type = "parameters")

# There are several hyperparameter sets with equivalent out of sample mean(mae) = 2.43. Let's use one that is minimizes depth of each tree (depth of 4 instead of 5 or 6) and also has a smaller number of trees.
KP_tuned_params <-
  tibble(mtry = 10
         , trees = 655
         , min_n = 21
         , tree_depth = 4
         , learn_rate = 0.00842
         , loss_reduction = 4.57e-5
         , sample_size = 0.113)

KP_final_workflow <-
  KP_workflow |>
  finalize_workflow(KP_tuned_params)

KP_final_fit <-
  KP_final_workflow |>
  fit(KP_model_data)

# Before moving on, let's view the performance of the model.
train_performance <- # performance over the training set
  predict(KP_final_fit, KP_model_data) |>
  bind_cols(KP_model_data |> select(KP)) |>
  mae(KP, .pred)
train_performance

# also record the average mae across all the resamples, which will measure the model's out-of-sample performance
resample_metrics <-
  fit_resamples(object = KP_final_workflow
                , resamples = KP_folds
                , metrics = metric_set(mae))
split_performance <- collect_metrics(resample_metrics)
split_performance

KP_model_performance <-
  tibble(model_iteration = 1
         , train_mae = pull(train_performance, .estimate)
         , split_mae = pull(split_performance, mean))

# ==== Model exploration (KP model) ====

# We can examine variable importance to get an idea of which variables make the biggest impact on goal difference. Let's start by plotting the gain attributed to each variable.
KP_final_fit |>
  extract_fit_parsnip() |>
  vip(geom = "point"
      # , method = "permute"
      # , train = defense_model_data
      # , target = "Goals_Allowed"
      # , metric = "RMSE"
      # , pred_wrapper = predict
  )
# Here we see PPA (passes into the penalty area) as by far the most important predictor variable for key passes, followed by PrgP and then the rest of the variables grouped together behind these two.

KP_model_data |>
  select(PPA, PrgP, Final_Third, PrgDist_Total, Cmp_percent_Total) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# Here we see a strong correlation between PrgP and Final_Third, also a strong correlation between PrgP and PPA. I actually expected a littler more correlation beween PPA and Final_Third, and although they're clearly correlated, they must be carrying more different information from each other than I thought.

# Let's continue our investigation of KP parameters.

KP_explainer <-
  explain_tidymodels(
    model = KP_final_fit
    , data = KP_model_data |> select(-KP)
    , y = KP_model_data$KP
    , label = "KP xgboost model"
    , verbose = FALSE
  )

# We can compute a type of feature importance in which we shuffle the values of a feature amongst the observations and then predict the target variable, then compare to what degree the model performance is affected.
permute_vip <- model_parts(KP_explainer
                           , loss_function = loss_root_mean_square)

ggplot_importance(permute_vip)
# This plot displays similar information to our initial variable importance plot. Once again PPA is the most important variable by far, followed by PrgP and then the rest of the variables grouped relatively close together.

