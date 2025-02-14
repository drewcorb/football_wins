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
# ==== Build a model ====

possession_predictors <-
  possession_exploration |>
  select(
    -Live_Touches
    , -contains("3rd")
    , -Succ_Take_Ons
    , -Rec_Receiving
  ) |>
  colnames()

num_predictors_possession <- length(possession_predictors)

possession_model_data <-
  possession_data |>
  select(all_of(possession_predictors), Goal_Difference)
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

# Before moving on, let's record the performance of the model.
train_performance <- # performance over the training set
  predict(possession_final_fit, possession_model_data) |>
  bind_cols(possession_model_data |> select(Goal_Difference)) |>
  mae(Goal_Difference, .pred)
train_performance

# also record the average mae across all the resamples, which will measure the model's out-of-sample performance
resample_metrics <-
  fit_resamples(object = possession_final_workflow
                , resamples = possession_folds
                , metrics = metric_set(mae))
split_performance <- collect_metrics(resample_metrics)
split_performance

possession_model_performance <-
  tibble(model_iteration = 1
         , train_mae = pull(train_performance, .estimate)
         , split_mae = pull(split_performance, mean))

# ==== Explore the model ====

# We can examine variable importance to get an idea of which variables make the biggest impact on goal difference. Let's start by plotting the gain attributed to each variable.
possession_final_fit |>
  pull_workflow_fit() |>
  vip(geom = "point"
      # , method = "permute"
      # , train = defense_model_data
      # , target = "Goals_Allowed"
      # , metric = "RMSE"
      # , pred_wrapper = predict
  )
# Wow! I wasn't sure if any variables would stand out too much, but clear we have a few stats here that jump out above the others: touches in the attacking penalty area, carries into the penalty area, total touches, and total carries. This really indicates to me that it might be preferable to cast these instead as total number of touches, percent of of touches in attacking penalty area, total number of carries, and percent of carries in the attacking penalty area, because that way we could decrease the correlation between the variables. Let's take a quick look at the correlation between these 4 variables.

possession_model_data |>
  select(Touches_Touches, Att_Pen_Touches, Carries_Carries, CPA_Carries) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# Clearly there is strong correlation between these variables, but most surprising to me (perhaps it shouldn't be) is the strongest correlation between total touches and total carries. That does make sense, but I suppose I just expected the strongest correlations to be within the same type of statistic (between total touches and penalty touches, and between total carries and penalty carries). I think there will be some opportunity to consolidate some variables and let others emerge as strong predictors too. But let's continue this analysis, and then come back to some different types of tweaks.

# Let's continue our investigation of possession parameters.

possession_explainer <-
  explain_tidymodels(
    model = possession_final_fit
    , data = possession_model_data |> select(-Goal_Difference)
    , y = possession_model_data$Goal_Difference
    , label = "possession xgboost model"
    , verbose = FALSE
  )

# We can compute a type of feature importance in which we shuffle the values of a feature amongst the observations and then predict the target variable, then compare to what degree the model performance is affected.
permute_vip <- model_parts(possession_explainer
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
# This time the feature importance plot looks sensible, as opposed to when I first created one for the defense variables. I suspected that had to do with leaving non-predictor variables in the data set (League and Match Date), so I omitted those columns here. It looks like that fixed the issue, so either the model_parts() function or the the custom plotting function doesn't handle those types of columns well. Onto the actual analysis though.

# In this variable importance plot created by permuting each variable amongst the observations, one more variable stands out as quite important: PrgR_Receiving, or total number of progressive passes received.

pdp_Att_Pen_Touches <- model_profile(possession_explainer, N = 500, variables = "Att_Pen_Touches")

ggplot_pdp <- function(obj, x) {
  p <-
    as_tibble(obj$agr_profiles) |>
    mutate(`_label_` = stringr::str_remove(`_label_`, "^[^_]*_")) |>
    ggplot(aes(`_x_`, `_yhat_`)) +
    geom_line(data = as_tibble(obj$cp_profiles)
              , aes(x = {{ x }}, group = `_ids_`)
              , linewidth = 0.5, alpha = 0.05, color = "gray50")
  
  num_colors <- n_distinct(obj$agr_profiles$`_label_`)
  
  if (num_colors > 1) {
    p <- p + geom_line(aes(color = `_label_`), linewidth = 1.2, alpha = 0.8)
  } else {
    p <- p + geom_line(color = "midnightblue", linewidth = 1.2, alpha = 0.8)
  }
  
  p
}

ggplot_pdp(pdp_Att_Pen_Touches, Att_Pen_Touches) +
  labs(x = "Att_Pen_Touches", y = "Goal difference"
       , color = NULL)
# Here we see that even committing a single error that leads to an opponent's shot leads to a significant increase in goals allowed, with the effect diminishing as more errors are committed.

pdp_PrgR_Receiving <- model_profile(possession_explainer, N = 1000, variables = "PrgR_Receiving")

ggplot_pdp(pdp_PrgR_Receiving, PrgR_Receiving) +
  labs(x = "PrgR_Receiving", y = "Goal difference"
       , color = NULL)

# ==== Second iteration of possession investigation ====
# We gained information about some of the most important possession statistics toward modeling goal differential, however the most important ones are highly correlated with each other. If we transform/remove some of those, it's possible that other important statistics will emerge.

# Another avenue would be to predict goal difference in terms of Home_Score - Away_Score based on the individual home and away stats. I think the highest performing model will be something along these lines. However it will double the number of initial variables (one of each type for home and away), and I don't feel that I yet understand the game well enough to make intelligent transformations of so many interacting variables.

# So let's start with option 1 and do some simpler transformations of the variables from our previous model.

# Beginning with Touches_Touches and Carries_Carries, let's create a new variable that is the geometric mean of these two:
possession_exploration |>
  select(all_of(possession_predictors)) |>
  mutate(Touches_Carries = (Touches_Touches + Carries_Carries) / 2
         # let's also convert attack penalty touches and CPA_Carries into a similar type of variable, but as a fraction of the one we just defined
         , Touch_Carry_Pen_Pct =
           (Att_Pen_Touches + CPA_Carries) /
           (Touches_Touches + Carries_Carries)
         ) |>
  select(-Touches_Touches, -Att_Pen_Touches
         , -Carries_Carries, -CPA_Carries) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")
  
# We can see that total distance of carries and progressive distance of carries are both highly correlated with several variables. Since we already have a measure of the total number of carries, let's instead convert these to units of distance per carry.

possession_exploration |>
  select(all_of(possession_predictors)) |>
  mutate(Touches_Carries = (Touches_Touches + Carries_Carries) / 2
         # convert attack penalty touches and CPA_Carries into fraction of geometric means
         , Touch_Carry_Pen_Pct =
           (Att_Pen_Touches + CPA_Carries) /
           (Touches_Touches + Carries_Carries)
         # distance and progressive distance per carry
         , Dist_per_Carry = TotDist_Carries/Carries_Carries
         , PrgDist_per_Carry = PrgDist_Carries/Carries_Carries
  ) |>
  select(-Touches_Touches, -Att_Pen_Touches
         , -Carries_Carries, -CPA_Carries
         , -TotDist_Carries, -PrgDist_Carries) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# We see that successful percent take ons and tackled percent take ons are strongly negatively correlated with each other. Because of that, if we were to take the geometric or arithmetic mean of the two variables, then we might lose information about them both (consider the information carried in two lines y1 = x and y2 = -x, vs the information carried in z = -x^2). So let's just keep successful percent take ons, and remove the Tkld percent variable.

possession2_exploration <-
  possession_exploration |>
  select(all_of(possession_predictors)) |>
  mutate(Touches_Carries = sqrt(Touches_Touches + Carries_Carries) / 2
         # convert attack penalty touches and CPA_Carries into fraction of geometric means
         , Touch_Carry_Pen_Pct =
           (Att_Pen_Touches + CPA_Carries) /
           (Touches_Touches + Carries_Carries)
         # distance and progressive distance per carry
         , Dist_per_Carry = TotDist_Carries/Carries_Carries
         , PrgDist_per_Carry = PrgDist_Carries/Carries_Carries
  ) |>
  select(-Touches_Touches, -Att_Pen_Touches
         , -Carries_Carries, -CPA_Carries
         , -TotDist_Carries, -PrgDist_Carries
         , -Tkld_percent_Take_Ons)

possession2_exploration |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# We're down from 16 to 13 variables now, and largely they are less correlated with each other than we began with. Let's build another model to predict Goal_Difference and see how it compares with our first try.

# ==== Build the second model ====

# There's plenty more we can explore with the data but let's first build the structure of a simple model.

possession2_predictors <- colnames(possession2_exploration)

num_predictors_possession2 <- length(possession2_predictors)

possession2_model_data <-
  possession_data |>
  mutate(Touches_Carries = (Touches_Touches + Carries_Carries) / 2
         # convert attack penalty touches and CPA_Carries into fraction of geometric means
         , Touch_Carry_Pen_Pct =
           (Att_Pen_Touches+CPA_Carries) /
           (Touches_Touches+Carries_Carries)
         # distance and progressive distance per carry
         , Dist_per_Carry = TotDist_Carries/Carries_Carries
         , PrgDist_per_Carry = PrgDist_Carries/Carries_Carries
  ) |>
  select(all_of(possession2_predictors), Goal_Difference)
# There's plenty more we can explore with the data but let's first build the structure of a simple model.

# Let's use n-fold cross-validation so that we can test out some hyperparameter values on out-of-sample data. Let's also stratify by the outcome variable.

possession2_folds <- vfold_cv(possession2_model_data
                              , v = 5
                              , strata = Goal_Difference)

# create the recipe
possession2_recipe <-
  recipe(Goal_Difference ~ .
         , data = possession2_model_data)
# make sure the League and Match_Date columns are not treated as predictors
# update_role(League, new_role = "league") |>
# update_role(Match_Date, new_role = "match_date")

# specify the model
possession2_model <-
  boost_tree(mtry = tune()
             , trees = 400 # let's semi-arbitrarily say 400 is just right
             , min_n = tune() 
             , tree_depth = tune()
             , learn_rate = tune() # eta
             , loss_reduction = tune() # gamma
             , sample_size = tune()) |>
  set_engine("xgboost") |>
  set_mode("regression")

possession2_workflow <-
  workflow() |>
  add_recipe(possession2_recipe) |>
  add_model(possession2_model)

possession2_eval_metrics <- metric_set(mae)

# Set up some ranges for tuning parameters
possession2_params <-
  possession2_workflow |>
  extract_parameter_set_dials() |>
  update(mtry = mtry(c(3, round(0.8*num_predictors_possession)))) |>
  update(tree_depth = tree_depth(c(2, 6))) |>
  update(learn_rate = learn_rate(c(-5, -1))) # 10^-5 to 10^-1

possession2_start_grid <-
  possession2_params |>
  grid_max_entropy(size = 64)

possession2_initial <-
  possession2_workflow |>
  tune_grid(resamples = possession2_folds
            , grid = possession2_start_grid
            , metrics = possession2_eval_metrics)

ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 20L)

possession2_sa <-
  possession_workflow |>
  tune_sim_anneal(
    resamples = possession_folds,
    metrics = possession_eval_metrics,
    initial = possession_initial,
    param_info = possession_params,
    iter = 200,
    control = ctrl_sa
  )

show_best(possession2_sa
          , metric = "mae"
          , n = 10)
autoplot(possession2_sa, type = "performance")
autoplot(possession2_sa, type = "parameters")

# There are several hyperparameter sets with equivalent out of sample mean(mae) =1.32. Let's use one that is minimizes complexity of each tree, with tree_depth = 4 (as opposed to others that are 5 or 6).
possession2_tuned_params <-
  tibble(mtry = 9
         , min_n = 29
         , tree_depth = 5
         , learn_rate = 0.0142
         , loss_reduction = 0.0803
         , sample_size = 0.367)

possession2_final_workflow <-
  possession2_workflow |>
  finalize_workflow(possession2_tuned_params)

possession2_final_fit <-
  possession2_final_workflow |>
  fit(possession2_model_data)

# Before moving on, let's record the performance of the model.
train_performance <- # performance over the training set
  predict(possession2_final_fit, possession2_model_data) |>
  bind_cols(possession2_model_data |> select(Goal_Difference)) |>
  mae(Goal_Difference, .pred)
train_performance

# also record the average mae across all the resamples, which will measure the model's out-of-sample performance
resample_metrics <-
  fit_resamples(object = possession2_final_workflow
                , resamples = possession2_folds
                , metrics = metric_set(mae))
split_performance <- collect_metrics(resample_metrics)
split_performance

# The performance of this version of the model is quite similar to the first, (equivalent training mae of 1.28, near-equivalent splits mae of 1.32 and 1.33), despite the fact we reduced our number of variables from 16 to 13. I'm pleased with the fact that we essentially maintained the same level of accuracy while significantly simplifying the model, which hopefully will reduce our chances of overfitting.

possession_model_performance <-
  possession_model_performance |>
  bind_rows(tibble(model_iteration = 2
                   , train_mae = pull(train_performance, .estimate)
                   , split_mae = pull(split_performance, mean))
  )

# ==== Explore the second model ====

# We can examine variable importance to get an idea of which variables make the biggest impact on goal difference. Let's start by plotting the gain attributed to each variable.
possession2_final_fit |>
  pull_workflow_fit() |>
  vip(geom = "point"
      # , method = "permute"
      # , train = defense_model_data
      # , target = "Goals_Allowed"
      # , metric = "RMSE"
      # , pred_wrapper = predict
  )
# How about that?! The two variables that we created for this model (Touch_Carry_Pen_Pct and root_Touches_Carries) by far contribute the most gain to the xgboost model.

possession2_explainer <-
  explain_tidymodels(
    model = possession2_final_fit
    , data = possession2_model_data |> select(-Goal_Difference)
    , y = possession2_model_data$Goal_Difference
    , label = "possession xgboost model"
    , verbose = FALSE
  )

# Now let's examine the feature importance by permuting each feature values amongst the observations..
permute_vip2 <- model_parts(possession2_explainer
                            , loss_function = loss_root_mean_square)

ggplot_importance(permute_vip2)
# The same two variables (Touch_Carry_Pen_Pct and root_Touches_Carries) stand out in this variable importance plot. Of the rest, it appears PrgR_Receiving (the number of progressive passes received) is slightly more important than the others, but it's a very small distinction compared to the top two variables.

# Let's examine the PDPs of our two most important variables.

pdp_Touch_Carry_Pen_Pct <- model_profile(possession2_explainer, N = 500
                                         , variables = "Touch_Carry_Pen_Pct")

ggplot_pdp(pdp_Att_Pen_Touches, Att_Pen_Touches) +
  labs(x = "Touch_Carry_Pen_Pct", y = "Goal difference"
       , color = NULL)
# Here we see that most of the action takes place in the range of 0 to 50% of touches and carries in the attacking penalty area. I think this makes intuitive sense, as it's inconceivable for a team to have more than half of its touches in the attacking penalty area. We can see the break even point (crosses the 0 goal difference line) at about 15% of the action in the attacking penalty area, and the returns diminish (though very slowly) as the percent approaches 50.

pdp_root_Touches_Carries <- model_profile(possession2_explainer, N = 1000, variables = "root_Touches_Carries")

ggplot_pdp(pdp_root_Touches_Carries, root_Touches_Carries) +
  labs(x = "root_Touches_Carries", y = "Goal difference"
       , color = NULL) +
  scale_x_continuous(breaks = seq(from = 100, to = 1000, by = 100))
# Here we see a very well-formed S-curve. Below about 400 touches and carries, it the goal difference doesn't get much worse. Then we see a roughly linear increase in goal difference until about 750 touches and carries before it levels out again. Interestingly, the lower asymptote is only just below 0 goal difference, while the upper asymptote is near a goal difference of 2. I wonder if this suggests that teams with low numbers of touches and carries tend to be defensive teams that both allow and score few goals, but teams that dominate on possession tend to generate large leads. If that is true, then it might be true that a very bad team can try to climb to the middle of the table simply by improving its defense, but in order to reach the top the team must assume a possession/attack focused strategy.

# There is clearly more to explore here, but let's leave possession for the time being. I've already learned a lot from this!