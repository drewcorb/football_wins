# Understanding football wins -- Shooting analysis

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

shooting_data <-
  read_csv(file = "GitHub/football_wins/summary_data.csv") |>
  # let's clean up a few more column names. get rid of spaces and some symbols
  rename_with(.fn = ~ str_replace_all(., " ", "_")) |>
  rename_with(.fn = ~ str_replace_all(., "\\+", "_plus_"))

# ==== Data exploration ====
# Let's use the shooting stats to predict goals scored for each team.
shooting_data <-
  shooting_data |>
  mutate(Goals_Scored = case_when(Home_Away == "Home" ~
                                    Home_Score
                                  , Home_Away == "Away" ~
                                    Away_Score))
# First let's take a look at the distribution of outcomes. This will be a symmetric distribution but that is fine for just exploring now.
shooting_data |>
  count(Goals_Scored)
# Hmm, there are many rows with an NA in goals scored. Let's inspect them to see if we can fill it in or simply remove them.
shooting_data |>
  filter(is.na(Goals_Scored)) |>
  ggplot() +
  geom_histogram(aes(x = Match_Date))
# It's kindof a crude plot, but we can see that many of these matches occurred early in 2020. When I go to FBref to check on them, I see that the matches were cancelled (obviously due to the COVID-19 pandemic). There are a couple of other matches with NAs in Goals_Scored, but overall I feel fine just removing these from the data set.
shooting_data <- shooting_data |> filter(!is.na(Goals_Scored))

# Let's take a more visual look at the distribution of the response variable, Goals_Allowed.
ggplot(data = shooting_data) +
  geom_histogram(aes(x = Goals_Scored)
                 , binwidth = 0.5
                 , color = "#000000", fill = "#0099F8") +
  scale_x_continuous(breaks = seq(from = -1, to = 10, by = 1)) +
  labs(title = "Distribution of goals scored"
       , x = "Goals")

# First examine the correlation between the variables. At this point we won't yet need to work with the wide data, so let's select the variables related to shooting that we haven't already explored in other analyses.
shooting_exploration <-
  shooting_data |>
  # select only the variables that are associated with shooting. And let's ignore Goals and Assists, since those will be so strongly correlated with Goals_Scored. Let's also neglect xG, xAG, etc as we can safely assume those will be strongly correlated with Goals_Scored as well.
  select(PK, PKatt
         , Sh, SoT
         , SCA_SCA, GCA_SCA) |>
  # for some reason we have 26 rows with NA in at least one of these variables, so let's drop those few rows out
  drop_na()

shooting_exploration |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# The correlation plot shows that there is strong correlation between some of these variables, most notably SCA_SCA and Sh. We can probably remove one of those variables completely and still retain almost all information in the model. Let's look at the distributions of those variables to help us make our choice.
shooting_exploration |>
  ggplot() +
  geom_histogram(aes(x = Sh, fill = "Sh"), alpha = 0.5) +
  geom_histogram(aes(x = SCA_SCA, fill = "SCA_SCA"), alpha = 0.5) +
  scale_fill_manual(name = "Statistic",
                    values = c("Sh" = "red", "SCA_SCA" = "blue")) +
  labs(x = "Sh or SCA_SCA")

# We can see that SCA_SCA is a little more widely dispersed and shifted further away from 0 than Sh is. I like the properties of SCA_SCA better, so I'm going to drop Sh from our scoring analysis in favor of SCA_SCA. Penalty kicks are rare events and I'm not sure the best way to handle them, so I'm going to let both PK stats remain in the model and figure out their role/importance that way.

shooting_exploration |>
  select(
    # remove shots as variable
    -Sh
  ) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200)
           , tl.col = "black"
           , method = "ellipse")

# This is going to be a pretty rudimentary model, but let's move forward with it and see what happens.
# ==== Build a model ====

shooting_predictors <-
  shooting_exploration |>
  select(
    -Sh
  ) |>
  colnames()

num_predictors_shooting <- length(shooting_predictors)

shooting_model_data <-
  shooting_data |>
  select(all_of(shooting_predictors), Goals_Scored)
# There's plenty more we can explore with the data but let's first build the structure of a simple model.

# Let's use n-fold cross-validation so that we can test out some hyperparameter values on out-of-sample data. Let's also stratify by the outcome variable.

shooting_folds <- vfold_cv(shooting_model_data
                          , v = 5
                          , strata = Goals_Scored)

# create the recipe
shooting_recipe <-
  recipe(Goals_Scored ~ .
         , data = shooting_model_data)
# make sure the League and Match_Date columns are not treated as predictors
# update_role(League, new_role = "league") |>
# update_role(Match_Date, new_role = "match_date")

# specify the model
shooting_model <-
  boost_tree(mtry = tune()
             , trees = tune()
             , min_n = tune() 
             , tree_depth = tune()
             , learn_rate = tune() # eta
             , loss_reduction = tune() # gamma
             , sample_size = tune()) |>
  set_engine("xgboost") |>
  set_mode("regression")

shooting_workflow <-
  workflow() |>
  add_recipe(shooting_recipe) |>
  add_model(shooting_model)

shooting_eval_metrics <- metric_set(mae)

# Set up some ranges for tuning parameters
shooting_params <-
  shooting_workflow |>
  extract_parameter_set_dials() |>
  update(mtry = mtry(c(3, round(0.8*num_predictors_shooting)))) |>
  update(trees = trees(c(200, 800))) |>
  update(tree_depth = tree_depth(c(2, 6))) |>
  update(learn_rate = learn_rate(c(-5, -1))) # 10^-5 to 10^-1

shooting_start_grid <-
  shooting_params |>
  grid_space_filling(size = 64)

shooting_initial <-
  shooting_workflow |>
  tune_grid(resamples = shooting_folds
            , grid = shooting_start_grid
            , metrics = shooting_eval_metrics)

ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 20L)

shooting_sa <-
  shooting_workflow |>
  tune_sim_anneal(
    resamples = shooting_folds,
    metrics = shooting_eval_metrics,
    initial = shooting_initial,
    param_info = shooting_params,
    iter = 200,
    control = ctrl_sa
  )

show_best(shooting_sa
          , metric = "mae"
          , n = 10)
autoplot(shooting_sa, type = "performance")
autoplot(shooting_sa, type = "parameters")

# There are several hyperparameter sets with equivalent out of sample mean(mae) = 0.19. Let's use the best performing set.
shooting_tuned_params <-
  tibble(mtry = 4
         , trees = 332
         , min_n = 31
         , tree_depth = 6
         , learn_rate = 0.00808
         , loss_reduction = 0.00842
         , sample_size = 0.411)

shooting_final_workflow <-
  shooting_workflow |>
  finalize_workflow(shooting_tuned_params)

shooting_final_fit <-
  shooting_final_workflow |>
  fit(shooting_model_data)

# Before moving on, let's view the performance of the model.
train_performance <- # performance over the training set
  predict(shooting_final_fit, shooting_model_data) |>
  bind_cols(shooting_model_data |> select(Goals_Scored)) |>
  mae(Goals_Scored, .pred)
train_performance

# also record the average mae across all the resamples, which will measure the model's out-of-sample performance
resample_metrics <-
  fit_resamples(object = shooting_final_workflow
                , resamples = shooting_folds
                , metrics = metric_set(mae))
split_performance <- collect_metrics(resample_metrics)
split_performance

shooting_model_performance <-
  tibble(model_iteration = 1
         , train_mae = pull(train_performance, .estimate)
         , split_mae = pull(split_performance, mean))

# ==== Explore the model ====

# We can examine variable importance to get an idea of which variables make the biggest impact on goal difference. Let's start by plotting the gain attributed to each variable.
shooting_final_fit |>
  extract_fit_parsnip() |>
  vip(geom = "point"
      # , method = "permute"
      # , train = defense_model_data
      # , target = "Goals_Allowed"
      # , metric = "RMSE"
      # , pred_wrapper = predict
  )

# Clearly GCA_SCA contains too much information about goals being scored here. We'll want to build a new model removing that variable. At the very least, we can see that SoT (shots on target) also is a key metric.

# Let's continue our investigation of shooting parameters.

shooting_explainer <-
  explain_tidymodels(
    model = shooting_final_fit
    , data = shooting_model_data |> select(-Goals_Scored)
    , y = shooting_model_data$Goals_Scored
    , label = "shooting xgboost model"
    , verbose = FALSE
  )

# We can compute a type of feature importance in which we shuffle the values of a feature amongst the observations and then predict the target variable, then compare to what degree the model performance is affected.
permute_vip <- model_parts(shooting_explainer
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
