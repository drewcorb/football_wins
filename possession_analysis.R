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


