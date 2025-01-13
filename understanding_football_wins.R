# Understanding football wins

library(dplyr)
library(readr)
library(tidyr) # pivot_wider
library(stringr) # str_replace_all
library(ggplot2)
library(corrplot)

library(lubridate)
library(tidymodels)
library(xgboost)
library(finetune) # control_sim_anneal
library(rsample) # sliding_period (among others)
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
         , Away_Goal_Scorers = Away_Goals)
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
  # let's clean up a few more column names. get rid of spaces and some symbols
  rename_with(.fn = ~ str_replace_all(., " ", "_")) |>
  rename_with(.fn = ~ str_replace_all(., "\\+", "_plus_")) |>
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

# As far as predictor variables, let's start looking at defensive variables.
# First examine the correlation between the variables
all_match_data |>
  # select only the variables that are associated with defense
  select(contains("Tackles") | contains("Challenges") | contains("Blocks") |
              contains("Int") | contains("Clr") | contains("Err")) |>
  # For now, 
  select(-Home_Tkl_Tackles
         , - Home_Blocks_Blocks
         , -Home_Tkl_Challenges, -Home_Lost_Challenges
         , -Home_Tkl_plus_Int
         ) |>
  cor() |> # compute correlation 
  corrplot(col = colorRampPalette(c("#91CBD765", "#CA225E"))(200), tl.col = "black", method = "ellipse")

# We have data from 7160 matches. Notice that some columns seem obviously redundant. {home_away}_Tkl_Tackles
