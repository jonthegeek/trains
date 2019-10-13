library(recipes)
library(textrecipes)
library(textstem)
library(tune)
library(dials)
library(parsnip)

# I want to reserve the trains data that mentions trains for something later.
set.seed(42)

trains_mentions_trains <- trains_data %>%
  dplyr::filter(mentions_train) %>%
  # I use sample_frac to scramble the order.
  dplyr::sample_frac() %>%
  dplyr::select(-mentions_train)
trains_no_trains <- trains_data %>%
  dplyr::filter(!mentions_train) %>%
  dplyr::sample_frac() %>%
  dplyr::select(-mentions_train)

trains_split <- rsample::initial_split(
  trains_no_trains,
  strata = label
)
trains_train <- rsample::training(trains_split)
trains_test <- rsample::testing(trains_split)

trains_folds <- rsample::vfold_cv(
  trains_train,
  strata = label
)

trains_recipe <- trains_train %>%
  recipes::recipe(label ~ sentence) %>%
  textrecipes::step_tokenize(sentence) %>%
  textrecipes::step_stem(
    sentence,
    custom_stemmer = textstem::lemmatize_words
  ) %>%
  textrecipes::step_stopwords(sentence) %>%
  textrecipes::step_tokenfilter(
    sentence,
    # I don't know how many tokens to use. Let's let tune figure that out!
    max_tokens = tune::tune()
    # max_tokens = 100
  ) %>%
  textrecipes::step_tf(sentence)

trains_model <- parsnip::boost_tree(
  mode = "classification",
  # We'll let tune figure out all the things!
  mtry = tune::tune(),
  trees = tune::tune(),
  min_n = tune::tune(),
  tree_depth = tune::tune(),
  learn_rate = tune::tune(),
  loss_reduction = tune::tune(),
  sample_size = tune::tune()
) %>%
  parsnip::set_engine("xgboost")

trains_workflow <- tune::workflow() %>%
  tune::add_model(trains_model) %>%
  tune::add_recipe(trains_recipe)

trains_parameter_set <- dials::param_set(trains_workflow) %>%
  # I checked, and, when I wrote this, there were a total of 558 unique
  # non-stop-words in the data. Looking at most at the top-100 of those seems
  # fine, so let's not let it go above that.
  update(
    max_tokens = dials::num_terms(c(10, 100)),
    mtry = dials::mtry_long(c(0, 3)),
    sample_size = dials::sample_prop(0:1)
  )
# Per the example I looked at, I let everything else stick with default. I don't
# know if that's a great idea or not.

# all_cores <- parallel::detectCores(logical = FALSE)
# library(doParallel)
# cl <- makePSOCKcluster(all_cores)
# registerDoParallel(cl)

trains_grid <- dials::grid_latin_hypercube(trains_parameter_set, size = 30)
saveRDS(trains_grid, "trains_grid.rds")

trains_search_result <- tune::tune_grid(
  trains_workflow,
  rs = trains_folds,
  grid = trains_grid
  # param_info = trains_parameter_set
  # initial = 5,
  # iter = 30,
  # perf = yardstick::metric_set(yardstick::roc_auc)
  # control = tune::Bayes_control(verbose = TRUE)
)
saveRDS(trains_search_result, "trains_search_result.rds")

# parallel::stopCluster(cl)

first_tune <- tune::estimate(trains_search_result) %>%
  dplyr::arrange(desc(mean)) %>%
  dplyr::slice(1)

second_tune <- tune::estimate(trains_search_result) %>%
  dplyr::filter(.metric == "kap") %>%
  dplyr::arrange(desc(mean)) %>%
  dplyr::slice(1)

# Build the real model!
trains_model_tuned <- trains_model %>%
  # I checked, and, when I wrote this, there were a total of 558 unique
  # non-stop-words in the data. Looking at most at the top-100 of those seems
  # fine, so let's not let it go above that.
  update(
    mtry = second_tune$mtry,
    trees = second_tune$trees,
    min_n = second_tune$min_n,
    tree_depth = second_tune$tree_depth,
    learn_rate = second_tune$learn_rate,
    loss_reduction = second_tune$loss_reduction,
    sample_size = second_tune$sample_size
  )
trains_recipe_tuned <- trains_train %>%
  recipes::recipe(label ~ sentence) %>%
  textrecipes::step_tokenize(sentence) %>%
  textrecipes::step_stem(
    sentence,
    custom_stemmer = textstem::lemmatize_words
  ) %>%
  textrecipes::step_stopwords(sentence) %>%
  textrecipes::step_tokenfilter(
    sentence,
    max_tokens = second_tune$max_tokens
  ) %>%
  textrecipes::step_tf(sentence)
trains_recipe_prepped <- recipes::prep(trains_recipe_tuned, training = trains_train)
trains_train_juiced <- recipes::juice(trains_recipe_prepped)
trains_fit <- parsnip::fit(
  trains_model_tuned,
  label ~ .,
  data = trains_train_juiced
)
saveRDS(trains_fit, "trains_fit.rds")

trains_test_baked <- recipes::bake(
  trains_recipe_prepped,
  new_data = trains_test
)

prediction <- predict(trains_fit, trains_test_baked)
prediction_matrix <- trains_test_baked %>%
  dplyr::select(label) %>%
  dplyr::mutate(
    prediction = prediction$.pred_class
  )

yardstick::kap(
  prediction_matrix,
  truth = label,
  estimate = prediction
)
yardstick::accuracy(
  prediction_matrix,
  truth = label,
  estimate = prediction
)
yardstick::f_meas(
  prediction_matrix,
  truth = label,
  estimate = prediction
)
yardstick::conf_mat(
  prediction_matrix,
  truth = label,
  estimate = prediction
)


# With Trains -------------------------------------------------------------

trains_mentions_trains

trains_test_baked_mentions_trains <- recipes::bake(
  trains_recipe_prepped,
  new_data = trains_mentions_trains
)

prediction_mentions <- predict(trains_fit, trains_test_baked_mentions_trains)
prediction_matrix_mentions <- trains_test_baked_mentions_trains %>%
  dplyr::select(label) %>%
  dplyr::mutate(
    prediction = prediction_mentions$.pred_class
  )

yardstick::kap(
  prediction_matrix_mentions,
  truth = label,
  estimate = prediction
)
yardstick::accuracy(
  prediction_matrix_mentions,
  truth = label,
  estimate = prediction
)
yardstick::f_meas(
  prediction_matrix_mentions,
  truth = label,
  estimate = prediction
)
yardstick::conf_mat(
  prediction_matrix_mentions,
  truth = label,
  estimate = prediction
)
