source(here::here("get_data.R"))
# trains_bert <- RBERT::extract_features(
#   trains_data$sentence,
#   model = "bert_base_uncased",
#   features = "output"
# )$output
#
# saveRDS(
#   trains_bert,
#   here::here(
#     "bert",
#     "trains_bert.rds"
#   )
# )
trains_bert <- readRDS(
  here::here(
    "bert",
    "trains_bert.rds"
  )
)

trains_bert_cls <- trains_bert %>%
  dplyr::filter(token == "[CLS]") %>%
  tidyr::pivot_wider(
    names_from = layer_index,
    values_from = tidyselect::starts_with("V")
  ) %>%
  # Everything has 1 segment_index and 1 token_index, so drop those. And all of
  # the tokens are "[CLS]".
  dplyr::select(-segment_index, -token_index, -token)

# Recombine.
trains_data_bert <- trains_data %>%
  dplyr::mutate(
    sequence_index = dplyr::row_number(),
    # I want to stratify by both label and mentions_train, so make a combined
    # variable.
    combined_label = factor(paste(label, mentions_train, sep = "_"))
  ) %>%
  dplyr::left_join(trains_bert_cls, by = "sequence_index")

# Get ready to model.
set.seed(424242)
trains_split <- rsample::initial_split(
  trains_data_bert,
  strata = combined_label
)
trains_train <- rsample::training(trains_split)
trains_test <- rsample::testing(trains_split)

trains_folds <- rsample::vfold_cv(
  trains_train,
  strata = combined_label
)

# Recipes
trains_recipe <- trains_train %>%
  recipes::recipe() %>%
  recipes::update_role(
    tidyselect::starts_with("V"),
    new_role = "predictor"
  ) %>%
  recipes::update_role(
    "label",
    new_role = "outcome"
  )

# I'm not tuning anything in the recipe, so I might as well get those objects
# ready.
trains_recipe_prepped <- recipes::prep(
  trains_recipe,
  training = trains_train
)
trains_train_juiced <- recipes::juice(trains_recipe_prepped)

# Workflow

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

trains_workflow <- workflows::workflow() %>%
  workflows::add_recipe(trains_recipe) %>%
  workflows::add_model(trains_model)

trains_parameters <- dials::parameters(trains_workflow) %>%
  # I didn't see dials::finalize() before. Maybe it didn't exist? Need to try this out.
  # dials::finalize()
  update(
    mtry = dials::mtry_long(c(0, 3)),
    sample_size = dials::sample_prop(0:1)
  )

# Tune!

all_cores <- parallel::detectCores(logical = FALSE)
library(doParallel)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
trains_grid <- dials::grid_latin_hypercube(trains_parameters, size = 30)
trains_search_result <- tune::tune_grid(
  trains_workflow,
  resamples = trains_folds,
  grid = trains_grid
  # param_info = trains_parameters
  # initial = 5,
  # iter = 30,
  # perf = yardstick::metric_set(yardstick::roc_auc)
  # control = tune::Bayes_control(verbose = TRUE)
)

trains_search_result_bayes <- tune::tune_bayes(
  trains_workflow,
  resamples = trains_folds,
  param_info = trains_parameters,
  initial = trains_search_result,
  # initial = 5,
  iter = 30
)
parallel::stopCluster(cl)

# Gah, the object is 553.5MB! Don't save that!

# saveRDS(
#   trains_search_result_bayes,
#   here::here("bert", "trains_search_result_bayes.rds")
# )
tune::show_best(trains_search_result_bayes, "roc_auc")
best_params <- tune::select_best(trains_search_result_bayes, "roc_auc")
final_bert_model <- tune::finalize_model(trains_model, best_params)
final_bert_workflow <- tune::finalize_workflow(trains_workflow,)

set.seed(424242)
trains_fit_bert <- parsnip::fit(
  final_bert_workflow,
  data = trains_train_juiced
)
saveRDS(
  trains_fit_bert,
  here::here("bert", "trains_fit.rds")
)

trains_test_baked <- recipes::bake(
  trains_recipe_prepped,
  new_data = trains_test
)

prediction <- predict(trains_fit_bert, trains_test_baked)
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
