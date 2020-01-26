trains_w_examples <- trains_data %>%
  dplyr::mutate(
    examples = RBERT::make_examples_simple(sentence)
  )

feats_trains <- RBERT::extract_features(examples = trains_w_examples$examples,
                                        vocab_file = vocab_file,
                                        bert_config_file = bert_config_file,
                                        init_checkpoint = init_checkpoint,
                                        layer_indexes = 0:12,
                                        batch_size = 2L)
train_embeddings_df <- trains_w_examples %>%
  tidyr::hoist(
    examples,
    sequence_index = "unique_id"
  ) %>%
  dplyr::filter(mentions_train) %>%
  dplyr::inner_join(
    feats_trains$output %>%
      dplyr::filter(stringr::str_detect(token, "^train")),
    by = "sequence_index"
  ) %>%
  dplyr::arrange(sequence_index, layer_index) %>%
  dplyr::select(
    -examples, -sentence, -mentions_train
  )

pca_plot <- train_embeddings_df %>%
  display_pca_dev(
    project_vectors = dplyr::filter(train_embeddings_df, layer_index %in% c(0, 12)),
    color_field = "label"
  )

train_embeddings_df %>%
  dplyr::filter(layer_index == 12) %>%
  display_pca_dev(
    project_vectors = dplyr::filter(train_embeddings_df, layer_index %in% c(0, 12)),
    color_field = "label",
    disambiguate_tokens = FALSE
  ) +
  ggplot2::scale_y_continuous(limits = c(-20, 25)) +
  ggplot2::scale_x_continuous(limits = c(-25, 20)) +
  NULL



anim <- pca_plot + gganimate::transition_states(layer_index,
                                                transition_length = 1,
                                                state_length = 0,
                                                wrap = FALSE) +
  ggplot2::theme(plot.caption = ggplot2::element_text(
    size = 30,
    hjust = 0,
    margin = ggplot2::margin(-2, 0, 1, 0, "cm")
  )) +
  ggplot2::labs(caption = " layer: {closest_state}")

fps <- 15
pause_time <- 2
time_per_layer <- 1
nframes <- n_layers*fps*time_per_layer + 2*pause_time*fps
gganimate::animate(anim, nframes = nframes, fps = fps,
                   start_pause = pause_time*fps,
                   end_pause = pause_time*fps)


train_embeddings_df <- trains_w_examples %>%
  tidyr::hoist(
    examples,
    sequence_index = "unique_id"
  ) %>%
  dplyr::filter(mentions_train) %>%
  dplyr::inner_join(
    feats_trains$output %>%
      dplyr::filter(token_index == 3),
    by = "sequence_index"
  ) %>%
  dplyr::arrange(sequence_index, layer_index) %>%
  dplyr::select(
    -examples, -sentence, -mentions_train
  )

pca_plot <- train_embeddings_df %>%
  display_pca_dev(
    color_field = "label"
  )
anim <- pca_plot + gganimate::transition_states(layer_index,
                                                transition_length = 1,
                                                state_length = 0,
                                                wrap = FALSE) +
  ggplot2::theme(plot.caption = ggplot2::element_text(
    size = 30,
    hjust = 0,
    margin = ggplot2::margin(-2, 0, 1, 0, "cm")
  )) +
  ggplot2::labs(caption = " layer: {closest_state}")

fps <- 15
pause_time <- 2
time_per_layer <- 1
nframes <- n_layers*fps*time_per_layer + 2*pause_time*fps
gganimate::animate(anim, nframes = nframes, fps = fps,
                   start_pause = pause_time*fps,
                   end_pause = pause_time*fps)
