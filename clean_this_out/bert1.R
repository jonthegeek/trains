bert_dir <- RBERT::download_BERT_checkpoint("bert_base_uncased")

vocab_file <- file.path(bert_dir, "vocab.txt")
init_checkpoint <- file.path(bert_dir, "bert_model.ckpt")
bert_config_file <- file.path(bert_dir, "bert_config.json")

chicken <- c("The chicken didn't cross the road because it was too tired.",
             "The chicken didn't cross the road because it was too wide.")
chicken_ex <- RBERT::make_examples_simple(chicken)

feats_chicken <- RBERT::extract_features(examples = chicken_ex,
                                         vocab_file = vocab_file,
                                         bert_config_file = bert_config_file,
                                         init_checkpoint = init_checkpoint,
                                         layer_indexes = 1:12,
                                         batch_size = 2L,
                                         features = "attention")
RBERTviz::visualize_attention(feats_chicken$attention, sequence_index = 1)
RBERTviz::visualize_attention(feats_chicken$attention, sequence_index = 2)

tuba <- c("The object didn't fit in the case because it was too big.",
          "The object didn't fit in the case because it was too small.")
tuba_ex <- RBERT::make_examples_simple(tuba)

feats_tuba <- RBERT::extract_features(examples = tuba_ex,
                                         vocab_file = vocab_file,
                                         bert_config_file = bert_config_file,
                                         init_checkpoint = init_checkpoint,
                                         layer_indexes = 0:12,
                                         batch_size = 2L,
                                         features = "attention")
RBERTviz::visualize_attention(feats_tuba$attention, sequence_index = 1)
RBERTviz::visualize_attention(feats_tuba$attention, sequence_index = 2)

library(magrittr)
weight_multiplier <- 5
feats_chicken$attention %>%
  dplyr::filter(
    token == "it"
  ) %>%
  # dplyr::group_by(sequence_index, layer_index, head_index) %>%
  # dplyr::mutate(max_attention = max(attention_weight)) %>%
  # dplyr::ungroup() %>%
  dplyr::filter(
    attention_token %in% c("chicken", "road")
  ) %>%
  dplyr::select(
    sequence_index,
    token,
    layer_index,
    head_index,
    attention_token,
    attention_weight
    # ,
    # max_attention
  ) %>%
  # View()
  dplyr::mutate(layer_index = layer_index - 1L) %>%
  # dplyr::filter(
  #   layer_index == 10, head_index == 9
  # )
  tidyr::pivot_wider(
    names_from = attention_token,
    values_from = attention_weight
  ) %>%
  dplyr::filter(
    (sequence_index == 1 & chicken > 0.1 & road < 0.1) |
      (sequence_index == 2 & road > 0.1 & chicken < 0.1)
  ) %>%
  dplyr::count(layer_index, head_index) %>%
  dplyr::filter(n == 2)

# weight_multiplier <- 1.1
# feats_tuba$attention %>%
#   dplyr::filter(
#     token == "it"
#   ) %>%
#   # dplyr::group_by(sequence_index, layer_index, head_index) %>%
#   # dplyr::mutate(max_attention = max(attention_weight)) %>%
#   # dplyr::ungroup() %>%
#   dplyr::filter(
#     attention_token %in% c("object", "case")
#   ) %>%
#   dplyr::select(
#     sequence_index,
#     token,
#     layer_index,
#     head_index,
#     attention_token,
#     attention_weight
#     # ,
#     # max_attention
#   ) %>%
#   # View()
#   dplyr::mutate(layer_index = layer_index - 1L) %>%
#   tidyr::pivot_wider(
#     names_from = attention_token,
#     values_from = attention_weight
#   ) %>%
#   dplyr::filter(
#     (sequence_index == 1 & object > 0.1 & case < 0.1) |
#       (sequence_index == 2 & case > 0.1 & object < 0.1)
#   ) %>%
#   dplyr::count(layer_index, head_index) %>%
#   dplyr::filter(n == 2)
#
# max(feats_tuba$attention$layer_index)


dog <- c("The dog fetched the ball. It was excited.",
         # "The dog fetched the ball. It was orange.",
         # "The dog fetched the ball. It was brown.",
         # "The dog fetched the ball. It was striped.",
         # "The dog fetched the ball. It was panting.",
         # "The dog fetched the ball. It was fuzzy.",
         # "The ball was fetched by the dog. It was excited.",
         # "I know a man with a wooden leg named Bob.",
         # "Bob gave Jim an apple. He had extra.",
         "I saw a presentation this morning. It was sunny."
         )
dog_ex <- RBERT::make_examples_simple(dog)

feats_dog <- RBERT::extract_features(examples = dog_ex,
                                      vocab_file = vocab_file,
                                      bert_config_file = bert_config_file,
                                      init_checkpoint = init_checkpoint,
                                      layer_indexes = 0:12,
                                      batch_size = 2L,
                                      features = "attention")
# RBERTviz::visualize_attention(feats_dog$attention, sequence_index = 1)
RBERTviz::visualize_attention(feats_dog$attention, sequence_index = 2)
RBERTviz::visualize_attention(feats_dog$attention, sequence_index = 3)
RBERTviz::visualize_attention(feats_dog$attention, sequence_index = 4)
RBERTviz::visualize_attention(feats_dog$attention, sequence_index = 5)
RBERTviz::visualize_attention(feats_dog$attention, sequence_index = 6)
feats_dog$attention %>%
  dplyr::filter(
    token %in% c("far", "bored")
  ) %>%
  # dplyr::group_by(sequence_index, layer_index, head_index) %>%
  # dplyr::mutate(max_attention = max(attention_weight)) %>%
  # dplyr::ungroup() %>%
  dplyr::filter(
    attention_token %in% c("dog", "ball")
  ) %>%
  dplyr::select(
    sequence_index,
    token,
    layer_index,
    head_index,
    attention_token,
    attention_weight
    # ,
    # max_attention
  ) %>%
  # View()
  dplyr::mutate(layer_index = layer_index - 1L) %>%
  tidyr::pivot_wider(
    names_from = attention_token,
    values_from = attention_weight
  ) %>%
  # View()
  dplyr::filter(
    (sequence_index == 1 & dog > 0.1 & ball < 0.1) |
      (sequence_index == 2 & ball > 0.1 & dog < 0.1)
  ) %>%
  dplyr::count(layer_index, head_index) %>%
  dplyr::filter(n == 2)


feats_tacos <- RBERT::extract_features(
  examples = RBERT::make_examples_simple("I love tacos."),
  vocab_file = vocab_file,
  bert_config_file = bert_config_file,
  init_checkpoint = init_checkpoint,
  layer_indexes = 1:12,
  batch_size = 2L,
  features = "attention")
RBERTviz::visualize_attention(feats_tacos$attention)
