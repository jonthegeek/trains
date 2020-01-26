embeddings_dims <- 50 # One of 50, 100, 200, 300
filename <- paste0("glove.6B.", embeddings_dims, "d.txt")
path <- here::here("glove6B", filename)

embeddings <- readr::read_delim(
  path,
  delim = " ",
  quote = "",
  col_names = c(
    "word",
    paste0("v", seq_len(embeddings_dims))
  ),
  col_types = paste0(
    c(
      "c",
      rep("d", embeddings_dims)
    ),
    collapse = ""
  )
)

dplyr::glimpse(embeddings)

