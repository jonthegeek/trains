display_pca_dev <- function(embedding_df,
                            project_vectors = embedding_df,
                            color_field = NULL,
                            disambiguate_tokens = TRUE,
                            hide = NULL) {
  num_rows <- nrow(embedding_df)
  num_rows_proj <- nrow(project_vectors)
  if (num_rows_proj < 3) {
    stop("At least three vectors are required for a ",
         "meaningful PCA plot.")
  }
  # Use just the indicated vectors to select the PCA projection
  proj_mat <- as.matrix(
    dplyr::select(project_vectors,
                  dplyr::matches("V[0-9]+"))
  )
  pcs <- stats::prcomp(proj_mat,
                       retx = TRUE, center = TRUE, scale. = TRUE, rank. = 2L)
  # pcs$rotation is the projection matrix.

  if (disambiguate_tokens) {
    embedding_df <- dplyr::mutate(embedding_df,
                                  token = paste(token,
                                                sequence_index,
                                                token_index,
                                                sep = "."))
  }
  # Just keep all the non-vector columns, for possible use in plotting.
  tok_labels <- dplyr::select(embedding_df,
                              -dplyr::matches("V[0-9]+"))

  vec_mat <- as.matrix(
    dplyr::select(embedding_df,
                  dplyr::matches("V[0-9]+"))
  )
  # instead of doing PCA here, do the scaling and projection manually
  # pcs <- stats::prcomp(vec_mat,
  #                      retx = TRUE, center = TRUE, scale. = TRUE, rank. = 2L)
  vec_mat <- scale(vec_mat, center = pcs$center, scale = pcs$scale)
  projected <- vec_mat %*% pcs$rotation

  pc_tbl <- dplyr::bind_cols(tok_labels, tibble::as_tibble(projected))

  class <- rep("a", num_rows)
  if (!is.null(color_field)) {
    if (color_field %in% names(embedding_df)) {
      class <- dplyr::pull(
        dplyr::select(embedding_df, dplyr::one_of(color_field))
      )
    } else {
      warning("Column ", color_field, " not found in input table." )
    }
  }
  class <- as.factor(class)

  unique_classes <- unique(class)
  num_class <- length(unique_classes)
  getPalette <- grDevices::colorRampPalette(
    RColorBrewer::brewer.pal(8, "Dark2")
  )
  pal <- getPalette(num_class)
  class_colors <- pal
  names(class_colors) <- unique_classes

  if (!is.null(hide)) {
    if (length(hide) == nrow(pc_tbl)) {
      pc_tbl <- pc_tbl[!hide, ]
      class <- class[!hide]
    } else {
      warning("Length of hide parameter doesn't match size of ",
              "input table, and will be ignored.")
    }
  }

  ggp <- ggplot2::ggplot(pc_tbl, ggplot2::aes(x = PC1, y = PC2,
                                              label = token,
                                              col = class)) +
    ggplot2::scale_color_manual(values = class_colors,
                                name = color_field) +
    ggplot2::geom_text(vjust = 0, nudge_y = 0.5) +
    ggplot2::geom_point()
  if (num_class <= 1) {
    ggp <- ggp + ggplot2::theme(legend.position = "none")
  }
  return(ggp)
}



