badwords <- hunspell::hunspell(badtext)[[1]]
if (length(badwords) == 0) {
  return(badtext) # no spelling errors that we can find
}
goodwords <- hunspell::hunspell_suggest(badwords) %>%
  vapply(function(x) {
    if (length(x) >= 1) {
      return(x[[1]])
    } else {
      return(NA_character_) # if no correct candidate found...
    }
  }, FUN.VALUE = character(1))
# ...just put unknown words back in.
goodwords[is.na(goodwords)] <- badwords[is.na(goodwords)]

names(goodwords) <- badwords
replace_in_text(text = badtext, replacements = goodwords, ignore_case = TRUE)

replace_in_text <- function(text, replacements, ignore_case = TRUE) {
  stringr::str_replace_all(
    string = text,
    pattern = stringr::regex(
      precisify_replacements(replacements),
      ignore_case = ignore_case
    )
  )
}
