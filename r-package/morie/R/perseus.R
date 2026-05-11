#' Build a Perseus agent prompt
#'
#' @param question User question.
#' @param context Optional context string.
#' @return Character scalar prompt.
#' @export
build_prompt <- function(question, context = NULL) {
  question <- trimws(as.character(question)[1])
  if (!nzchar(question)) {
    stop("`question` must be non-empty.", call. = FALSE)
  }

  if (is.null(context) || !nzchar(trimws(as.character(context)[1]))) {
    return(question)
  }

  paste0(
    "Context:\n", trimws(as.character(context)[1]),
    "\n\nQuestion:\n", question
  )
}

#' Query Perseus via Python
#'
#' @param question User question.
#' @param context Optional context string.
#' @param python_bin Python executable to use. Defaults to `MORIE_PYTHON_BIN` or `python3`.
#' @return Agent text response.
#' @export
ask_percy <- function(question, context = NULL, python_bin = Sys.getenv("MORIE_PYTHON_BIN", "python3")) {
  prompt <- build_prompt(question, context = context)

  code <- paste(
    "import json, sys",
    "from morie.perseus import ask_percy",
    "payload = ask_percy(question=sys.argv[1])",
    "print(payload['output_text'])",
    sep = "; "
  )

  out <- system2(
    python_bin,
    c("-c", shQuote(code), shQuote(prompt)),
    stdout = TRUE,
    stderr = TRUE
  )

  status <- attr(out, "status")
  if (!is.null(status) && status != 0) {
    stop(
      "Perseus agent call failed. Ensure the Python MORIE package is available ",
      "and an LLM provider is configured.",
      call. = FALSE
    )
  }

  paste(out, collapse = "\n")
}

build_assistant_prompt <- build_prompt
morie_assistant_query <- ask_percy
