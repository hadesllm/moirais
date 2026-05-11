\name{ask_percy}
\alias{ask_percy}
\title{Ask the Perseus assistant a question}
\description{
  Send a free-form question to the Perseus assistant via the bundled
  Python LLM bridge. Returns the assistant's text response.
}
\usage{
  ask_percy(question, context = NULL,
            python_bin = Sys.getenv("MORIE_PYTHON_BIN", "python3"))
}
\arguments{
  \item{question}{Character string. The natural-language question.}
  \item{context}{Optional named list of context variables (data summaries,
    column descriptions, etc.) appended to the prompt.}
  \item{python_bin}{Path to the Python interpreter that has the
    \pkg{morie} Python package installed. Defaults to
    \code{$MORIE_PYTHON_BIN} or \code{python3}.}
}
\value{
  Character string containing the assistant's response.
}
\keyword{internal}
