\name{build_prompt}
\alias{build_prompt}
\title{Build a structured prompt for the Perseus assistant}
\description{
  Compose a question and optional context into the structured prompt
  format expected by \code{\link{ask_percy}}.
}
\usage{
  build_prompt(question, context = NULL)
}
\arguments{
  \item{question}{Character string. The natural-language question.}
  \item{context}{Optional named list of context variables.}
}
\value{
  Character string. The composed prompt.
}
\keyword{internal}
