\name{gpRPostmave}
\alias{gpRPostmave}
\title{Compute moving average in columns in a window [i+1:i+w]}

\description{
    This function computes moving average of in columns in a window [i+1:i+w].
}

\usage{
    gpRPostmave(points,w)
}

\arguments{
    \item{points}{a matrix of floating point numbers in which each column is a vector.}
	\item{w}{window width of moving average e.g. for w=2 for i position (value[i+1]+value[i+2])/2}
}

\details{
    Compute moving average in columns
}

\value{
    a matrix of floating point numbers in which each column is a vector
}

\seealso{
    gpRPremave
}

\examples{
numVectors <- 10
dimension <- 5
Vectors <- matrix(runif(numVectors*dimension), numVectors, dimension)
gpRPostmave(Vectors,2)
}

\keyword{math}
\keyword{array & algebra}


