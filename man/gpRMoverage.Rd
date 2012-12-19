\name{gpRMoverage}
\alias{gpRMoverage}
\title{Compute double moving average with GPU}

\description{
    This function computes two moving average of in columns in windows [i-1:i-w] [i+1:i+w] and divides their sum by 2.
}

\usage{
    gpRMoverage(points,w)
}

\arguments{
    \item{points}{a matrix of floating point numbers in which each column is a vector.}
	\item{w}{window width of double moving average}
}

\details{
    Compute moving average in columns
}

\value{
    a matrix of floating point numbers in which each column is a vector
}

\seealso{
    gpRPostmave,gpRPremave
}

\examples{
numVectors <- 10
dimension <- 5
Vectors <- matrix(runif(numVectors*dimension), numVectors, dimension)
gpRMoverage(Vectors,2)
}

\keyword{math}
\keyword{array & algebra}

